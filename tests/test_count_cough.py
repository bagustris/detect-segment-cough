from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

import pytest

from coughkit.cli.count import count, _count_mic_streaming, add_arguments, build_parser

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "sample_recordings"


class TestCount:
    def test_cough_file_returns_one(self):
        assert count(input_file=str(SAMPLES / "cough.wav")) == 1

    def test_not_cough_file_returns_zero(self):
        assert count(input_file=str(SAMPLES / "not_cough.wav")) == 0

    def test_artif_cough_returns_one(self):
        assert count(input_file=str(SAMPLES / "artif-cough.wav")) == 1

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            count(input_file="nonexistent.wav")

    def test_custom_threshold_zero_accepts_all_segments(self):
        # With threshold=0 every segment passes the classifier; must be >= 1
        assert count(input_file=str(SAMPLES / "cough.wav"), threshold=0.0) >= 1

    def test_custom_threshold_one_rejects_all_segments(self):
        # Nothing ever reaches p=1.0
        assert count(input_file=str(SAMPLES / "cough.wav"), threshold=1.0) == 0

    def test_verbose_does_not_raise(self, capsys):
        count(input_file=str(SAMPLES / "cough.wav"), verbose=True)
        out = capsys.readouterr().out
        assert "p=" in out

    def test_custom_sampling_rate(self):
        # A rate at or above the classifier's internal 12 kHz target should
        # still detect the cough reliably.
        assert count(input_file=str(SAMPLES / "cough.wav"), fs_out=22050) == 1

    def test_low_sampling_rate_warns(self):
        # Below the classifier's internal 12 kHz target, accuracy degrades;
        # this should be surfaced to the caller rather than fail silently.
        with pytest.warns(UserWarning, match="fs_out=8000"):
            count(input_file=str(SAMPLES / "cough.wav"), fs_out=8000)

    def test_no_input_file_and_no_mic_raises(self):
        with pytest.raises(ValueError, match="Provide input_file or set use_mic=True"):
            count(input_file=None, use_mic=False)


class TestMicStreaming:
    """Test the real-time microphone streaming functionality."""

    @patch('coughkit.cli.count.mic_stream')
    @patch('coughkit.cli.count.load_cough_classifier')
    @patch('coughkit.cli.count.load_scaler')
    @patch('coughkit.cli.count.classify_cough')
    def test_mic_streaming_detects_cough(self, mock_classify, mock_scaler, mock_model, mock_mic):
        # Mock the microphone stream to provide cough-like audio
        fs_out = 16000
        chunk_size = int(0.1 * fs_out)  # 100ms chunks
        
        # Create chunks: silence, then cough burst, then silence
        silence = np.zeros(chunk_size, dtype=np.float32)
        cough_burst = np.random.randn(chunk_size * 3).astype(np.float32) * 0.5
        cough_burst /= np.max(np.abs(cough_burst))
        
        chunks = [silence] * 5 + [cough_burst] + [silence] * 5
        mock_mic.return_value = chunks
        
        # Mock classifier to return high probability for cough
        mock_classify.return_value = 0.9
        
        result = _count_mic_streaming(
            duration=None,
            fs_out=fs_out,
            threshold=0.5,
            verbose=False,
            show_mic_level=False
        )
        
        assert result >= 1  # Should detect at least one cough

    @patch('coughkit.cli.count.mic_stream')
    @patch('coughkit.cli.count.load_cough_classifier')
    @patch('coughkit.cli.count.load_scaler')
    @patch('coughkit.cli.count.classify_cough')
    def test_mic_streaming_bounds_event_against_realistic_ambient_noise(
        self, mock_classify, mock_scaler, mock_model, mock_mic
    ):
        # A fixed low background-power seed (far below a real room's ambient
        # power) used to latch the detector open for the whole session: once
        # a chunk exceeded it, bg_power was never updated again, so the low
        # threshold for closing the event became unreachable, and event_buf
        # grew for the entire session - one classification no matter how
        # many coughs actually occurred. Calibrating bg_power from leading
        # ambient audio fixes onset detection, but note: for genuinely
        # stationary ambient noise the _LOW_MULT "silence" threshold stays
        # effectively unreachable (per-chunk power sits within a few percent
        # of its own mean, never 10x below it), so closure in practice comes
        # from the _MAX_EVENT safety cap rather than silence detection. This
        # test checks the guarantee that fix actually provides: bounded
        # growth, not that the event closes right after the burst.
        fs_out = 16000
        chunk_size = int(0.1 * fs_out)
        rng = np.random.default_rng(0)

        def ambient_chunk():
            return rng.standard_normal(chunk_size).astype(np.float32) * 0.01

        cough_burst = rng.standard_normal(chunk_size * 3).astype(np.float32) * 0.5
        cough_burst /= np.max(np.abs(cough_burst))

        chunks = ([ambient_chunk() for _ in range(5)] + [cough_burst]
                  + [ambient_chunk() for _ in range(5)])
        mock_mic.return_value = chunks

        received_lens = []

        def fake_classify(audio, fs, model, scaler):
            received_lens.append(len(audio))
            return 0.9

        mock_classify.side_effect = fake_classify

        result = _count_mic_streaming(
            duration=None, fs_out=fs_out, threshold=0.5,
            verbose=False, show_mic_level=False,
        )

        from coughkit.cli.count import _MAX_EVENT

        assert result == 1
        assert len(received_lens) == 1  # classified once, not once per ambient chunk
        # Bounded by the safety cap, not left to grow for the whole stream/session.
        assert received_lens[0] <= int(_MAX_EVENT * fs_out)

    @patch('coughkit.cli.count.mic_stream')
    @patch('coughkit.cli.count.load_cough_classifier')
    @patch('coughkit.cli.count.load_scaler')
    @patch('coughkit.cli.count.classify_cough')
    def test_mic_streaming_with_duration(self, mock_classify, mock_scaler, mock_model, mock_mic):
        # Test duration limiting: mic_stream is unbounded (like the real
        # generator), so only the duration cutoff should stop consumption.
        fs_out = 16000
        chunk_size = int(0.1 * fs_out)
        silence = np.zeros(chunk_size, dtype=np.float32)

        consumed = []

        def endless_silence(**kwargs):
            while True:
                consumed.append(1)
                yield silence

        mock_mic.side_effect = endless_silence
        mock_classify.return_value = 0.0  # No coughs

        result = _count_mic_streaming(
            duration=0.5,  # Should stop after 0.5 seconds = 5 chunks
            fs_out=fs_out,
            threshold=0.5,
            verbose=False,
            show_mic_level=False
        )

        assert result == 0  # No coughs detected
        assert len(consumed) == 5  # exactly 0.5s / 0.1s chunks consumed

    @patch('coughkit.cli.count.mic_stream')
    @patch('coughkit.cli.count.load_cough_classifier')
    @patch('coughkit.cli.count.load_scaler')
    @patch('coughkit.cli.count.classify_cough')
    def test_mic_streaming_zero_duration_stops_immediately(self, mock_classify, mock_scaler, mock_model, mock_mic):
        # duration=0 is falsy, not "no limit" - must still cut off after the
        # first chunk rather than streaming forever.
        fs_out = 16000
        chunk_size = int(0.1 * fs_out)
        silence = np.zeros(chunk_size, dtype=np.float32)

        consumed = []

        def endless_silence(**kwargs):
            while True:
                consumed.append(1)
                yield silence

        mock_mic.side_effect = endless_silence
        mock_classify.return_value = 0.0

        result = _count_mic_streaming(
            duration=0,
            fs_out=fs_out,
            threshold=0.5,
            verbose=False,
            show_mic_level=False
        )

        assert result == 0
        assert len(consumed) == 1  # stopped after the first chunk, not unbounded

    @patch('coughkit.cli.count.mic_stream')
    @patch('coughkit.cli.count.load_cough_classifier')
    @patch('coughkit.cli.count.load_scaler')
    @patch('coughkit.cli.count.classify_cough')
    def test_mic_streaming_caps_runaway_event(self, mock_classify, mock_scaler, mock_model, mock_mic):
        # A sustained loud event that never drops to silence must still be
        # force-classified periodically (the _MAX_EVENT safety cap) instead
        # of buffering for the whole session.
        from coughkit.cli.count import _MAX_EVENT

        fs_out = 16000
        chunk_size = int(0.1 * fs_out)
        ambient = np.zeros(chunk_size, dtype=np.float32)
        loud = np.ones(chunk_size, dtype=np.float32) * 0.9

        chunks = [ambient] * 5 + [loud] * 100  # calibrate quiet, then 10s continuous loud
        mock_mic.return_value = chunks
        mock_classify.return_value = 0.0

        received_lens = []
        mock_classify.side_effect = lambda audio, fs, model, scaler: (
            received_lens.append(len(audio)) or 0.0
        )

        _count_mic_streaming(
            duration=None, fs_out=fs_out, threshold=0.5,
            verbose=False, show_mic_level=False,
        )

        max_event_samples = int(_MAX_EVENT * fs_out)
        assert len(received_lens) > 1  # force-closed more than once
        assert all(n <= max_event_samples for n in received_lens)

    @patch('coughkit.cli.count.mic_stream')
    @patch('coughkit.cli.count.load_cough_classifier')
    @patch('coughkit.cli.count.load_scaler')
    @patch('coughkit.cli.count.classify_cough')
    def test_mic_streaming_threshold_filtering(self, mock_classify, mock_scaler, mock_model, mock_mic):
        # Test that threshold filters correctly
        fs_out = 16000
        chunk_size = int(0.1 * fs_out)
        
        cough_burst = np.random.randn(chunk_size * 3).astype(np.float32) * 0.5
        cough_burst /= np.max(np.abs(cough_burst))
        silence = np.zeros(chunk_size, dtype=np.float32)
        
        chunks = [silence] * 2 + [cough_burst] + [silence] * 2
        mock_mic.return_value = chunks
        
        # Mock classifier to return probability below threshold
        mock_classify.return_value = 0.3
        
        result = _count_mic_streaming(
            duration=None,
            fs_out=fs_out,
            threshold=0.5,  # Higher than mock probability
            verbose=False,
            show_mic_level=False
        )
        
        assert result == 0  # Should not count due to threshold

    @patch('coughkit.cli.count.mic_stream')
    @patch('coughkit.cli.count.load_cough_classifier')
    @patch('coughkit.cli.count.load_scaler')
    @patch('coughkit.cli.count.classify_cough')
    def test_mic_streaming_verbose_output(self, mock_classify, mock_scaler, mock_model, mock_mic, capsys):
        # Test verbose output
        fs_out = 16000
        chunk_size = int(0.1 * fs_out)
        
        cough_burst = np.random.randn(chunk_size * 3).astype(np.float32) * 0.5
        cough_burst /= np.max(np.abs(cough_burst))
        silence = np.zeros(chunk_size, dtype=np.float32)
        
        chunks = [silence] * 2 + [cough_burst] + [silence] * 2
        mock_mic.return_value = chunks
        
        mock_classify.return_value = 0.9
        
        _count_mic_streaming(
            duration=None,
            fs_out=fs_out,
            threshold=0.5,
            verbose=True,
            show_mic_level=False
        )
        
        out = capsys.readouterr().out
        assert "cough detected" in out


class TestCLI:
    """Test CLI argument parsing and main function."""

    def test_build_parser_creates_valid_parser(self):
        parser = build_parser(prog='cough-count')
        assert parser is not None
        assert parser.prog == 'cough-count'

    def test_parser_file_argument(self):
        parser = build_parser(prog='cough-count')
        args = parser.parse_args(['-f', 'test.wav'])
        assert args.file == 'test.wav'
        assert args.mic is False

    def test_parser_mic_mode(self):
        parser = build_parser(prog='cough-count')
        args = parser.parse_args(['-m'])
        assert args.mic is True
        assert args.file is None

    def test_parser_with_threshold(self):
        parser = build_parser(prog='cough-count')
        args = parser.parse_args(['-f', 'test.wav', '-t', '0.7'])
        assert args.threshold == 0.7

    def test_parser_with_duration(self):
        parser = build_parser(prog='cough-count')
        args = parser.parse_args(['-m', '-d', '10'])
        assert args.duration == 10

    def test_parser_with_verbose(self):
        parser = build_parser(prog='cough-count')
        args = parser.parse_args(['-f', 'test.wav', '-v'])
        assert args.verbose is True

    def test_parser_no_mic_level(self):
        parser = build_parser(prog='cough-count')
        args = parser.parse_args(['-m', '--no-mic-level'])
        assert args.no_mic_level is True

    def test_parser_with_custom_fs(self):
        parser = build_parser(prog='cough-count')
        args = parser.parse_args(['-f', 'test.wav', '-fs', '8000'])
        assert args.fs_out == 8000
