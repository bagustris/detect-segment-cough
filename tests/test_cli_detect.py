"""Test the detect CLI module."""

from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

import pytest

from coughkit.cli.detect import detect, add_arguments, build_parser

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "sample_recordings"


class TestDetect:
    """Test the detect function."""

    def test_detect_cough_file(self, capsys):
        """Test detecting a cough file."""
        prob = detect(str(SAMPLES / "cough.wav"))
        
        assert isinstance(prob, (float, np.floating))
        assert 0.0 <= prob <= 1.0
        assert prob > 0.5  # Should be high for actual cough
        
        out = capsys.readouterr().out
        assert "probability of cough" in out
        assert str(SAMPLES / "cough.wav") in out

    def test_detect_not_cough_file(self, capsys):
        """Test detecting a non-cough file."""
        prob = detect(str(SAMPLES / "not_cough.wav"))
        
        assert isinstance(prob, (float, np.floating))
        assert 0.0 <= prob <= 1.0
        assert prob < 0.5  # Should be low for non-cough
        
        out = capsys.readouterr().out
        assert "probability of cough" in out

    def test_detect_artif_cough_file(self, capsys):
        """Test detecting an artificial cough file."""
        prob = detect(str(SAMPLES / "artif-cough.wav"))
        
        assert isinstance(prob, (float, np.floating))
        assert 0.0 <= prob <= 1.0
        
        out = capsys.readouterr().out
        assert "probability of cough" in out

    def test_detect_missing_file_raises(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Input audio file not found"):
            detect("nonexistent.wav")

    @patch('coughkit.cli.detect.wavfile.read')
    @patch('coughkit.cli.detect.classify_cough')
    @patch('coughkit.cli.detect.load_scaler')
    @patch('coughkit.cli.detect.load_cough_classifier')
    def test_detect_with_mocked_audio(self, mock_model, mock_scaler, mock_classify, mock_read):
        """Test detect with mocked audio processing."""
        # Setup mocks - wavfile.read returns (sample_rate, data)
        mock_read.return_value = (16000, np.array([0.1, 0.2, 0.3]))
        mock_classify.return_value = 0.75
        
        prob = detect(str(SAMPLES / "cough.wav"))
        
        assert prob == 0.75
        mock_read.assert_called_once()
        mock_classify.assert_called_once()

    @patch('coughkit.cli.detect.wavfile.read')
    @patch('coughkit.cli.detect.classify_cough')
    @patch('coughkit.cli.detect.load_scaler')
    @patch('coughkit.cli.detect.load_cough_classifier')
    def test_detect_handles_zero_probability(self, mock_model, mock_scaler, mock_classify, mock_read):
        """Test detect when classifier returns zero probability."""
        mock_read.return_value = (16000, np.array([0.0, 0.0, 0.0]))
        mock_classify.return_value = 0.0
        
        prob = detect(str(SAMPLES / "cough.wav"))
        
        assert prob == 0.0

    @patch('coughkit.cli.detect.wavfile.read')
    @patch('coughkit.cli.detect.classify_cough')
    @patch('coughkit.cli.detect.load_scaler')
    @patch('coughkit.cli.detect.load_cough_classifier')
    def test_detect_handles_high_probability(self, mock_model, mock_scaler, mock_classify, mock_read):
        """Test detect when classifier returns high probability."""
        mock_read.return_value = (16000, np.array([0.5, 0.7, 0.9]))
        mock_classify.return_value = 0.95
        
        prob = detect(str(SAMPLES / "cough.wav"))
        
        assert prob == 0.95

    @patch('coughkit.cli.detect.load_scaler')
    @patch('coughkit.cli.detect.load_cough_classifier')
    @patch('coughkit.cli.detect.wavfile.read')
    def test_detect_handles_read_error(self, mock_read, mock_model, mock_scaler):
        """Test that read errors are handled properly."""
        mock_read.side_effect = IOError("Cannot read file")
        
        with pytest.raises(IOError):
            detect(str(SAMPLES / "cough.wav"))


class TestCLIDetect:
    """Test CLI argument parsing for detect command."""

    def test_build_parser_creates_valid_parser(self):
        parser = build_parser(prog='cough-detect')
        assert parser is not None
        assert parser.prog == 'cough-detect'

    def test_parser_file_argument(self):
        parser = build_parser(prog='cough-detect')
        args = parser.parse_args(['-f', 'test.wav'])
        assert args.file == 'test.wav'

    def test_parser_with_long_argument(self):
        parser = build_parser(prog='cough-detect')
        args = parser.parse_args(['--file', 'test.wav'])
        assert args.file == 'test.wav'

    def test_parser_requires_file(self):
        parser = build_parser(prog='cough-detect')
        with pytest.raises(SystemExit):
            parser.parse_args([])  # Should fail without required -f argument

    def test_parser_help_shows_description(self):
        parser = build_parser(prog='cough-detect')
        help_text = parser.format_help()
        assert "detect" in help_text.lower()
        assert "probability" in help_text.lower()