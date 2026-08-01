"""Test the segment CLI module."""

from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import numpy as np

import pytest

from coughkit.cli.segment import segment, add_arguments, build_parser

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "sample_recordings"


class TestSegment:
    """Test the segment function."""

    def test_segment_cough_file(self, tmp_path):
        """Test segmenting a cough file."""
        output_dir = tmp_path / "output"
        result = segment(str(SAMPLES / "cough.wav"), str(output_dir))
        
        assert isinstance(result, list)
        assert len(result) > 0
        # Check that files were created
        for path in result:
            assert Path(path).exists()
            assert path.endswith('.wav')

    def test_segment_not_cough_file(self, tmp_path):
        """Test segmenting a non-cough file."""
        output_dir = tmp_path / "output"
        result = segment(str(SAMPLES / "not_cough.wav"), str(output_dir))

        # The segmenter has no cough/non-cough filter (see cli/segment.py),
        # so it still writes one file per detected high-energy region.
        assert isinstance(result, list)
        for path in result:
            assert Path(path).exists()
            assert path.endswith('.wav')

    def test_segment_with_custom_fs(self, tmp_path):
        """Test segmenting with custom sampling rate."""
        output_dir = tmp_path / "output"
        result = segment(str(SAMPLES / "cough.wav"), str(output_dir), fs_out=8000)
        
        assert isinstance(result, list)
        assert len(result) > 0

    def test_segment_missing_file_raises(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Input audio file not found"):
            segment("nonexistent.wav", "./output")

    def test_segment_creates_output_dir(self, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        output_dir = tmp_path / "new_dir" / "nested"
        result = segment(str(SAMPLES / "cough.wav"), str(output_dir))
        
        assert Path(output_dir).exists()
        assert isinstance(result, list)

    def test_segment_default_output_dir(self, tmp_path):
        """Test segmenting with default output directory."""
        # Change to temp directory to avoid cluttering current dir
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = segment(str(SAMPLES / "cough.wav"), "./")
            
            assert isinstance(result, list)
            assert len(result) > 0
            # Check files were created in current directory
            for path in result:
                assert Path(path).exists()
        finally:
            os.chdir(original_cwd)

    @patch('coughkit.cli.segment.sf.write')
    @patch('coughkit.cli.segment.segment_cough')
    @patch('coughkit.cli.segment.librosa.load')
    def test_segment_handles_write_error(self, mock_load, mock_segment, mock_write, tmp_path):
        """Test that write errors are handled properly."""
        # Setup mocks
        signal = np.array([0.1, 0.2, 0.3])
        mock_load.return_value = (signal, 16000)
        mock_segment.return_value = ([np.array([0.1, 0.2])], np.array([True, False]))
        mock_write.side_effect = IOError("Disk full")
        
        output_dir = tmp_path / "output"
        with pytest.raises(IOError):
            segment(str(SAMPLES / "cough.wav"), str(output_dir))


class TestCLISegment:
    """Test CLI argument parsing for segment command."""

    def test_build_parser_creates_valid_parser(self):
        parser = build_parser(prog='cough-segment')
        assert parser is not None
        assert parser.prog == 'cough-segment'

    def test_parser_file_argument(self):
        parser = build_parser(prog='cough-segment')
        args = parser.parse_args(['-f', 'test.wav'])
        assert args.file == 'test.wav'

    def test_parser_with_output_dir(self):
        parser = build_parser(prog='cough-segment')
        args = parser.parse_args(['-f', 'test.wav', '-o', 'output_dir'])
        assert args.output_dir == 'output_dir'

    def test_parser_with_custom_fs(self):
        parser = build_parser(prog='cough-segment')
        args = parser.parse_args(['-f', 'test.wav', '-fs', '8000'])
        assert args.fs_out == 8000

    def test_parser_with_long_arguments(self):
        parser = build_parser(prog='cough-segment')
        args = parser.parse_args(['--file', 'test.wav', '--output_dir', 'out', '--fs_out', '22050'])
        assert args.file == 'test.wav'
        assert args.output_dir == 'out'
        assert args.fs_out == 22050

    def test_parser_requires_file(self):
        parser = build_parser(prog='cough-segment')
        with pytest.raises(SystemExit):
            parser.parse_args([])  # Should fail without required -f argument