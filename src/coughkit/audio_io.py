"""Audio I/O helpers: COUGHVID dataset conversion."""

import importlib
import subprocess
import sys
from pathlib import Path


def _require(module_name, install_hint):
    """Import an optional dependency, exiting with an actionable message if absent."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        sys.exit(f"{module_name} is not installed. Run: pip install {install_hint}")


def convert_files(folder):
    """Convert files from .webm and .ogg to .wav.

    folder: path to the COUGHVID database and ``metadata_compiled.csv``.
    Requires ffmpeg on PATH and the optional ``train`` extra (pandas):
    ``pip install coughkit[train]``.
    """
    pd = _require("pandas", "coughkit[train]")
    folder_path = Path(folder)

    df = pd.read_csv(folder_path / "metadata_compiled.csv")
    names_to_convert = df.uuid.to_numpy()
    for counter, name in enumerate(names_to_convert):
        if counter % 1000 == 0:
            print("Finished {0}/{1}".format(counter, len(names_to_convert)))

        wav_path = folder_path / f"{name}.wav"
        source_path = next(
            (folder_path / f"{name}{suffix}"
             for suffix in (".webm", ".ogg")
             if (folder_path / f"{name}{suffix}").is_file()),
            None,
        )
        if source_path is None:
            print("Error: No file name {0}".format(name))
            continue

        subprocess.run(["ffmpeg", "-i", str(source_path), str(wav_path)],
                       check=True)
