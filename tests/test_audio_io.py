from pathlib import Path

from coughkit import audio_io


class _FakeUuidColumn:
    def __init__(self, names):
        self._names = names

    def to_numpy(self):
        return self._names


class _FakeDataFrame:
    def __init__(self, names):
        self.uuid = _FakeUuidColumn(names)


class _FakePandas:
    def __init__(self, names, captured):
        self._names = names
        self._captured = captured

    def read_csv(self, path):
        self._captured["metadata_path"] = Path(path)
        return _FakeDataFrame(self._names)


def test_convert_files_accepts_folder_without_trailing_separator(tmp_path, monkeypatch):
    captured = {}
    (tmp_path / "metadata_compiled.csv").write_text("uuid\nalpha\nbeta\n")
    (tmp_path / "alpha.webm").write_bytes(b"webm")
    (tmp_path / "beta.ogg").write_bytes(b"ogg")

    monkeypatch.setattr(
        audio_io,
        "_require",
        lambda module_name, install_hint: _FakePandas(["alpha", "beta"], captured),
    )
    calls = []

    def fake_run(command, check):
        calls.append((command, check))

    monkeypatch.setattr(audio_io.subprocess, "run", fake_run)

    audio_io.convert_files(str(tmp_path))

    assert captured["metadata_path"] == tmp_path / "metadata_compiled.csv"
    assert calls == [
        (["ffmpeg", "-i", str(tmp_path / "alpha.webm"),
          str(tmp_path / "alpha.wav")], True),
        (["ffmpeg", "-i", str(tmp_path / "beta.ogg"),
          str(tmp_path / "beta.wav")], True),
    ]
