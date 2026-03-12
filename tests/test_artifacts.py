import pytest
from pathlib import Path
from aura.artifacts import FileArtifact, DirectoryArtifact


def test_file_artifact_read_write(tmp_path: Path):
    f = tmp_path / "code.py"
    f.write_text("print('hello')")
    artifact = FileArtifact(f)

    assert artifact.read() == "print('hello')"
    assert artifact.name == "code.py"

    artifact.write("print('world')")
    assert artifact.read() == "print('world')"


def test_file_artifact_snapshot_restore(tmp_path: Path):
    f = tmp_path / "code.py"
    f.write_text("version 1")
    artifact = FileArtifact(f)

    # Snapshot
    snapshot_dir = tmp_path / "snapshot"
    artifact.snapshot(snapshot_dir)
    assert (snapshot_dir / "code.py").read_text() == "version 1"

    # Modify
    artifact.write("version 2")
    assert artifact.read() == "version 2"

    # Restore
    artifact.restore(snapshot_dir)
    assert artifact.read() == "version 1"


def test_file_artifact_diff(tmp_path: Path):
    f = tmp_path / "code.py"
    f.write_text("line 1\nline 2\n")
    artifact = FileArtifact(f)

    snapshot_dir = tmp_path / "snapshot"
    artifact.snapshot(snapshot_dir)

    artifact.write("line 1\nline 2 modified\nline 3\n")

    diff = artifact.diff(snapshot_dir)
    assert diff is not None
    assert "line 2 modified" in diff
    assert "-line 2" in diff


def test_file_artifact_diff_no_change(tmp_path: Path):
    f = tmp_path / "code.py"
    f.write_text("same content")
    artifact = FileArtifact(f)

    snapshot_dir = tmp_path / "snapshot"
    artifact.snapshot(snapshot_dir)

    assert artifact.diff(snapshot_dir) is None


def test_directory_artifact_read_write(tmp_path: Path):
    d = tmp_path / "project"
    d.mkdir()
    (d / "a.py").write_text("code a")
    (d / "b.py").write_text("code b")

    artifact = DirectoryArtifact(d)
    content = artifact.read()
    assert content["a.py"] == "code a"
    assert content["b.py"] == "code b"

    artifact.write({"a.py": "new code a", "c.py": "code c"})
    assert (d / "a.py").read_text() == "new code a"
    assert (d / "c.py").read_text() == "code c"


def test_directory_artifact_snapshot_restore(tmp_path: Path):
    d = tmp_path / "project"
    d.mkdir()
    (d / "a.py").write_text("v1")

    artifact = DirectoryArtifact(d)
    snapshot_dir = tmp_path / "snapshot"
    artifact.snapshot(snapshot_dir)

    (d / "a.py").write_text("v2")
    artifact.restore(snapshot_dir)
    assert (d / "a.py").read_text() == "v1"
