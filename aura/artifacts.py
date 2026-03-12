from __future__ import annotations

import difflib
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Artifact(ABC):
    """Base class for anything being improved across iterations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this artifact."""
        ...

    @abstractmethod
    def snapshot(self, dest: Path) -> None:
        """Save current state to dest directory for versioning."""
        ...

    @abstractmethod
    def restore(self, src: Path) -> None:
        """Restore state from a previous snapshot directory."""
        ...

    @abstractmethod
    def read(self) -> Any:
        """Read current state."""
        ...

    @abstractmethod
    def write(self, content: Any) -> None:
        """Update state."""
        ...

    def diff(self, src: Path) -> str | None:
        """Compute diff from a previous snapshot. Returns None if not supported."""
        return None


class FileArtifact(Artifact):
    """A single text file being improved across iterations."""

    def __init__(self, path: Path):
        self.path = Path(path)

    @property
    def name(self) -> str:
        return self.path.name

    def snapshot(self, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.path, dest / self.path.name)

    def restore(self, src: Path) -> None:
        src_file = src / self.path.name
        if src_file.exists():
            shutil.copy2(src_file, self.path)

    def read(self) -> str:
        return self.path.read_text()

    def write(self, content: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(content)

    def diff(self, src: Path) -> str | None:
        src_file = src / self.path.name
        if not src_file.exists():
            return None
        old_lines = src_file.read_text().splitlines(keepends=True)
        new_lines = self.path.read_text().splitlines(keepends=True)
        diff = difflib.unified_diff(old_lines, new_lines, fromfile=f"previous/{self.path.name}", tofile=f"current/{self.path.name}")
        result = "".join(diff)
        return result if result else None


class DirectoryArtifact(Artifact):
    """A directory of files being improved together. Placeholder for future use."""

    def __init__(self, path: Path):
        self.path = Path(path)

    @property
    def name(self) -> str:
        return self.path.name

    def snapshot(self, dest: Path) -> None:
        dest_dir = dest / self.path.name
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(self.path, dest_dir)

    def restore(self, src: Path) -> None:
        src_dir = src / self.path.name
        if src_dir.exists():
            if self.path.exists():
                shutil.rmtree(self.path)
            shutil.copytree(src_dir, self.path)

    def read(self) -> dict[str, str]:
        result = {}
        for f in sorted(self.path.rglob("*")):
            if f.is_file():
                try:
                    result[str(f.relative_to(self.path))] = f.read_text()
                except UnicodeDecodeError:
                    continue
        return result

    def write(self, content: dict[str, str]) -> None:
        for rel_path, text in content.items():
            full_path = self.path / rel_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(text)

    def diff(self, src: Path) -> str | None:
        return None  # TODO: implement multi-file diff
