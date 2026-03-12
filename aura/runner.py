from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aura.pipeline import Pipeline
    from aura.workspace import Workspace


class Runner(ABC):
    @abstractmethod
    def setup_inputs(self, workspace: Workspace) -> None:
        ...

    @abstractmethod
    def build_pipeline(self, workspace: Workspace) -> Pipeline:
        ...

    def run(self, run_dir: Path | None = None) -> Workspace:
        from aura.workspace import Workspace as WS

        if run_dir is None:
            run_dir = Path("runs") / f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        workspace = WS.create(run_dir)
        self.setup_inputs(workspace)
        pipeline = self.build_pipeline(workspace)
        pipeline.run()
        return workspace
