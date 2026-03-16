from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from aura.interfaces import Environment

if TYPE_CHECKING:
    from aura.types import Hypothesis
    from aura.workspace import Workspace


class CondaEnvironment(Environment):
    """Verify a conda environment exists and provide an activation prefix.

    ``setup()`` returns ``{"conda_env": name, "activation_prefix": "conda run -n <name>"}``
    which ScriptExecutor will prepend to commands.
    """

    def __init__(self, env_name: str):
        self.env_name = env_name

    def setup(self, task: Hypothesis, workspace: Workspace) -> dict:
        result = subprocess.run(
            ["conda", "env", "list"], capture_output=True, text=True
        )
        envs = [line.split()[0] for line in result.stdout.splitlines() if line and not line.startswith("#")]
        if self.env_name not in envs:
            raise RuntimeError(f"Conda environment '{self.env_name}' not found. Available: {envs}")
        context = {
            "conda_env": self.env_name,
            "activation_prefix": f"conda run -n {self.env_name}",
        }
        workspace.update_manifest(environment={"type": "conda", "env_name": self.env_name})
        return context


class VenvEnvironment(Environment):
    """Create or reuse a virtualenv and optionally install requirements.

    ``setup()`` returns ``{"python": "/venv/bin/python", "pip": "/venv/bin/pip"}``.
    """

    def __init__(
        self,
        path: str | Path,
        requirements: str | None = None,
        auto_cleanup: bool = False,
    ):
        self.path = Path(path)
        self.requirements = requirements
        self.auto_cleanup = auto_cleanup

    def setup(self, task: Hypothesis, workspace: Workspace) -> dict:
        import sys

        if not (self.path / "bin" / "python").exists():
            subprocess.run([sys.executable, "-m", "venv", str(self.path)], check=True)

        pip = str(self.path / "bin" / "pip")
        python = str(self.path / "bin" / "python")

        if self.requirements:
            subprocess.run([pip, "install", "-r", self.requirements], check=True)

        workspace.update_manifest(environment={
            "type": "venv",
            "path": str(self.path),
            "requirements": self.requirements,
        })
        return {"python": python, "pip": pip}

    def teardown(self, context: dict, workspace: Workspace) -> None:
        if self.auto_cleanup and self.path.exists():
            import shutil

            shutil.rmtree(self.path, ignore_errors=True)


class UvEnvironment(VenvEnvironment):
    """Like VenvEnvironment but uses ``uv`` for faster venv creation and installs."""

    def setup(self, task: Hypothesis, workspace: Workspace) -> dict:
        if not (self.path / "bin" / "python").exists():
            subprocess.run(["uv", "venv", str(self.path)], check=True)

        python = str(self.path / "bin" / "python")
        pip = str(self.path / "bin" / "pip")

        if self.requirements:
            subprocess.run(["uv", "pip", "install", "-r", self.requirements, "--python", python], check=True)

        workspace.update_manifest(environment={
            "type": "uv",
            "path": str(self.path),
            "requirements": self.requirements,
        })
        return {"python": python, "pip": pip}


class DockerEnvironment(Environment):
    """Pull a Docker image and start a container with ``sleep infinity``.

    ``setup()`` returns ``{"container_id": "abc123", "image": image_name}``.
    ``teardown()`` stops the container.
    """

    def __init__(
        self,
        image: str,
        volumes: dict[str, str] | None = None,
        env_vars: dict[str, str] | None = None,
    ):
        self.image = image
        self.volumes = volumes or {}
        self.env_vars = env_vars or {}

    def setup(self, task: Hypothesis, workspace: Workspace) -> dict:
        subprocess.run(["docker", "pull", self.image], check=True)

        cmd = ["docker", "run", "-d"]
        for host, container in self.volumes.items():
            cmd += ["-v", f"{host}:{container}"]
        for k, v in self.env_vars.items():
            cmd += ["-e", f"{k}={v}"]
        cmd += [self.image, "sleep", "infinity"]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        container_id = result.stdout.strip()
        workspace.update_manifest(environment={"type": "docker", "image": self.image, "container_id": container_id})
        return {"container_id": container_id, "image": self.image}

    def teardown(self, context: dict, workspace: Workspace) -> None:
        container_id = context.get("container_id")
        if container_id:
            subprocess.run(["docker", "stop", container_id], capture_output=True)
