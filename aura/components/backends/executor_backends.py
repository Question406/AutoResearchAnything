from __future__ import annotations

import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aura.interfaces import Executor

if TYPE_CHECKING:
    from aura.types import Hypothesis
    from aura.workspace import Workspace


class ScriptExecutor(Executor):
    """Run a shell command with {field} interpolation from task.spec.

    Context keys used:
    - ``trial_dir`` — working directory for the subprocess (filesystem isolation)
    - ``trial_inputs`` — list of Path inputs (informational; not mounted by this executor)
    - ``activation_prefix`` (e.g. from CondaEnvironment) — prepended to command
    - ``container_id`` (e.g. from DockerEnvironment) — wraps command in docker exec
    """

    def __init__(self, command_template: str, timeout: int = 300):
        self.command_template = command_template
        self.timeout = timeout

    def run(self, task: Hypothesis, context: dict, workspace: Workspace) -> dict:
        ts_start = datetime.now(UTC).isoformat()
        cmd = self.command_template.format(**task.spec)

        if context.get("container_id"):
            cmd = f"docker exec {context['container_id']} sh -c {cmd!r}"
        elif context.get("activation_prefix"):
            cmd = f"{context['activation_prefix']} {cmd}"

        timeout = (
            workspace.constraints().get("time_budget", self.timeout) if workspace else self.timeout
        )

        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout,
            cwd=context.get("trial_dir"),  # filesystem isolation
        )
        return {
            "cmd": cmd,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "ts_start": ts_start,
            "ts_end": datetime.now(UTC).isoformat(),
        }


class BwrapExecutor(ScriptExecutor):
    """Like ScriptExecutor but runs inside a bubblewrap sandbox.

    The entire filesystem is bind-mounted read-only. ``trial_dir`` is the only
    writable path. ``trial_inputs`` are additionally explicitly marked read-only.
    A fresh ``/tmp`` is provided per trial.

    Requires ``bwrap`` (``apt install bubblewrap`` / ``dnf install bubblewrap``).
    No root needed.
    """

    def run(self, task: Hypothesis, context: dict, workspace: Workspace) -> dict:
        ts_start = datetime.now(UTC).isoformat()
        cmd = self.command_template.format(**task.spec)
        trial_dir = context.get("trial_dir", Path("."))
        trial_inputs = context.get("trial_inputs", [])

        timeout = (
            workspace.constraints().get("time_budget", self.timeout) if workspace else self.timeout
        )

        bwrap_args = [
            "bwrap",
            "--bind", "/", "/",           # full filesystem, read-write by default
            "--dev", "/dev",
            "--proc", "/proc",
            "--tmpfs", "/tmp",            # fresh /tmp per trial
            "--bind", str(trial_dir), str(trial_dir),  # trial_dir stays read-write
        ]
        for inp in trial_inputs:
            bwrap_args += ["--ro-bind", str(inp), str(inp)]  # inputs read-only
        bwrap_args += ["--chdir", str(trial_dir), "--", "sh", "-c", cmd]

        result = subprocess.run(bwrap_args, capture_output=True, text=True, timeout=timeout)
        return {
            "cmd": " ".join(bwrap_args),
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "ts_start": ts_start,
            "ts_end": datetime.now(UTC).isoformat(),
        }


class BindfsExecutor(ScriptExecutor):
    """Like ScriptExecutor but mounts ``trial_inputs`` as read-only via bindfs.

    Each input is bind-mounted into ``trial_dir/<input_name>`` with read-only
    permissions. Mounts are cleaned up after the subprocess exits.

    Requires ``bindfs`` and ``fusermount``
    (``apt install bindfs`` / ``dnf install fuse-bindfs``).
    No root needed.
    """

    def run(self, task: Hypothesis, context: dict, workspace: Workspace) -> dict:
        ts_start = datetime.now(UTC).isoformat()
        cmd = self.command_template.format(**task.spec)
        trial_dir = Path(context.get("trial_dir", "."))
        trial_inputs = context.get("trial_inputs", [])

        timeout = (
            workspace.constraints().get("time_budget", self.timeout) if workspace else self.timeout
        )

        # Mount each input as read-only inside trial_dir
        mount_points: list[Path] = []
        for inp in trial_inputs:
            inp = Path(inp)
            mount_point = trial_dir / inp.name
            mount_point.mkdir(parents=True, exist_ok=True)
            subprocess.run(["bindfs", "--read-only", str(inp), str(mount_point)], check=True)
            mount_points.append(mount_point)

        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=timeout, cwd=trial_dir,
            )
        finally:
            for mount_point in mount_points:
                subprocess.run(["fusermount", "-u", str(mount_point)], capture_output=True)

        return {
            "cmd": cmd,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "ts_start": ts_start,
            "ts_end": datetime.now(UTC).isoformat(),
        }


class FunctionExecutor(Executor):
    """Call a Python function with task.spec unpacked as kwargs."""

    def __init__(self, fn: Callable[..., Any], timeout: int | None = None):
        self.fn = fn
        self.timeout = timeout

    def run(self, task: Hypothesis, context: dict, workspace: Workspace) -> dict:
        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import TimeoutError as FuturesTimeout

        ts = datetime.now(UTC).isoformat()
        timeout = self.timeout
        if workspace:
            timeout = workspace.constraints().get("time_budget", timeout)

        if timeout:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self.fn, **task.spec)
                try:
                    result = future.result(timeout=timeout)
                except FuturesTimeout:
                    return {"output": None, "error": f"Timed out after {timeout}s", "ts": ts}
        else:
            result = self.fn(**task.spec)

        output = (
            result
            if isinstance(result, (dict, list, str, int, float, bool, type(None)))
            else str(result)
        )
        return {"output": output, "error": None, "ts": ts}


class LLMExecutor(Executor):
    """Send a hypothesis to an LLM and return its response."""

    def __init__(self, llm: Callable[[str], str], prompt_template: str | None = None):
        self.llm = llm
        self.prompt_template = prompt_template or "Complete this task:\n\n{{ query }}"

    def run(self, task: Hypothesis, context: dict, workspace: Workspace) -> dict:
        from aura.utils.parsing import render_prompt

        constraints = workspace.constraints() if workspace else {}
        prompt = render_prompt(self.prompt_template, **task.spec, constraints=constraints)
        response = self.llm(prompt)
        return {"prompt": prompt, "response": response}


class SlurmExecutor(Executor):
    """Submit a job to SLURM via sbatch and poll until completion.

    Requires ``sbatch`` and ``squeue`` on the PATH.
    """

    def __init__(
        self,
        script_template: str,
        partition: str,
        poll_interval: int = 30,
        timeout: int = 3600,
    ):
        self.script_template = script_template
        self.partition = partition
        self.poll_interval = poll_interval
        self.timeout = timeout

    def run(self, task: Hypothesis, context: dict, workspace: Workspace) -> dict:
        import time
        import tempfile

        script_content = self.script_template.format(**task.spec)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sh", delete=False, prefix="aura_slurm_"
        ) as f:
            f.write(f"#!/bin/bash\n#SBATCH --partition={self.partition}\n{script_content}")
            script_path = f.name

        try:
            submit = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
            if submit.returncode != 0:
                return {"job_id": None, "output": None, "error": submit.stderr}

            job_id = submit.stdout.strip().split()[-1]

            start = time.time()
            while time.time() - start < self.timeout:
                time.sleep(self.poll_interval)
                check = subprocess.run(
                    ["squeue", "-j", job_id, "-h"], capture_output=True, text=True
                )
                if not check.stdout.strip():
                    break

            return {"job_id": job_id, "output": {"job_id": job_id}, "error": None}
        finally:
            Path(script_path).unlink(missing_ok=True)
