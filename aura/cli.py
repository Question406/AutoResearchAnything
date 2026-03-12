"""Aura CLI entry point — detect, validate, and run user experiment files."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
import json
import runpy
import sys
import traceback
from pathlib import Path


def _detect_from_ast(tree: ast.Module) -> str | None:
    """Pure AST analysis: return entry pattern or None."""
    has_pipeline_assign = False
    has_workspace_assign = False

    has_main = False
    has_runner = False

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main":
            has_main = True

        elif isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id == "Runner":
                    has_runner = True
                elif isinstance(base, ast.Attribute) and base.attr == "Runner":
                    has_runner = True

        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id == "pipeline":
                        has_pipeline_assign = True
                    elif target.id == "workspace":
                        has_workspace_assign = True
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                if node.target.id == "pipeline":
                    has_pipeline_assign = True
                elif node.target.id == "workspace":
                    has_workspace_assign = True

    # Priority order: main > runner > globals
    if has_main:
        return "main"
    if has_runner:
        return "runner"
    if has_pipeline_assign and has_workspace_assign:
        return "globals"
    return None


def detect_entry_pattern(file_path: Path) -> str | None:
    """Read a Python file and return its entry pattern: 'main', 'runner', 'globals', or None."""
    try:
        source = Path(file_path).read_text()
        tree = ast.parse(source, filename=str(file_path))
    except (OSError, SyntaxError):
        return None
    return _detect_from_ast(tree)


def validate_file(file_path: Path) -> dict:
    """Validate a Python file and return structured results."""
    result: dict = {
        "valid": False,
        "entry_pattern": None,
        "errors": [],
        "warnings": [],
    }

    file_path = Path(file_path)
    if not file_path.exists():
        result["errors"].append(f"File not found: {file_path}")
        return result

    try:
        source = file_path.read_text()
    except OSError as e:
        result["errors"].append(f"Cannot read file: {e}")
        return result

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        result["errors"].append(f"Syntax error: {e}")
        return result

    pattern = _detect_from_ast(tree)
    result["entry_pattern"] = pattern

    if pattern is None:
        result["errors"].append(
            "No entry pattern found: define main(), a Runner subclass, "
            "or pipeline/workspace globals"
        )
    else:
        result["valid"] = True

    return result


def run_file(file_path: Path, run_dir: Path | None = None) -> None:
    """Detect entry pattern and execute the file accordingly."""
    pattern = detect_entry_pattern(file_path)

    if pattern is None:
        print(f"Error: no recognised entry pattern in {file_path}", file=sys.stderr)
        sys.exit(1)

    if pattern == "main":
        spec = importlib.util.spec_from_file_location("__user_module__", str(file_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main()

    elif pattern == "runner":
        import aura.runner

        spec = importlib.util.spec_from_file_location("__user_module__", str(file_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        runner_cls = None
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if obj is not aura.runner.Runner and issubclass(obj, aura.runner.Runner):
                runner_cls = obj
                break

        if runner_cls is None:
            print("Error: Runner subclass detected in AST but not loadable", file=sys.stderr)
            sys.exit(1)

        instance = runner_cls()
        instance.run(run_dir=run_dir)

    elif pattern == "globals":
        runpy.run_path(str(file_path), run_name="__main__")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point with run and validate subcommands."""
    parser = argparse.ArgumentParser(prog="aura", description="Aura experiment runner")
    sub = parser.add_subparsers(dest="command")

    # --- run ---
    run_parser = sub.add_parser("run", help="Run an experiment file")
    run_parser.add_argument("file", type=Path, help="Python file to run")
    run_parser.add_argument("--run-dir", type=Path, default=None, help="Output directory")

    # --- validate ---
    val_parser = sub.add_parser("validate", help="Validate an experiment file")
    val_parser.add_argument("file", type=Path, help="Python file to validate")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "validate":
        result = validate_file(args.file)
        # JSON to stdout
        print(json.dumps(result, indent=2))
        # Human summary to stderr
        if result["valid"]:
            print(f"OK: pattern={result['entry_pattern']}", file=sys.stderr)
        else:
            for err in result["errors"]:
                print(f"ERROR: {err}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "run":
        try:
            run_file(args.file, run_dir=args.run_dir)
        except SystemExit:
            raise
        except Exception:
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)
