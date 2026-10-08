"""A single collection-wide guard for the tutorial's pedagogical boundary."""

import ast
from pathlib import Path
import re
import shlex

TUTORIALS = Path(__file__).resolve().parents[2] / "tutorials"
GENERATED = {"solution", "solutions", "samples", "figures", "study_results", "__pycache__"}


def _tutorial_files(pattern):
    return [
        path
        for path in TUTORIALS.rglob(pattern)
        if not GENERATED.intersection(path.relative_to(TUTORIALS).parts)
        and path.name != "paraview_state.py"
    ]


def test_tutorial_scripts_contain_physics_and_direct_solver_calls_only():
    for path in _tutorial_files("*.py"):
        source = path.read_text()
        tree = ast.parse(source)
        assert not any(
            isinstance(node, ast.Try | ast.TryStar | ast.Raise | ast.Assert)
            or isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
            and node.decorator_list
            for node in ast.walk(tree)
        ), path
        assert not any(
            isinstance(node, ast.ImportFrom) and node.level for node in ast.walk(tree)
        ), path
        assert not re.search(r"sys\.path|__package__|tutorial_runner", source), path
        assert not re.search(
            r"MPLBACKEND|MPLCONFIGDIR|os\.environ|(?:matplotlib|mpl)\.use\(", source
        ), path
        assert not re.search(
            r"lru_cache|functools\.cache|hashlib|\.(?:is_file|exists|is_dir)\("
            r"|\b(?:getattr|hasattr)\(",
            source,
        ), path
        if path.name == "setup.py":
            assert not re.search(r"run_manifest\.json|run_metadata\.json", source), path
            assert not re.search(r"parents\[[3-9]", source), path


def test_launchers_clean_then_run_plain_scripts():
    for script in _tutorial_files("all*.sh"):
        lines = [line for line in script.read_text().splitlines() if line and line[0] != "#"]
        if re.fullmatch(r"allrun_[a-z_]+\.sh", script.name):
            # A parameter study runs copies of the case, one hard-coded run per line
            assert lines[:2] == ["set -e", 'cd "$(dirname "$0")"'], script
            assert not any("||" in line or "rm " in line for line in lines), script
            continue
        standard = {"allrun.sh", "allplot.sh", "allclean.sh", "allcontinue.sh"}
        assert script.name in standard or re.fullmatch(r"allplot_[a-z_]+\.sh", script.name), script
        if script.name == "allrun.sh":
            assert "./allclean.sh" in lines, script
        for line in lines:
            if line in ('cd "$(dirname "$0")"', "set -e") or (
                script.name == "allrun.sh" and line == "./allclean.sh"
            ):
                continue
            command = shlex.split(line)
            if script.name == "allclean.sh":
                assert command[:2] == ["rm", "-rf"], (script, line)
                assert set(command[2:]) <= GENERATED, (script, line)
                continue
            assert command[0] == "python" and "||" not in command, (script, line)
            if command[1] == "setup.py":
                assert script.name in {"allrun.sh", "allcontinue.sh"}, (script, line)
                continue
            assert command[1] == "-m" and command[2].startswith("assets."), (script, line)
            assert (script.parent / (command[2].replace(".", "/") + ".py")).is_file(), script
