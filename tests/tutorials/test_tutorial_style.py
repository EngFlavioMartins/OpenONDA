"""A single collection-wide guard for the tutorial's pedagogical boundary."""

import ast
from pathlib import Path
import re
import shlex

from openonda.tutorials import _EXCLUDED_NAMES, _EXCLUDED_PARTS

TUTORIALS = Path(__file__).resolve().parents[2] / "tutorials"


def test_tutorials_keep_configuration_and_infrastructure_out_of_the_learning_surface():
    for path in TUTORIALS.rglob("*.py"):
        if path.name in _EXCLUDED_NAMES or any(
            part in _EXCLUDED_PARTS for part in path.relative_to(TUTORIALS).parts
        ):
            continue
        source = path.read_text()
        tree = ast.parse(source)
        assert not any(
            isinstance(node, ast.Try | ast.TryStar | ast.Raise | ast.Assert)
            or isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
            and node.decorator_list
            for node in ast.walk(tree)
        ), path
        assert not re.search(r"sys\.path|__package__|case_package", source), path
        assert not re.search(
            r"MPLBACKEND|MPLCONFIGDIR|os\.environ|(?:matplotlib|mpl)\.use\(", source
        ), path
        assert not re.search(
            r"lru_cache|functools\.cache|hashlib|_file_stamp|\b(?:np|numpy)\.load\("
            r"|\b(?:pd|pandas)\.read_csv|json\.load\(|json\.loads\([^\n]*\.read_text\("
            r"|csv\.(?:DictReader|reader)|\.mkdir\(|TemporaryDirectory|mkdtemp|shutil\."
            r"|\.write_text\(|\.to_csv\(|\b(?:np|numpy)\.savetxt\("
            r"|subprocess\.|\.(?:is_file|exists|is_dir)\(|\b(?:getattr|hasattr)\(",
            source,
        ), path
        if not path.name.startswith("setup"):
            continue
        assert not re.search(
            r"os\.environ|run_manifest\.json|run_metadata\.json|motion_params\.json",
            source,
        ), path
        assert not re.search(r"parents\[[3-9]", source), path

    for script in TUTORIALS.rglob("all*.sh"):
        if any(part in _EXCLUDED_PARTS for part in script.relative_to(TUTORIALS).parts):
            continue
        assert script.name in {"allrun.sh", "allplot.sh", "allclean.sh", "allcontinue.sh"}, script
        for line in script.read_text().splitlines():
            if not line or line.startswith("#") or line == "set -e":
                continue
            if line == 'cd "$(dirname "$0")"':
                # Resolve direct commands relative to the tutorial, even when
                # its launcher is invoked from a different working directory.
                continue
            if script.name == "allrun.sh" and line == "./allclean.sh":
                continue
            command = shlex.split(line)
            assert command[0] == "python", (script, line)
            assert "||" not in command, (script, line)
            if command[:3] == ["python", "-m", "openonda.tutorial_runner"]:
                assert command[3] == ".", (script, line)
                module = command[4]
                if script.name == "allclean.sh":
                    assert module == "clean", (script, line)
                else:
                    assert (script.parent / (module.replace(".", "/") + ".py")).is_file(), script
                    assert not any(
                        word in module for word in ("validate", "verify", "check_run")
                    ), script
                continue
            if script.name == "allplot.sh" and command[:4] == [
                "python",
                "-m",
                "openonda.results",
                "restore",
            ]:
                continue
            raise AssertionError((script, line))
