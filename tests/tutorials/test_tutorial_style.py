"""A single collection-wide guard for the tutorial's pedagogical boundary."""

import ast
from pathlib import Path
import re
import shlex

TUTORIALS = Path(__file__).resolve().parents[2] / "tutorials"


def test_tutorials_keep_configuration_and_infrastructure_out_of_the_learning_surface():
    for setup in TUTORIALS.rglob("setup*.py"):
        source = setup.read_text()
        tree = ast.parse(source)
        assert not any(
            isinstance(node, ast.Try | ast.Raise | ast.Assert) for node in ast.walk(tree)
        ), setup
        assert not re.search(
            r"sys\.path|os\.environ|run_manifest\.json|run_metadata\.json|motion_params\.json",
            source,
        ), setup
        assert not re.search(r"parents\[[3-9]", source), setup

    for script in TUTORIALS.rglob("all*.sh"):
        assert script.name in {"allrun.sh", "allplot.sh", "allclean.sh"}, script
        for line in script.read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            if line == 'cd -- "$(dirname -- "$0")"':
                # Resolve direct commands relative to the tutorial, even when
                # its launcher is invoked from a different working directory.
                continue
            if script.name == "allclean.sh":
                # Destructive commands retain exactly one case-local directory change.
                assert line.startswith(("cd -- ", "rm -")), (script, line)
                continue
            assert line.startswith("python "), (script, line)
            command = shlex.split(line)
            assert (script.parent / command[1]).is_file(), (script, command[1])
            assert not re.search(
                r"\b(?:export|mkdir|tee|allclean|allplot|tutorial_runner)\b", line
            ), (script, line)
            if script.name == "allrun.sh":
                assert "postprocess" not in line and "validate" not in line, (script, line)
