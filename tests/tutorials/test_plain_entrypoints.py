"""Exercise every copied default setup and shell launcher without production runs."""

import ast
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

from openonda.tutorials import TUTORIALS, materialize_tutorial


def test_fvm_tutorials_leave_mpi_configuration_and_ownership_in_the_library():
    """Guard setups, launchers and assets against reintroducing MPI workarounds."""
    root = Path(__file__).resolve().parents[2] / "tutorials"
    runtime_modules = ("mpi4py", "petsc4py", "openonda.runtime", "source.solvers.fvm.core.parallel")
    runtime_attributes = {
        "COMM_WORLD",
        "Get_rank",
        "Get_size",
        "RunConfig",
        "ParallelContext",
        "ensure_mpi",
        "ensure_runtime",
        "is_root",
        "is_master",
        "is_host",
    }
    runtime_tokens = re.compile(
        r"\b(?:mpiexec|mpirun|PYTHONPATH|OPENONDA_MPIEXEC|FVM_PETSC_\w+|"
        r"OMP_NUM_THREADS|OPENBLAS_NUM_THREADS|MKL_NUM_THREADS|VECLIB_MAXIMUM_THREADS|"
        r"TI_CPU_MAX_NUM_THREADS|OMPI_\w+|PMI_\w+|PMIX_\w+|SLURM_\w+)\b"
    )
    for family in ("fvm", "coupled_fvm_vpm"):
        for path in (root / family).rglob("*"):
            if path.suffix not in (".py", ".sh"):
                continue
            text = path.read_text()
            if path.suffix == ".sh":
                commands = "\n".join(
                    line for line in text.splitlines() if not line.lstrip().startswith("#")
                )
                assert not runtime_tokens.search(commands), path
                continue
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                    names.extend(f"{node.module}.{alias.name}" for alias in node.names)
                else:
                    names = []
                assert not any(
                    name == banned or name.startswith(banned + ".")
                    for name in names
                    for banned in runtime_modules
                ), (path, node.lineno)
                if isinstance(node, ast.Attribute):
                    assert node.attr not in runtime_attributes, (path, node.lineno)
                    assert not (
                        isinstance(node.value, ast.Name)
                        and node.value.id == "sys"
                        and node.attr == "path"
                    ), (path, node.lineno)
                if isinstance(node, ast.keyword):
                    assert node.arg not in ("parallel_mode", "linear_backend"), (path, node.lineno)
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    # Exact configuration tokens, rather than documentation prose.
                    assert not runtime_tokens.fullmatch(node.value), (path, node.lineno)


@pytest.mark.parametrize(
    "tutorial",
    (*TUTORIALS, None),
    ids=lambda tutorial: tutorial.name if tutorial else "quadcopter_studies",
)
def test_default_setup_reaches_solver_without_environment_knobs(tmp_path, tutorial):
    if tutorial is None:
        parent = materialize_tutorial("vpm/quadcopter", tmp_path / "studies with spaces")
        case = parent / "studies"
    else:
        case = materialize_tutorial(tutorial.name, tmp_path / "workspace with spaces")
    probe = tmp_path / "probe.py"
    probe.write_text("""
import runpy
import sys
from openonda import coupler, fvm, vpm

class Constructed(Exception):
    pass
def capture(*args, **kwargs):
    print("DEFAULT_CASE_CONSTRUCTED")
    raise Constructed
fvm.create_fvm_solver = capture
fvm.FVMSolver = capture
vpm.VPMSolver = capture
coupler.create_coupler = capture
sys.argv = [sys.argv[1]]
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
except Constructed:
    pass
else:
    raise AssertionError("The default setup did not construct a solver")
""")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-I", str(probe), str(case / "setup.py")],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout[-1000:] + result.stderr[-3000:]
    assert result.stdout.count("DEFAULT_CASE_CONSTRUCTED") == 1


def test_all_shell_launchers_work_outside_the_case_and_stop_on_failure(tmp_path):
    root = Path(__file__).resolve().parents[2] / "tutorials"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "assert Path(sys.argv[1]).is_file(), sys.argv[1]\n"
        "with open(os.environ['CALLS'], 'a') as out:\n"
        "    out.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "raise SystemExit(int(os.environ['FAIL']))\n"
    )
    python.chmod(0o755)
    for i, original in enumerate(sorted(root.rglob("all*.sh"))):
        if original.name == "allclean.sh":
            continue
        case = tmp_path / f"case {i}"
        case.mkdir()
        script = case / original.name
        shutil.copy2(original, script)
        (case / "allclean.sh").write_text("#!/bin/bash\nexit 0\n")
        (case / "allclean.sh").chmod(0o755)
        # Stub every executable Python asset, never call its real solver/plotter.
        for source in original.parent.rglob("*.py"):
            dest = case / source.relative_to(original.parent)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.touch()
        formats = (None, "png", "pdf") if original.name == "allplot.sh" else (None,)
        for figure_format in formats:
            for fail in (0, 23):
                calls = tmp_path / "calls.jsonl"
                calls.write_text("")
                env = {
                    **os.environ,
                    "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
                    "CALLS": str(calls),
                    "FAIL": str(fail),
                }
                args = ["bash", "-e", str(script)]
                if figure_format:
                    args.append(figure_format)
                completed = subprocess.run(
                    args, cwd=tmp_path, env=env, capture_output=True, text=True
                )
                assert completed.returncode == fail, (original, completed.stderr)
                records = [json.loads(line) for line in calls.read_text().splitlines()]
                assert records, original
                if fail:
                    assert len(records) == 1, original
                for record in records:
                    if "--format" in record:
                        assert record[record.index("--format") + 1] == (figure_format or "png")
