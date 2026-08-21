"""User tutorial setup files must expose only the canonical public workflow."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
import runpy
import sys

import pytest

import openonda.coupler as coupling
import openonda.fvm as fvm
import openonda.vpm as vpm

ROOT = Path(__file__).resolve().parents[1]
SETUP_FILES = tuple(
    sorted(
        {
            *(ROOT / "tutorials").rglob("*_setup.py"),
            *(ROOT / "tutorials").rglob("setup_*.py"),
        }
    )
)
COUPLED_SETUPS = tuple(
    path
    for path in SETUP_FILES
    if "coupled_FVM_VPM" in path.parts and "referenceFlow" not in path.parts
)
PUBLIC_NAMESPACES = {
    "openonda.fvm": "fvm",
    "openonda.vpm": "vpm",
    "openonda.coupler": "coupling",
}
PUBLIC_MODULES = {"coupling": coupling, "fvm": fvm, "vpm": vpm}
FORBIDDEN_TEXT = (
    "source.solvers",
    "is_master_rank",
    "vpm_solver = None",
    "setup_fvm_solver",
    "setup_vpm_solver",
    "setup_coupler",
    "backup",
    "SAMPLE_PERIOD",
)
FORBIDDEN_METADATA_KEYS = {
    "backup_frequency",
    "backup_directory",
    "backup_file_name",
    "processing_unit",
    "raw_backup_interval",
    "dt",
    "num_steps",
    "sample_interval",
}


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _public_call_target(node: ast.expr):
    attributes: list[str] = []
    while isinstance(node, ast.Attribute):
        attributes.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name) or node.id not in PUBLIC_MODULES:
        return None

    target = PUBLIC_MODULES[node.id]
    for attribute in reversed(attributes):
        target = getattr(target, attribute, None)
        if target is None:
            return None
    return target


@pytest.mark.parametrize("path", SETUP_FILES, ids=lambda path: str(path.relative_to(ROOT)))
def test_tutorial_uses_public_names_and_has_a_main_guard(path: Path):
    source = path.read_text(encoding="utf-8")
    assert not any(token.lower() in source.lower() for token in FORBIDDEN_TEXT)

    tree = _tree(path)
    docstring = ast.get_docstring(tree) or ""
    assert "Usage:" in docstring or "Example:" in docstring
    assert any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        for node in tree.body
    ), f"{path} must not launch a simulation merely by being imported"


@pytest.mark.parametrize("path", SETUP_FILES, ids=lambda path: str(path.relative_to(ROOT)))
def test_tutorial_module_imports_without_launching(path: Path, monkeypatch):
    """Execute top-level setup declarations while leaving the main guard closed."""
    existing_modules = set(sys.modules)
    monkeypatch.syspath_prepend(str(path.parent))
    try:
        runpy.run_path(str(path), run_name=f"_openonda_tutorial_{path.stem}")
    finally:
        for name in set(sys.modules) - existing_modules:
            if name == "assets" or name.startswith("assets."):
                sys.modules.pop(name, None)


@pytest.mark.parametrize("path", SETUP_FILES, ids=lambda path: str(path.relative_to(ROOT)))
def test_tutorial_imports_solver_modules_as_namespaces(path: Path):
    tree = _tree(path)
    imports = {
        alias.name: alias.asname
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name in PUBLIC_NAMESPACES
    }
    direct_imports = {
        node.module for node in tree.body if isinstance(node, ast.ImportFrom)
    }.intersection(PUBLIC_NAMESPACES)

    if path.parts[-3] == "FVM" or "referenceFlow" in path.parts:
        expected = {"openonda.fvm": "fvm"}
    elif path.parts[-3] == "VPM":
        expected = {"openonda.vpm": "vpm"}
    else:
        expected = PUBLIC_NAMESPACES

    assert not direct_imports
    assert imports.items() >= expected.items()


@pytest.mark.parametrize("path", COUPLED_SETUPS, ids=lambda path: path.parent.name)
def test_coupled_tutorial_imports_public_module_namespaces(path: Path):
    aliases = {
        (node.module, alias.asname)
        for node in _tree(path).body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    imports = {
        (alias.name, alias.asname)
        for node in _tree(path).body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert ("openonda.fvm", "fvm") in imports
    assert ("openonda.vpm", "vpm") in imports
    assert ("openonda.coupler", "coupling") in imports
    assert not aliases.intersection(
        {
            ("source.solvers.FVM", None),
            ("source.solvers.VPM", None),
            ("source.coupler", None),
        }
    )


@pytest.mark.parametrize("path", SETUP_FILES, ids=lambda path: str(path.relative_to(ROOT)))
def test_solver_construction_has_an_explicit_case_directory(path: Path):
    for node in ast.walk(_tree(path)):
        if not isinstance(node, ast.Call):
            continue
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name not in {"VPMSolver", "create_vpm_solver", "create_fvm_solver"}:
            continue
        assert any(keyword.arg == "case_dir" for keyword in node.keywords), (
            f"{path}:{node.lineno}: {name} output would otherwise depend on cwd"
        )


@pytest.mark.parametrize("path", SETUP_FILES, ids=lambda path: str(path.relative_to(ROOT)))
def test_tutorial_metadata_and_paths_are_canonical(path: Path):
    source = path.read_text(encoding="utf-8")
    assert "solution/samples" not in source.replace(" ", "")

    string_keys = {
        node.value
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert not string_keys.intersection(FORBIDDEN_METADATA_KEYS)


@pytest.mark.parametrize("path", SETUP_FILES, ids=lambda path: str(path.relative_to(ROOT)))
def test_public_tutorial_call_keywords_match_current_signatures(path: Path):
    """Catch stale public constructor keywords without launching a simulation."""
    for node in ast.walk(_tree(path)):
        if not isinstance(node, ast.Call):
            continue
        target = _public_call_target(node.func)
        if target is None or not callable(target):
            continue
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            continue
        if any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            continue
        unexpected = sorted(
            keyword.arg
            for keyword in node.keywords
            if keyword.arg is not None and keyword.arg not in signature.parameters
        )
        assert not unexpected, (
            f"{path}:{node.lineno}: {ast.unparse(node.func)} has stale keyword(s) "
            f"{', '.join(unexpected)}"
        )


@pytest.mark.parametrize("path", SETUP_FILES, ids=lambda path: str(path.relative_to(ROOT)))
def test_vlm_tutorials_give_vpm_the_same_molecular_viscosity(path: Path):
    """The VPM owns molecular viscosity in a coupled VLM--VPM setup."""
    tree = _tree(path)
    vlm_values = {
        ast.dump(keyword.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("VLMSetup")
        for keyword in node.keywords
        if keyword.arg == "kinematic_viscosity"
    }
    if not vlm_values:
        return

    vpm_values = {
        ast.dump(keyword.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("ViscousConfig.cs")
        for keyword in node.keywords
        if keyword.arg == "kinematic_viscosity"
    }
    assert vlm_values & vpm_values, (
        f"{path}: VLM and VPM must explicitly share the same kinematic_viscosity"
    )
