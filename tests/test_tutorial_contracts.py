"""User tutorial setup files must expose only the canonical public workflow."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

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
