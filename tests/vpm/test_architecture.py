"""Architecture tests: enforce the VPM subsystem dependency boundaries.

Mirrors ``source/solvers/VPM/ARCHITECTURE.md``.  These tests statically parse
the import graph (no Taichi backend needed) and fail when a runtime import
crosses a forbidden or unlisted subsystem boundary.  ``if TYPE_CHECKING:``
imports are ignored; ``io/logging.py`` is a leaf allowed from any subsystem.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

VPM_ROOT = Path(__file__).resolve().parents[2] / "source" / "solvers" / "VPM"

SUBSYSTEMS = {d.name for d in VPM_ROOT.iterdir() if d.is_dir() and (d / "__init__.py").exists()}

# Allowed edges exactly as documented in ARCHITECTURE.md (io/logging is a leaf
# available to every subsystem, so `io` is allowed wherever logging is listed).
ALLOWED_EDGES: dict[str, set[str]] = {
    "core": {
        "physics",
        "stabilization",
        "coupling",
        "diagnostics",
        "io",
        "config",
        "particles",
        "turbulence",
        "boundary_elements",
        "acceleration",
    },
    "physics": {"particles", "numerics", "kernels", "acceleration", "config", "io"},
    "stabilization": {"particles", "numerics", "diagnostics", "config", "io"},
    "coupling": {"physics", "particles", "boundary_elements", "core"},
    "diagnostics": {"particles", "physics", "numerics", "config", "io"},
    "io": {"config", "diagnostics", "particles"},
    "config": {"boundary_elements"},
    "initial_conditions": {"particles"},
    "particles": {"config", "initial_conditions"},
    "boundary_elements": {"config", "io"},
    "turbulence": {"config"},
    "acceleration": {"config"},
    "kernels": {"config"},
    "numerics": {"config"},
}

# Edges explicitly forbidden regardless of the allowed set.
FORBIDDEN_EDGES: set[tuple[str, str]] = {
    ("physics", "stabilization"),
    ("physics", "core"),
    ("particles", "core"),
    ("particles", "stabilization"),
    ("diagnostics", "core"),
    ("stabilization", "core"),
}


def _module_package(rel: tuple[str, ...]) -> tuple[str, ...]:
    """Relative package path of a module file, e.g. ('core',) for core/solver.py."""
    return rel[:-1]


def _collect_runtime_edges() -> set[tuple[str, str]]:
    """Scan the tree for runtime cross-subsystem relative imports."""
    edges: set[tuple[str, str]] = set()
    for path in VPM_ROOT.rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        rel = path.relative_to(VPM_ROOT).parts
        if len(rel) == 1:
            continue  # top-level modules (facade) may import anything
        src, pkgpath = rel[0], rel[:-1]
        tree = ast.parse(path.read_text(encoding="utf-8"))
        _attach_parents(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.level >= 1:
                if _inside_type_checking(node):
                    continue
                base = pkgpath[: -(node.level - 1)] if node.level > 1 else pkgpath
                target = base + tuple(node.module.split("."))
                if target and target[0] in SUBSYSTEMS and target[0] != src:
                    edges.add((src, target[0]))
    return edges


def _inside_type_checking(node: ast.AST) -> bool:
    parent = getattr(node, "_parent", None)
    while parent is not None:
        if isinstance(parent, ast.If):
            names = {n.id for n in ast.walk(parent.test) if isinstance(n, ast.Name)}
            if "TYPE_CHECKING" in names:
                return True
        parent = getattr(parent, "_parent", None)
    return False


def _attach_parents(tree: ast.AST) -> None:
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child._parent = parent


@pytest.mark.unit
def test_no_forbidden_cross_subsystem_edges():
    edges = _collect_runtime_edges()
    violations = sorted(e for e in edges if e in FORBIDDEN_EDGES)
    assert violations == [], f"Forbidden dependency edges found: {violations}"


@pytest.mark.unit
def test_no_unlisted_cross_subsystem_edges():
    edges = _collect_runtime_edges()
    violations = sorted((a, b) for a, b in edges if b not in ALLOWED_EDGES.get(a, set()))
    assert violations == [], (
        f"Unlisted dependency edges found: {violations}. "
        "Add the edge to ALLOWED_EDGES and ARCHITECTURE.md only if it is intentional."
    )


@pytest.mark.unit
def test_all_documented_edges_are_used_or_benign():
    """Sanity: allowed sets only name existing subsystems."""
    for src, targets in ALLOWED_EDGES.items():
        assert src in SUBSYSTEMS, f"Unknown source subsystem {src!r} in ALLOWED_EDGES"
        for tgt in targets:
            assert tgt in SUBSYSTEMS, f"Unknown target subsystem {tgt!r} in ALLOWED_EDGES"
