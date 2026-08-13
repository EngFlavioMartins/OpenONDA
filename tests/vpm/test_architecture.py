"""Architecture tests: enforce the VPM subsystem dependency boundaries.

Mirrors ``source/solvers/VPM/ARCHITECTURE.md``.  These tests statically parse
the import graph (no Taichi backend needed) and fail when a runtime import
crosses a forbidden or unlisted subsystem boundary.  ``if TYPE_CHECKING:``
imports are ignored; ``io/logging.py`` is a leaf allowed from any subsystem.

Also guards against reintroducing imports from VPM namespaces that were deleted
during the `utils/` -> `io/sampling/` + `numerics/fourier_integrals.py`
restructure (see ``git log`` for the refactor).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

VPM_ROOT = Path(__file__).resolve().parents[2] / "source" / "solvers" / "VPM"
REPO_ROOT = VPM_ROOT.parents[2]

SUBSYSTEMS = {d.name for d in VPM_ROOT.iterdir() if d.is_dir() and (d / "__init__.py").exists()}

# Dotted VPM namespaces deleted by the refactor.  Nothing may import any path
# under these anymore (compatibility was deliberately not provided).
DELETED_NAMESPACES = (
    "source.solvers.VPM.utils",
    "source.solvers.VPM.diagnostics.fourier_integrals",
)

_SCAN_SKIP_PARTS = {
    ".git",
    "__pycache__",
    ".venv",
    "build",
    "dist",
    "node_modules",
    "figures",
    "samples",
}


def _file_package_for(file: Path) -> list[str]:
    """Absolute dotted package ('source.solvers.VPM...') containing `file`."""
    return ["source"] + list(file.relative_to(REPO_ROOT).parent.parts)


def _all_repo_py_files():
    for path in REPO_ROOT.rglob("*.py"):
        if any(s in set(path.relative_to(REPO_ROOT).parts) for s in _SCAN_SKIP_PARTS):
            continue
        yield path


def _banned(dotted: str) -> bool:
    return any(dotted == ns or dotted.startswith(ns + ".") for ns in DELETED_NAMESPACES)


def _import_targets(file: Path, tree: ast.AST):
    base = _file_package_for(file)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            if node.level:
                up = node.level - 1
                prefix = base[:-up] if up else base
                yield ".".join(prefix + node.module.split("."))
            else:
                yield node.module


def _collect_deleted_namespace_imports() -> list[str]:
    hits: list[str] = []
    for path in sorted(_all_repo_py_files()):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for dotted in _import_targets(path, tree):
            if _banned(dotted):
                hits.append(f"{path.relative_to(REPO_ROOT)} imports {dotted!r}")
    return hits


@pytest.mark.unit
def test_no_imports_from_deleted_vpm_namespaces():
    """No production/tutorial/test code may import the refactored-away paths.

    The `utils/` package (field_samplers, flow_models, offline_diagnostics,
    simulation_checks) moved to `io/sampling/` + `initial_conditions/` and
    `diagnostics/fourier_integrals.py` moved to `numerics/fourier_integrals.py`.
    Importing the deleted dotted paths would silently grab a stale module or,
    worse, resurrect an old duplicate behaviour.  Use the new canonical paths.
    """
    hits = _collect_deleted_namespace_imports()
    assert hits == [], "Imports from deleted VPM namespaces found:\n  " + "\n  ".join(hits)


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
