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
        "runtime",
    },
    "physics": {"particles", "numerics", "kernels", "acceleration", "config", "io"},
    "stabilization": {"particles", "numerics", "diagnostics", "config", "io"},
    "coupling": {"physics", "particles", "boundary_elements", "core"},
    "diagnostics": {"particles", "physics", "numerics", "config", "io"},
    "io": {"config", "diagnostics", "particles"},
    "config": {"boundary_elements", "runtime"},
    "initial_conditions": {"particles"},
    "particles": {
        "config",
        "initial_conditions",
        "io",
    },
    "boundary_elements": {"config", "io"},
    "turbulence": {"config"},
    "acceleration": {"config"},
    "kernels": {"config"},
    "numerics": {"config"},
    "runtime": {"config"},
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


def _subsystem_of_dotted(dotted: str) -> str | None:
    """First VPM subsystem in an absolute dotted path, or ``None``.

    ``source.solvers.VPM.<subsystem>...`` maps to ``<subsystem>`` when that name
    is a real subsystem package.  Anything outside the VPM package (or a
    top-level VPM module such as ``factory``) returns ``None``.
    """
    parts = dotted.split(".")
    if parts[:3] != ["source", "solvers", "VPM"]:
        return None
    if len(parts) > 3 and parts[3] in SUBSYSTEMS:
        return parts[3]
    return None


def _import_dotted(node: ast.AST) -> str | None:
    """Resolve one import node to an absolute dotted path, or ``None``.

    Handles every internal import form the tree can express:

    - ``import source.solvers.VPM.<sub>...``        (``ast.Import``)
    - ``from source.solvers.VPM.<sub>... import ...`` (absolute ``ast.ImportFrom``)
    - ``from .something import ...``                (relative ``ast.ImportFrom``)
    - ``from ..something import ...``               (relative ``ast.ImportFrom``)
    """
    if isinstance(node, ast.ImportFrom):
        if node.level:
            module_parts = tuple(node.module.split(".")) if node.module else ()
            up = node.level - 1
            prefix = _dotted_pkgpath[:-up] if up else _dotted_pkgpath
            return ".".join(prefix + module_parts)
        return node.module
    if isinstance(node, ast.Import):
        return node.names[0].name
    return None


_dotted_pkgpath: tuple[str, ...] = ()


def _collect_runtime_edges(root: Path = VPM_ROOT) -> set[tuple[str, str]]:
    """Scan the tree for runtime cross-subsystem imports.

    Relative and absolute internal imports are normalized to the same dotted
    path, so ``from ..stabilization import ...`` and
    ``from source.solvers.VPM.stabilization import ...`` produce the identical
    edge and are enforced identically.
    """
    global _dotted_pkgpath
    edges: set[tuple[str, str]] = set()
    for path in root.rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        rel = path.relative_to(root).parts
        if len(rel) == 1:
            continue  # top-level modules (facade) may import anything
        src = rel[0]
        _dotted_pkgpath = ("source", "solvers", "VPM") + rel[:-1]
        tree = ast.parse(path.read_text(encoding="utf-8"))
        _attach_parents(tree)
        for node in ast.walk(tree):
            if _inside_type_checking(node):
                continue
            dotted = _import_dotted(node)
            target = _subsystem_of_dotted(dotted or "")
            if target is not None and target != src:
                edges.add((src, target))
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


def _write_synthetic_module(root: Path, module_package: list[str], source: str) -> None:
    """Create ``source/solvers/VPM/<...>/<name>.py`` under a synthetic repo root."""
    vpm = root / "source" / "solvers" / "VPM"
    for subsystem in ("particles", "physics", "stabilization"):
        init = vpm / subsystem / "__init__.py"
        init.parent.mkdir(parents=True, exist_ok=True)
        init.touch()
    target = vpm.joinpath(*module_package)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")


@pytest.mark.unit
def test_absolute_forbidden_import_is_detected(tmp_path):
    """An absolute internal import must hit the same dependency policy as the
    equivalent relative import (physics -> stabilization is forbidden)."""
    _write_synthetic_module(
        tmp_path,
        ["physics", "module.py"],
        "from source.solvers.VPM.stabilization.manager import StabilizationManager\n",
    )
    edges = _collect_runtime_edges(tmp_path / "source" / "solvers" / "VPM")
    assert ("physics", "stabilization") in edges, (
        "absolute internal import escaped the dependency scanner"
    )


@pytest.mark.unit
def test_relative_and_absolute_imports_map_to_the_same_edge(tmp_path):
    """The two spellings of the same dependency must be indistinguishable."""
    _write_synthetic_module(
        tmp_path,
        ["physics", "rel.py"],
        "from ..stabilization.manager import StabilizationManager\n",
    )
    rel_edges = _collect_runtime_edges(tmp_path / "source" / "solvers" / "VPM")
    assert ("physics", "stabilization") in rel_edges

    _write_synthetic_module(
        tmp_path,
        ["physics", "abs.py"],
        "from source.solvers.VPM.stabilization.manager import StabilizationManager\n",
    )
    abs_edges = _collect_runtime_edges(tmp_path / "source" / "solvers" / "VPM")
    assert ("physics", "stabilization") in abs_edges


@pytest.mark.unit
def test_absolute_module_import_form_is_detected(tmp_path):
    """``import source.solvers.VPM.stabilization`` must be caught as well."""
    _write_synthetic_module(
        tmp_path,
        ["particles", "module.py"],
        "import source.solvers.VPM.stabilization\n",
    )
    edges = _collect_runtime_edges(tmp_path / "source" / "solvers" / "VPM")
    assert ("particles", "stabilization") in edges, (
        "``import source.solvers.VPM.<sub>`` escaped the dependency scanner"
    )
