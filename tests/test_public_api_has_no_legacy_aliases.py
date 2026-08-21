"""docs/nomenclature.md + docs/rename-manifest.md: the public/setup/serialization
surface must not contain pre-rename names, and obsolete top-level public names
must fail rather than silently resolve to the renamed target.

The AST sweep below is deliberately scoped to *definitions* (dataclass fields,
function/method signatures, class names, and ``self.<attr> =`` assignments) in
the setup/config/facade/core modules that make up the public and serialized
surface -- not arbitrary source text, so it cannot flag a legitimate compact
math variable (``nu``, ``dt``, ``U`` ...) used inside a numerical kernel, a
VLM ``circulation`` (the true scalar bound-vortex Gamma, which is NOT
renamed), an OpenFOAM on-disk contract string, or an archived migration
reader. See docs/nomenclature.md for the full old-name -> new-name contract.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Files that make up the public/setup/serialization surface: solver setups,
# subsystem configs, the openonda facade, and the core solver/coupler modules
# that own the renamed public methods and attributes.
SCANNED_FILES = [
    "source/solvers/FVM/config/types.py",
    "source/solvers/FVM/core/solver.py",
    "source/solvers/VPM/config/setup.py",
    "source/solvers/VPM/config/types.py",
    "source/solvers/VPM/config/advection.py",
    "source/solvers/VPM/config/divergence_relaxation.py",
    "source/solvers/VPM/config/filament_refinement.py",
    "source/solvers/VPM/config/stabilization.py",
    "source/solvers/VPM/config/state.py",
    "source/solvers/VPM/config/stretching.py",
    "source/solvers/VPM/config/turbulence.py",
    "source/solvers/VPM/config/velocity.py",
    "source/solvers/VPM/config/viscous.py",
    "source/solvers/VPM/core/solver.py",
    "source/solvers/VPM/diagnostics/conservation.py",
    "source/solvers/VPM/diagnostics/offline.py",
    "source/solvers/VPM/diagnostics/resolution.py",
    "source/solvers/VPM/boundary_elements/vlm/config.py",
    "source/solvers/VPM/boundary_elements/vlm/solver/diagnostics.py",
    "source/solvers/VPM/boundary_elements/vlm/solver/vlm_solver.py",
    "source/coupler/config/types.py",
    "source/coupler/solver.py",
    "source/coupler/boundary.py",
    "source/coupler/blending.py",
    "source/coupler/vorticity_transfer.py",
    "openonda/fvm.py",
    "openonda/vpm.py",
    "openonda/coupler.py",
]

# Exact identifiers (dataclass fields, function/method names, class names,
# `self.<attr>` targets) that must never appear on the scanned surface.
FORBIDDEN_EXACT = {
    # PLAN.md §28 examples
    "processing_unit",
    "particles_kernel",
    "max_targets",
    "characteristic_distance",
    "backup_frequency",
    "backup_directory",
    "backup_file_name",
    "coupler_backup_period",
    "initial_p",
    "alpha_u",
    "alpha_p",
    "momentum_tol",
    "pressure_tol",
    "V_inf",
    "U_ref",
    "V_external",
    "V_wake_field",
    # docs/nomenclature.md additions
    "vpm_domain_bounds",
    "dvh_rd_ratio",
    "number_of_particles",
    "num_sources",
    "SetFlowModel",
    "CachedParticleProperty",
    "transfer_region_box",
    "bc_patch_name",
    "vpm_bc_mode",
    "transfer_prune_vorticity_min",
    "overlap_zone_dead_zone_width",
    "overlap_zone_ramp_width",
    "momentum_maxiter",
    "amg_maxiter",
    "MeshConfig",
    "SchemesConfig",
    "ExecutionConfig",
    "OutputSetup",
    "LogConfig",
    "DynamicMeshConfig",
    "evolve",
    "update_state",
    "backup_solution",
    "continue_from_backup",
    "DEFAULT_BACKUP_FILENAME",
    "BackupSystem",
    # Particle alpha_p [L^3/T] must not be exposed as scalar VLM circulation
    # Gamma [L^2/T].
    "particles_strengths",
    "total_strength",
    "update_particle_circulations",
    "centroid_of_circulation",
    "centroids_of_circulation",
    "compute_total_circulation",
    "circulation_bound",
    "circulation_wake",
    "circulation_total",
    "gamma_bound",
    "gamma_wake",
}

# `self.<attr> = ...` targets that must never appear: these are *stored
# attribute* renames (docs/nomenclature.md "Coupler" table). Checked only
# against genuine `self.X = ...` assignments, not local variables or function
# parameters of the same short name -- a function taking a `vpm` parameter or
# a local `vpm = self.vpm_solver` alias is exactly the compact, unambiguous
# local usage the nomenclature contract explicitly allows.
FORBIDDEN_SELF_ATTRS = {
    "fvm",  # self.fvm -> self.fvm_solver (coupler)
    "vpm",  # self.vpm -> self.vpm_solver (coupler)
    "transfer",  # self.transfer -> self.vorticity_transfer (coupler)
    "blending",  # self.blending -> self.blending_zone (coupler)
}

# Identifier *prefixes* that must never appear (the canonical form is the
# same word with a different, longer prefix/suffix).
FORBIDDEN_PREFIXES = (
    "regen_",  # -> regeneration_*
)


def _iter_definitions(tree: ast.AST):
    """Yield (identifier, lineno) for arg/def/dataclass-field definitions."""
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            yield node.arg, node.lineno
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            yield node.name, node.lineno
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            yield node.target.id, node.lineno


def _iter_self_attr_assignments(tree: ast.AST):
    """Yield (attr, lineno) for every `self.<attr> = ...` assignment."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                yield target.attr, node.lineno


def _forbidden_hits(path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits = []
    for name, lineno in _iter_definitions(tree):
        if name in FORBIDDEN_EXACT or name.startswith(FORBIDDEN_PREFIXES):
            hits.append((name, lineno))
    for name, lineno in _iter_self_attr_assignments(tree):
        if name in FORBIDDEN_SELF_ATTRS:
            hits.append((f"self.{name}", lineno))
    return hits


@pytest.mark.parametrize("relative_path", SCANNED_FILES)
def test_no_legacy_names_defined_on_public_surface(relative_path):
    path = REPO_ROOT / relative_path
    if not path.exists():
        pytest.skip(f"{relative_path} not present in this tree")
    hits = _forbidden_hits(path)
    assert not hits, f"{relative_path} defines forbidden legacy identifier(s): " + ", ".join(
        f"{name!r} (line {lineno})" for name, lineno in hits
    )


@pytest.mark.parametrize(
    "module,name",
    [
        ("openonda.fvm", "Solver"),
        ("openonda.vpm", "Solver"),
        ("openonda.fvm", "setup_fvm_solver"),
        ("openonda.vpm", "setup_vpm_solver"),
        ("openonda.coupler", "setup_coupler"),
    ],
)
def test_pre_rename_public_name_is_gone(module, name):
    mod = __import__(module, fromlist=[name])
    assert not hasattr(mod, name)
