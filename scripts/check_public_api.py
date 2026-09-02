#!/usr/bin/env python3
"""Statically reject removed identifiers on public and serialized surfaces."""

from __future__ import annotations

import ast
from pathlib import Path
import sys

PUBLIC_SURFACE_FILES = (
    "source/solvers/fvm/config/types.py",
    "source/solvers/fvm/core/solver.py",
    "source/solvers/vpm/config/setup.py",
    "source/solvers/vpm/config/artifacts.py",
    "source/solvers/vpm/config/case.py",
    "source/solvers/vpm/config/health.py",
    "source/solvers/vpm/config/types.py",
    "source/solvers/vpm/config/advection.py",
    "source/solvers/vpm/config/divergence_relaxation.py",
    "source/solvers/vpm/config/filament_refinement.py",
    "source/solvers/vpm/config/stabilization.py",
    "source/solvers/vpm/config/state.py",
    "source/solvers/vpm/config/stretching.py",
    "source/solvers/vpm/config/turbulence.py",
    "source/solvers/vpm/config/velocity.py",
    "source/solvers/vpm/config/viscous.py",
    "source/solvers/vpm/initialization/data.py",
    "source/solvers/vpm/initialization/disturbances.py",
    "source/solvers/vpm/initialization/distributions/rectangular.py",
    "source/solvers/vpm/initialization/distributions/triangular.py",
    "source/solvers/vpm/initialization/distributions/toroidal.py",
    "source/solvers/vpm/initialization/distributions/cylindrical.py",
    "source/solvers/vpm/initialization/flows/vortex_ring.py",
    "source/solvers/vpm/initialization/flows/vortex_filament.py",
    "source/solvers/vpm/initialization/flows/doublet.py",
    "source/solvers/vpm/initialization/flows/taylor_green.py",
    "source/solvers/vpm/initialization/flows/isotropic_turbulence.py",
    "source/solvers/vpm/particles/container.py",
    "source/solvers/vpm/core/solver.py",
    "source/solvers/vpm/io/backup.py",
    "source/solvers/vpm/diagnostics/conservation.py",
    "source/solvers/vpm/diagnostics/offline.py",
    "source/solvers/vpm/diagnostics/resolution.py",
    "source/solvers/vpm/boundary_elements/vlm/config.py",
    "source/solvers/vpm/boundary_elements/vlm/solver/diagnostics.py",
    "source/solvers/vpm/boundary_elements/vlm/solver/vlm_solver.py",
    "source/coupler/config/types.py",
    "source/coupler/solver.py",
    "source/coupler/boundary.py",
    "source/coupler/vorticity_transfer.py",
    "openonda/fvm/__init__.py",
    "openonda/fvm/mesher.py",
    "openonda/vpm.py",
    "openonda/coupler.py",
)

FORBIDDEN_DEFINITIONS = frozenset(
    {
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
        "vpm_domain_bounds",
        "dvh_rd_ratio",
        "n_sources",
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
        "CheckpointManager",
        "BackupManager",
        "DEFAULT_BACKUP_FILENAME",
        "BackupSystem",
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
        "avg_particle_radius",
        "anti_diffuse_flag",
        "epsilon_W",
        "epsilon_w",
        "ring_strength",
        "ring_thickness",
        "normalize_circulation",
        "volume_cpu",
    }
)
FORBIDDEN_SELF_ATTRIBUTES = frozenset({"fvm", "vpm", "transfer"})
FORBIDDEN_PREFIXES = ("regen_",)
FORBIDDEN_NAME_WORDS = frozenset(
    {"policy", "policies", "contract", "contracts", "checkpoint", "checkpoints"}
)


def _name_words(name: str) -> tuple[str, ...]:
    """Split snake/camel names into words for exact nomenclature checks."""
    import re

    snake_case = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    return tuple(part for part in re.split(r"[^a-z0-9]+", snake_case) if part)


def _definitions(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            yield node.arg, node.lineno
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            yield node.name, node.lineno
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            yield node.target.id, node.lineno


def _self_attribute_assignments(tree: ast.AST):
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


def scan_public_api(root: Path) -> list[str]:
    """Return removed-name findings without importing the package."""
    findings: list[str] = []
    for relative in PUBLIC_SURFACE_FILES:
        path = root / relative
        if not path.is_file():
            findings.append(f"{relative}:required public-surface file is missing")
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError) as error:
            findings.append(f"{relative}:cannot scan public surface: {error}")
            continue
        for name, line_number in _definitions(tree):
            if (
                name in FORBIDDEN_DEFINITIONS
                or name.startswith(FORBIDDEN_PREFIXES)
                or FORBIDDEN_NAME_WORDS.intersection(_name_words(name))
            ):
                findings.append(f"{relative}:{line_number}:removed public definition {name!r}")
        for name, line_number in _self_attribute_assignments(tree):
            if name in FORBIDDEN_SELF_ATTRIBUTES:
                findings.append(f"{relative}:{line_number}:removed stored attribute 'self.{name}'")
    return findings


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    findings = scan_public_api(root)
    if findings:
        print("Removed public API definitions:", file=sys.stderr)
        print("\n".join(findings), file=sys.stderr)
        return 1
    print("Public API static scan passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
