"""Shared geometry, physics, mesh, timing, and sampling for the cylinder benchmark.

The fully meshed reference and coupled replacement import this module so the
cylinder STL, Reynolds number, cut-cell resolution, seed, and physical sample
times cannot silently diverge.  Environment variables select a documented
verification variant without editing the case:

``OPENONDA_GRID``
    ``g0``, ``g1`` (default), ``g2``, or ``smoke``.
``OPENONDA_DOMAIN``
    ``baseline`` (default) or ``large`` for the fully meshed reference.
``OPENONDA_DT_SCALE``
    Positive multiplier applied to the grid's nominal FVM time step.
``OPENONDA_END_TIME``
    Override the production horizon (default 60 convective units).
``OPENONDA_SEED_AMPLITUDE``
    Shared divergence-free antisymmetric seed (default 1e-3).
``OPENONDA_FIELD_OUTPUT_INTERVAL``
    Physical interval between compressed full-field/checkpoint backups
    (default 2 convective units).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path

import numpy as np

import openonda.fvm as fvm

SOURCE_DIR = Path(__file__).resolve().parent
CYLINDER_STL = SOURCE_DIR / "assets" / "cylinder_long.stl"

# Dimensional values are also the non-dimensional reference scales.
DIAMETER = 1.0
CYLINDER_LENGTH = 4.0
CYLINDER_Z_BOUNDS = (-0.5 * CYLINDER_LENGTH, 0.5 * CYLINDER_LENGTH)
STL_CYLINDER_LENGTH = 12.0
CYLINDER_STL_BOUNDS = (
    -0.5 * DIAMETER,
    0.5 * DIAMETER,
    -0.5 * DIAMETER,
    0.5 * DIAMETER,
    -0.5 * STL_CYLINDER_LENGTH,
    0.5 * STL_CYLINDER_LENGTH,
)
FREESTREAM_VELOCITY = (1.0, 0.0, 0.0)
FREESTREAM_SPEED = float(np.linalg.norm(FREESTREAM_VELOCITY))
DENSITY = 1.0
REYNOLDS = 150.0
KINEMATIC_VISCOSITY = FREESTREAM_SPEED * DIAMETER / REYNOLDS
REFERENCE_AREA = DIAMETER * CYLINDER_LENGTH
REFERENCE_LENGTH = DIAMETER
INITIAL_VELOCITY = FREESTREAM_VELOCITY

REFERENCE_DOMAINS = {
    "baseline": (-8.0, 20.0, -8.0, 8.0, -2.0, 2.0),
    "large": (-10.0, 25.0, -12.0, 12.0, -2.0, 2.0),
}

# These boxes stay fixed in the domain study.  The additional far-field cells
# therefore test blockage/outlet placement without changing near-body physics.
SHEAR_LAYER_BOX = (-0.75, 1.25, -0.90, 0.90, -2.25, 2.25)
# Keep the strongly resolved near wake around the vortex cores without paying
# for that isotropic resolution all the way to the spanwise slip planes. The
# wall and separating shear layers remain fine across the complete span.
NEAR_WAKE_BOX = (-0.75, 4.0, -1.25, 1.25, -1.50, 1.50)
DOWNSTREAM_WAKE_BOX = (3.5, 12.0, -2.0, 2.0, -2.75, 2.75)

# The coupled box is aligned with the baseline reference's 0.5-D background
# lattice.  Hence all near-body cut cells share the same Cartesian ancestry.
COUPLED_FVM_BOX = (-3.0, 4.5, -3.5, 3.5, -2.0, 2.0)
TRANSFER_REGION_BOX = (-2.75, 4.25, -3.25, 3.25, -1.50, 1.50)
VPM_DOMAIN = (-8.0, 20.0, -8.0, 8.0, -4.0, 4.0)


@dataclass(frozen=True)
class GridSpec:
    """One ratio-two spatial/temporal verification level."""

    name: str
    background: float
    surface: float
    shear_layer: float
    near_wake: float
    downstream_wake: float
    time_step: float
    target_cells: int
    first_cell_height: float
    wall_layers: int
    layer_growth: float
    transition_layers: int


GRID_SPECS = {
    "smoke": GridSpec(
        "smoke", 0.5, 0.125, 0.25, 0.25, 0.5, 0.005, 100_000,
        1.0 / 64.0, 4, 1.10, 4,
    ),
    "g0": GridSpec(
        "g0", 0.5, 0.125, 0.25, 0.25, 0.5, 0.004, 100_000,
        1.0 / 64.0, 6, 1.10, 10,
    ),
    "g1": GridSpec(
        "g1", 0.5, 1.0 / 16.0, 1.0 / 8.0, 1.0 / 8.0, 1.0 / 4.0, 0.002, 300_000,
        1.0 / 128.0, 8, 1.12, 10,
    ),
    "g2": GridSpec(
        "g2", 0.5, 1.0 / 32.0, 1.0 / 16.0, 1.0 / 16.0, 1.0 / 8.0, 0.001, 725_000,
        1.0 / 256.0, 10, 1.20, 8,
    ),
}

BOUNDARY_LAYER_INTERFACE_HALF_WIDTH = 0.75
# Re=150 is below the secondary three-dimensional wake instability. Keep the
# complete 4D physical span, but avoid duplicating the x-y refinement in z:
# sixteen native hexahedral slabs are sufficient to verify spanwise uniformity.
SPANWISE_CELL_SIZE = 0.25


def _choice(name: str, values: dict, default: str):
    selected = (os.environ.get(name) or default).strip().lower()
    if selected not in values:
        choices = ", ".join(values)
        raise ValueError(f"{name}={selected!r} is invalid; choose one of {choices}")
    return selected, values[selected]


def selected_grid() -> GridSpec:
    default = "smoke" if os.environ.get("OPENONDA_SMOKE", "0") == "1" else "g1"
    return _choice("OPENONDA_GRID", GRID_SPECS, default)[1]


def selected_reference_domain() -> tuple[float, ...]:
    return _choice("OPENONDA_DOMAIN", REFERENCE_DOMAINS, "baseline")[1]


def selected_domain_name() -> str:
    return _choice("OPENONDA_DOMAIN", REFERENCE_DOMAINS, "baseline")[0]


def positive_environment_float(name: str, default: float) -> float:
    value = float((os.environ.get(name) or str(default)).strip())
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value}")
    return value


def end_time() -> float:
    # Ten smoke-grid FVM steps exercise force/probe and line sampling as well
    # as mesh construction and linear solves.
    smoke_default = 0.2 if os.environ.get("OPENONDA_SMOKE", "0") == "1" else 60.0
    return positive_environment_float("OPENONDA_END_TIME", smoke_default)


def minimum_available_memory_gib(grid: GridSpec) -> float:
    """Conservative serial-run memory floor for this 14-GiB workstation."""
    return {"smoke": 2.0, "g0": 4.0, "g1": 6.0, "g2": 8.0}[grid.name]


def fvm_time_step(grid: GridSpec) -> float:
    return grid.time_step * positive_environment_float("OPENONDA_DT_SCALE", 1.0)


def seed_amplitude() -> float:
    value = float((os.environ.get("OPENONDA_SEED_AMPLITUDE") or "1e-3").strip())
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("OPENONDA_SEED_AMPLITUDE must be finite and non-negative")
    return value


def sampling_interval() -> float:
    """Force/probe period shared by every grid and both solvers."""
    # 0.1 is exactly representable on G0/G1/G2 and on the coupled VPM clock.
    # It still gives about 55 force samples per Re=150 shedding period.
    return 0.1


def field_output_interval() -> float:
    """Shared coarse full-field/checkpoint cadence in convective units."""
    return positive_environment_float("OPENONDA_FIELD_OUTPUT_INTERVAL", 2.0)


def _clip_box(box: tuple[float, ...], domain: tuple[float, ...]) -> tuple[float, ...]:
    clipped = []
    for axis in range(3):
        clipped.extend(
            (
                max(box[2 * axis], domain[2 * axis]),
                min(box[2 * axis + 1], domain[2 * axis + 1]),
            )
        )
    if not all(clipped[2 * axis] < clipped[2 * axis + 1] for axis in range(3)):
        raise ValueError(f"Refinement box {box} does not overlap domain {domain}")
    return tuple(clipped)


def build_mesh(
    domain: tuple[float, ...],
    grid: GridSpec,
    *,
    merge_outer_patch: str | None = None,
) -> fvm.AdaptiveCartesianMesher:
    """Construct the native cfMesh-derived conformal cylinder mesh."""
    if not CYLINDER_STL.is_file():
        raise FileNotFoundError(
            f"Missing {CYLINDER_STL}; run `python assets/generate_cylinder_stl.py`"
        )
    return fvm.AdaptiveCartesianMesher(
        domain=domain,
        max_cell_size=grid.background,
        surface_file=CYLINDER_STL,
        wall_patch_name="cylinder",
        surface_cell_size=grid.surface,
        boundary_layer=fvm.BoundaryLayerSpec(
            first_cell_height=grid.first_cell_height,
            layers=grid.wall_layers,
            growth_ratio=grid.layer_growth,
            transition_layers=grid.transition_layers,
            interface_half_width=BOUNDARY_LAYER_INTERFACE_HALF_WIDTH,
            spanwise_cell_size=SPANWISE_CELL_SIZE,
        ),
        surface_may_cross_domain_boundary=True,
        refinements=(
            fvm.BoxRefinement(
                _clip_box(SHEAR_LAYER_BOX, domain), grid.shear_layer, "shearLayerBox"
            ),
            fvm.BoxRefinement(_clip_box(NEAR_WAKE_BOX, domain), grid.near_wake, "nearWakeBox"),
            fvm.BoxRefinement(
                _clip_box(DOWNSTREAM_WAKE_BOX, domain),
                grid.downstream_wake,
                "downstreamWakeBox",
            ),
        ),
        merge_outer_patch=merge_outer_patch,
        preserve_outer_patches=("zmin", "zmax") if merge_outer_patch else (),
    )


def physical_sample_steps(dt: float, period: float | None = None) -> int:
    """Resolve an exact integer cadence or reject a time-grid mismatch."""
    requested = sampling_interval() if period is None else float(period)
    steps = int(round(requested / dt))
    if steps < 1 or not np.isclose(steps * dt, requested, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"Sampling period {requested} is not an integer multiple of dt={dt}")
    return steps


def reference_run_id(grid: GridSpec, domain_name: str, dt_scale: float) -> str:
    return f"{grid.name}_{domain_name}_dt{dt_scale:g}".replace(".", "p")


__all__ = [name for name in globals() if name.isupper()] + [
    "GridSpec",
    "build_mesh",
    "end_time",
    "field_output_interval",
    "fvm_time_step",
    "minimum_available_memory_gib",
    "physical_sample_steps",
    "positive_environment_float",
    "reference_run_id",
    "sampling_interval",
    "seed_amplitude",
    "selected_domain_name",
    "selected_grid",
    "selected_reference_domain",
]
