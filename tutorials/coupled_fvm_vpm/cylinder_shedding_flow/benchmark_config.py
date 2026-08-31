"""Single source of truth for the cylinder reference-flow grid study.

``reference_flow/reference_flow.py --dx <wall-size> -name <case>`` selects
the explicit four-zone mesh used by the study.  The named legacy grids remain
available only for the coupled tutorial; they are not used by the grid runner.
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

DIAMETER = 1.0
CYLINDER_LENGTH = 4.0
CYLINDER_Z_BOUNDS = (-0.5 * CYLINDER_LENGTH, 0.5 * CYLINDER_LENGTH)
STL_CYLINDER_LENGTH = 12.0
CYLINDER_STL_BOUNDS = (-0.5, 0.5, -0.5, 0.5, -6.0, 6.0)
FREESTREAM_VELOCITY = (1.0, 0.0, 0.0)
FREESTREAM_SPEED = 1.0
DENSITY = 1.0
REYNOLDS = 150.0
KINEMATIC_VISCOSITY = 1.0 / REYNOLDS
REFERENCE_AREA = DIAMETER * CYLINDER_LENGTH
REFERENCE_LENGTH = DIAMETER
INITIAL_VELOCITY = FREESTREAM_VELOCITY

REFERENCE_DOMAINS = {
    "baseline": (-8.0, 20.0, -8.0, 8.0, -2.0, 2.0),
    "large": (-10.0, 25.0, -12.0, 12.0, -2.0, 2.0),
}

# Retained only for the coupled tutorial's pre-existing entry point.
COUPLED_FVM_BOX = (-3.0, 4.5, -3.5, 3.5, -2.0, 2.0)
TRANSFER_REGION_BOX = (-2.75, 4.25, -3.25, 3.25, -1.5, 1.5)
VPM_DOMAIN = (-8.0, 20.0, -8.0, 8.0, -4.0, 4.0)

SPANWISE_CELL_SIZE = 0.5


@dataclass(frozen=True)
class GridSpec:
    """Mesh and time-step parameters for one reference-flow calculation."""

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
    study_dx: float | None = None


# These values keep the coupled tutorial importable.  The grid study always
# goes through :func:`grid_study_spec`, which uses the explicit mesh contract.
GRID_SPECS = {
    "smoke": GridSpec("smoke", 0.5, 0.125, 0.25, 0.25, 0.5, 0.005, 100_000, 1 / 64, 4, 1.1, 4),
    "g0": GridSpec("g0", 0.5, 0.125, 0.25, 0.25, 0.5, 0.004, 100_000, 1 / 64, 6, 1.1, 10),
    "g1": GridSpec("g1", 0.5, 1 / 16, 1 / 8, 1 / 8, 1 / 4, 0.002, 300_000, 1 / 128, 8, 1.12, 10),
}


def positive_environment_float(name: str, default: float) -> float:
    """Read one finite, strictly positive floating-point environment value."""
    value = float((os.environ.get(name) or str(default)).strip())
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value}")
    return value


def _choice(name: str, values: dict, default: str):
    selected = (os.environ.get(name) or default).strip().lower()
    if selected not in values:
        raise ValueError(f"{name}={selected!r} is invalid; choose one of {', '.join(values)}")
    return selected, values[selected]


def grid_study_spec(dx: float, name: str) -> GridSpec:
    """Return the exact four-zone body-fitted grid requested for this study.

    The only free spatial value is the wall spacing ``dx``.  It applies on
    the cylinder O-grid and its collar; the other Cartesian zones are exactly
    ``2*dx``, ``4*dx``, and ``12*dx``.  ``dx=D/(12 n)`` is required so that
    every study-zone interface aligns with the baseline-domain lattice.
    """
    dx = float(dx)
    if not math.isfinite(dx) or dx <= 0.0:
        raise ValueError("Grid-study dx must be finite and positive")
    ratio = 1.0 / (12.0 * dx)
    if not np.isclose(ratio, round(ratio), rtol=0.0, atol=1.0e-10):
        raise ValueError(
            "Grid-study dx must be D/(12 n) so the 1:2:4:12 zones align exactly; "
            f"received dx={dx:g}"
        )
    if not name or any(character in name for character in "/\\"):
        raise ValueError("Grid-study case name must be a non-empty simple directory name")
    return GridSpec(
        name=name,
        background=12.0 * dx,
        surface=dx,
        shear_layer=2.0 * dx,
        near_wake=4.0 * dx,
        downstream_wake=4.0 * dx,
        time_step=0.001,
        target_cells=1_000_000,
        first_cell_height=dx,
        wall_layers=1,
        layer_growth=1.0,
        transition_layers=1,
        study_dx=dx,
    )


def selected_grid() -> GridSpec:
    study_dx = os.environ.get("OPENONDA_GRID_STUDY_DX", "").strip()
    if study_dx:
        name = os.environ.get("OPENONDA_GRID_STUDY_NAME", "grid_study").strip()
        return grid_study_spec(float(study_dx), name)
    default = "smoke" if os.environ.get("OPENONDA_SMOKE", "0") == "1" else "g1"
    return _choice("OPENONDA_GRID", GRID_SPECS, default)[1]


def selected_reference_domain() -> tuple[float, ...]:
    return _choice("OPENONDA_DOMAIN", REFERENCE_DOMAINS, "baseline")[1]


def selected_domain_name() -> str:
    return _choice("OPENONDA_DOMAIN", REFERENCE_DOMAINS, "baseline")[0]


def end_time() -> float:
    default = 0.2 if os.environ.get("OPENONDA_SMOKE", "0") == "1" else 60.0
    return positive_environment_float("OPENONDA_END_TIME", default)


def minimum_available_memory_gib(grid: GridSpec) -> float:
    """A practical free-memory floor before constructing one study mesh."""
    if grid.study_dx is not None:
        return 4.0 if grid.study_dx >= 1.0 / 36.0 else 6.0
    return {"smoke": 2.0, "g0": 4.0, "g1": 6.0}[grid.name]


def fvm_time_step(grid: GridSpec) -> float:
    return grid.time_step * positive_environment_float("OPENONDA_DT_SCALE", 1.0)


def seed_amplitude() -> float:
    value = float((os.environ.get("OPENONDA_SEED_AMPLITUDE") or "5e-2").strip())
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("OPENONDA_SEED_AMPLITUDE must be finite and non-negative")
    return value


def sampling_interval() -> float:
    """Frequent force/probe cadence for Cd/Cl and shedding-frequency estimates."""
    return 0.02


def field_output_interval() -> float:
    """Sparse inspectable full-field and checkpoint cadence."""
    return positive_environment_float("OPENONDA_FIELD_OUTPUT_INTERVAL", 2.5)


def physical_sample_steps(dt: float, period: float | None = None) -> int:
    requested = sampling_interval() if period is None else float(period)
    steps = int(round(requested / dt))
    if steps < 1 or not np.isclose(steps * dt, requested, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"Sampling period {requested} is not an integer multiple of dt={dt}")
    return steps


def reference_run_id(grid: GridSpec, domain_name: str, dt_scale: float) -> str:
    return f"{grid.name}_{domain_name}_dt{dt_scale:g}".replace(".", "p")


def build_mesh(
    domain: tuple[float, ...],
    grid: GridSpec,
    *,
    merge_outer_patch: str | None = None,
) -> fvm.AdaptiveCartesianMesher | fvm.ExplicitCylinderGridMesher:
    """Build the requested body-fitted mesh without any IBM treatment."""
    if not CYLINDER_STL.is_file():
        raise FileNotFoundError(f"Missing cylinder surface: {CYLINDER_STL}")
    if grid.study_dx is not None:
        if merge_outer_patch is not None:
            raise ValueError("The reference grid study does not merge outer patches")
        return fvm.ExplicitCylinderGridMesher(
            domain=domain,
            surface_file=CYLINDER_STL,
            wall_patch_name="cylinder",
            wall_cell_size=grid.study_dx,
            near_body_half_width=2.0,
            wake_half_width=4.0,
            wake_xmin=-4.0,
            interface_half_width=2.0 / 3.0,
            spanwise_cell_size=SPANWISE_CELL_SIZE,
        )

    # Compatibility mesh for the independent coupled tutorial; this is not a
    # member of the reference-flow grid study.
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
            interface_half_width=0.75,
            spanwise_cell_size=SPANWISE_CELL_SIZE,
        ),
        surface_may_cross_domain_boundary=True,
        refinements=(),
        merge_outer_patch=merge_outer_patch,
        preserve_outer_patches=("zmin", "zmax") if merge_outer_patch else (),
    )


__all__ = [
    "COUPLED_FVM_BOX",
    "CYLINDER_LENGTH",
    "CYLINDER_STL",
    "CYLINDER_STL_BOUNDS",
    "CYLINDER_Z_BOUNDS",
    "DENSITY",
    "DIAMETER",
    "FREESTREAM_SPEED",
    "FREESTREAM_VELOCITY",
    "GRID_SPECS",
    "GridSpec",
    "INITIAL_VELOCITY",
    "KINEMATIC_VISCOSITY",
    "REFERENCE_AREA",
    "REFERENCE_LENGTH",
    "REYNOLDS",
    "SPANWISE_CELL_SIZE",
    "TRANSFER_REGION_BOX",
    "VPM_DOMAIN",
    "build_mesh",
    "end_time",
    "field_output_interval",
    "fvm_time_step",
    "grid_study_spec",
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
