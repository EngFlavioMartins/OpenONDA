#!/usr/bin/env python3
"""Run a real two-way VPM--VLM tandem-wing interaction case.

The incoming structure is a declarative Gaussian vortex ring.  It is advanced
by the VPM owner clock, while the attached two-surface VLM solves a temporary
boundary system at every particle Runge--Kutta stage and releases the
accepted wake rows only once per accepted step.  The case is deliberately
small enough for a CPU smoke run, but it is a real coupled run: no synthetic
particle protocol, manually prescribed endpoint positions, or post-hoc VLM
solve is used.

Typical usage from this directory::

    python setup.py
    python setup.py --steps 4 --precision f32

The first command writes accepted forces, loading, particle, metadata, and
restart records below ``samples/`` and ``solution/``.
The companion study script consumes the same declarative builder for the
refinement and backend/restart evidence tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import openonda.vpm as vpm


TUTORIAL_DIR = Path(__file__).resolve().parent
CASE_NAME = "real_tandem_vpm_vlm"
SAMPLE_DIRECTORY = "tandem"
SURFACE_FILE = TUTORIAL_DIR / "assets" / "tandem_delta_surface.json"

# Physical and numerical scales.  These values intentionally leave the
# incoming ring close to the outboard edge to exercise wake/surface coupling.
WING_SEPARATION = 0.60
FREESTREAM_VELOCITY = (4.0, 0.0, 0.0)
TIME_STEP_SIZE = 0.01
N_STEPS = 30
PARTICLE_SPACING = 0.045
PARTICLE_CORE_RADIUS_RATIO = 1.5
RING_RADIUS = 0.13
RING_TUBE_RADIUS = 0.045
RING_CORE_RADIUS = 0.10
RING_CIRCULATION = 0.70
RING_CENTRE = (-0.45, 0.49, 0.14)
MAX_N_PARTICLES = 100_000


def _surface_mapping(*, n_chordwise_panels: int = 3, n_spanwise_panels: int = 4) -> dict:
    """Return the portable delta-wing mapping used by both tandem surfaces."""
    sweep = np.radians(45.0)
    incidence = np.radians(8.0)
    offset = 0.45 * np.tan(sweep)
    chord = 0.60
    tip_chord = 0.08
    half_span = 0.45
    base = {
        "a": np.array([0.0, 0.0, 0.0]),
        "b": np.array([offset, half_span, 0.0]),
        "c": np.array([offset + tip_chord, half_span, 0.0]),
        "d": np.array([chord, 0.0, 0.0]),
    }

    def rotate(point: np.ndarray) -> np.ndarray:
        return np.array(
            [
                point[0] * np.cos(incidence) + point[2] * np.sin(incidence),
                point[1],
                -point[0] * np.sin(incidence) + point[2] * np.cos(incidence),
            ]
        )

    vertices = {name: rotate(point).tolist() for name, point in base.items()}
    return {
        "uid": "qualification_delta",
        "wings": [
            {
                "uid": "main_wing",
                "symmetry": 2,
                "segments": [
                    {
                        "uid": "segment_0",
                        "vertex_position": vertices,
                        "n_chordwise_panels": n_chordwise_panels,
                        "n_spanwise_panels": n_spanwise_panels,
                        "airfoils": {"inner": "flat", "outer": "flat"},
                    }
                ],
            }
        ],
        "refs": {
            "area": 2.0 * 0.5 * (chord + tip_chord) * half_span,
            "chord": chord,
            "span": 2.0 * half_span,
            "geometry_centre": [chord / 2.0, 0.0, 0.0],
            "reference_point": [chord / 2.0, 0.0, 0.0],
        },
    }


def _write_surface(
    path: Path = SURFACE_FILE,
    *,
    n_chordwise_panels: int = 3,
    n_spanwise_panels: int = 4,
) -> Path:
    """Write the self-contained surface asset and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _surface_mapping(
                n_chordwise_panels=n_chordwise_panels,
                n_spanwise_panels=n_spanwise_panels,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _incoming_ring(
    *,
    particle_spacing: float = PARTICLE_SPACING,
    centre: tuple[float, float, float] = RING_CENTRE,
) -> vpm.VortexRing:
    """Declare the finite Gaussian ring that is advanced by the VPM owner."""
    distribution = vpm.ToroidalDistribution(
        ring_radius=RING_RADIUS,
        tube_radius=RING_TUBE_RADIUS,
        spacing=particle_spacing,
        core_radius_ratio=PARTICLE_CORE_RADIUS_RATIO,
        centre=centre,
        axis="x",
    )
    return vpm.VortexRing(
        kinematic_viscosity=0.0,
        centre=centre,
        axis=(1.0, 0.0, 0.0),
        radius=RING_RADIUS,
        circulation=RING_CIRCULATION,
        vortex_core_radius=RING_CORE_RADIUS,
        distribution=distribution,
        group_id=7,
    )


def build_case(
    *,
    n_steps: int = N_STEPS,
    compute_device: str = "CPU",
    precision: str = "f64",
    directory: str | Path = TUTORIAL_DIR,
    sample_directory: str = SAMPLE_DIRECTORY,
    name: str = CASE_NAME,
    time_step_size: float = TIME_STEP_SIZE,
    particle_spacing: float = PARTICLE_SPACING,
    n_chordwise_panels: int = 3,
    n_spanwise_panels: int = 4,
    boundary_response: str = "responsive",
    ring_centre: tuple[float, float, float] = RING_CENTRE,
    include_ring: bool = True,
    tandem_surfaces: bool = True,
) -> vpm.VPMCase:
    """Build the coupled case without allocating a solver or particles."""
    surface_file = _write_surface(
        Path(directory) / "assets" / SURFACE_FILE.name,
        n_chordwise_panels=n_chordwise_panels,
        n_spanwise_panels=n_spanwise_panels,
    )
    surfaces = [
        vpm.VLMSurfaceSetup(
            str(surface_file),
            name="front_wing",
            translation=(0.0, 0.0, 0.0),
            group_id=0,
        )
    ]
    if tandem_surfaces:
        surfaces.append(
            vpm.VLMSurfaceSetup(
                str(surface_file),
                name="rear_wing",
                translation=(WING_SEPARATION, 0.0, 0.0),
                group_id=1,
            )
        )
    vlm_setup = vpm.VLMSetup(
        surfaces=tuple(surfaces),
        mesh=vpm.VLMMeshSetup(),
        dtype=precision,
        linear_solver="SCIPY",
        kinematic_viscosity=0.0,
        density=1.0,
        freestream_velocity=FREESTREAM_VELOCITY,
        logging_interval_steps=1,
        sample_surface_forces=True,
        wake_core_overlap=2.5,
        boundary_response=boundary_response,
        force=vpm.ForceConfig.kutta_joukowski(unsteady=True),
    )
    return vpm.VPMCase(
        name=name,
        directory=directory,
        initial_conditions=(
            (_incoming_ring(particle_spacing=particle_spacing, centre=ring_centre),)
            if include_ring
            else ()
        ),
        numerics=vpm.Numerics(
            time_step_size=time_step_size,
            compute_device=compute_device,
            precision=precision,
            write_precision=precision,
            integrator=vpm.SSPRK3(),
            induction=vpm.DirectInduction(stretching_scheme="transposed"),
            viscous=vpm.ViscousConfig(scheme="NONE", kinematic_viscosity=0.0),
            turbulence=vpm.TurbulenceConfig.dns(),
            stabilization=vpm.StabilizationConfig.disabled(),
            vlm=vlm_setup,
            freestream_velocity=FREESTREAM_VELOCITY,
            max_n_particles=MAX_N_PARTICLES,
            random_seed=42,
        ),
        backup=vpm.Backup(
            interval_steps=max(1, min(10, n_steps)),
            directory="solution/tandem",
            log_directory="solution/tandem",
        ),
        samplers=vpm.Samplers(
            samples=(vpm.FlowIntegralsSampler(schedule=vpm.EverySteps(1), initial=True),),
            directory=sample_directory,
        ),
        run=vpm.RunPlan(steps=n_steps, final_backup=True, health_limit_action="RAISE"),
    )


def _sample_root(case_directory: Path, sample_directory: str) -> Path:
    """Resolve the owner-managed sample directory."""
    candidate = case_directory / "samples" / sample_directory
    return candidate if candidate.exists() else case_directory / "samples"


def _write_run_summary(
    case_directory: Path,
    *,
    sample_directory: str,
    precision: str,
    compute_device: str,
    n_steps: int,
    case_name: str,
    time_step_size: float,
    boundary_response: str,
) -> Path:
    """Summarize actual owner-emitted tables without inventing observations."""
    sample_root = _sample_root(case_directory, sample_directory)
    force_path = sample_root / "vlm_surface_forces.csv"
    if not force_path.is_file():
        raise FileNotFoundError(f"coupled VLM force table was not written: {force_path}")
    forces = pd.read_csv(force_path)
    rows: list[dict[str, object]] = []
    for surface, group in forces.groupby("surface", sort=True):
        group = group.sort_values("time")
        time = group["time"].to_numpy(dtype=float)
        lift = group.get("lift", pd.Series(0.0, index=group.index)).to_numpy(dtype=float)
        drag = group.get("drag", pd.Series(0.0, index=group.index)).to_numpy(dtype=float)
        rows.append(
            {
                "case": case_name,
                "precision": precision,
                "compute_device": compute_device,
                "boundary_response": boundary_response,
                "surface": surface,
                "n_force_samples": int(len(group)),
                "final_lift": float(lift[-1]),
                "peak_abs_lift": float(np.max(np.abs(lift))),
                "integrated_lift": float(np.trapezoid(lift, time)) if len(time) > 1 else 0.0,
                "final_drag": float(drag[-1]),
            }
        )
    summary_path = sample_root / "qualification_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    manifest = {
        "case": case_name,
        "physics": "real VPM vortex ring + responsive two-surface VLM",
        "boundary_response": boundary_response,
        "compute_device": compute_device,
        "precision": precision,
        "time_step_size": float(time_step_size),
        "accepted_steps_requested": n_steps,
        "force_table": str(force_path.relative_to(case_directory)),
        "summary": str(summary_path.relative_to(case_directory)),
        "n_force_rows": int(len(forces)),
    }
    (sample_root / "qualification_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return summary_path


def run(
    *,
    n_steps: int = N_STEPS,
    compute_device: str = "CPU",
    precision: str = "f64",
    directory: str | Path = TUTORIAL_DIR,
    sample_directory: str = SAMPLE_DIRECTORY,
    name: str = CASE_NAME,
    time_step_size: float = TIME_STEP_SIZE,
    particle_spacing: float = PARTICLE_SPACING,
    n_chordwise_panels: int = 3,
    n_spanwise_panels: int = 4,
    boundary_response: str = "responsive",
    ring_centre: tuple[float, float, float] = RING_CENTRE,
    include_ring: bool = True,
    tandem_surfaces: bool = True,
) -> Path:
    """Run the real coupled case and return the generated summary path."""
    case_directory = Path(directory).resolve()
    case = build_case(
        n_steps=n_steps,
        compute_device=compute_device,
        precision=precision,
        directory=case_directory,
        sample_directory=sample_directory,
        name=name,
        time_step_size=time_step_size,
        particle_spacing=particle_spacing,
        n_chordwise_panels=n_chordwise_panels,
        n_spanwise_panels=n_spanwise_panels,
        boundary_response=boundary_response,
        ring_centre=ring_centre,
        include_ring=include_ring,
        tandem_surfaces=tandem_surfaces,
    )
    vpm.VPMSolver(case).run()
    summary = _write_run_summary(
        case_directory,
        sample_directory=sample_directory,
        precision=precision,
        compute_device=compute_device,
        n_steps=n_steps,
        case_name=name,
        time_step_size=time_step_size,
        boundary_response=boundary_response,
    )
    print(f"Wrote real coupled qualification artifacts to {summary.parent}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--compute-device", choices=("CPU", "AUTO"), default="CPU")
    parser.add_argument("--precision", choices=("f32", "f64"), default="f64")
    args = parser.parse_args()
    run(n_steps=args.steps, compute_device=args.compute_device, precision=args.precision)


if __name__ == "__main__":
    main()
