#!/usr/bin/env python3
"""Evaluate wake-induced downwash at the final flat-plate span stations.

The diagnostic uses the latest ``exp_moving_aoa05`` restart backup and
writes ``samples/exp_moving_aoa05/exp_moving_aoa05_downwash.csv``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from generate_surface import create_flat_plate, save_surface
from openonda.vpm import (
    Backup,
    ForceConfig,
    SmoothRampVLM,
    ViscousConfig,
    VPMSolver,
    VLMLoadingDistribution,
    VLMMeshSetup,
    VLMSurfaceSetup,
    VLMSetup,
    VPMSetup,
)
from theoretical_model import lifting_line_circulation


TUTORIAL_DIR = Path(__file__).resolve().parent.parent
SOLUTION_DIR = TUTORIAL_DIR / "solution"
CASE_NAME = "exp_moving_aoa05"

CHORD = 1.0
SPAN = 10.0
ANGLE_OF_ATTACK = 5.0
FREESTREAM_SPEED = 10.0
TIME_STEP_SIZE = 0.0125
RAMP_LENGTH = 0.6
CHORDWISE_PANELS = 8
SPANWISE_PANELS = 14


def latest_backup() -> Path:
    """Return the newest restart backup for the diagnostic case."""
    files = sorted(
        SOLUTION_DIR.glob(f"vpm_{CASE_NAME}_*.h5"),
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )
    if not files:
        raise FileNotFoundError(f"No restart backup found for {CASE_NAME}")
    return files[-1]


def travelled_distance(time: float, ramp_time: float) -> float:
    """Return the distance travelled by the smooth-ramp plate."""
    if time >= ramp_time:
        return 0.5 * FREESTREAM_SPEED * ramp_time + FREESTREAM_SPEED * (time - ramp_time)
    return (
        0.5 * FREESTREAM_SPEED * (time - ramp_time / math.pi * math.sin(math.pi * time / ramp_time))
    )


def build_solver() -> VPMSolver:
    """Build the VLM geometry needed to query the saved wake."""
    surface_file = TUTORIAL_DIR / "assets" / "surfaces" / "diag_downwash.json"
    surface_file.parent.mkdir(parents=True, exist_ok=True)
    save_surface(
        create_flat_plate(
            chord=CHORD,
            span=SPAN,
            angle_of_attack_degrees=ANGLE_OF_ATTACK,
            n_chordwise_panels=CHORDWISE_PANELS,
            n_spanwise_panels=SPANWISE_PANELS,
        ),
        str(surface_file),
    )

    ramp_time = 2.0 * RAMP_LENGTH * CHORD / FREESTREAM_SPEED
    vlm = VLMSetup(
        surfaces=(
            VLMSurfaceSetup(
                str(surface_file),
                kinematics=SmoothRampVLM(
                    final_velocity=[-FREESTREAM_SPEED, 0.0, 0.0],
                    acceleration_time=ramp_time,
                ),
            ),
        ),
        mesh=VLMMeshSetup.geometric(ratio=4.0, region="end"),
        density=1.0,
        kinematic_viscosity=1.0e-2,
        freestream_velocity=(FREESTREAM_SPEED, 0.0, 0.0),
        force=ForceConfig.kutta_joukowski(),
        sigma_factor=2.5,
        sample_surface_forces=True,
    )
    return VPMSolver(
        setup=VPMSetup.les_simulation(
            smagorinsky_coefficient=0.30,
            time_step_size=TIME_STEP_SIZE,
            vlm=vlm,
            viscous=ViscousConfig.cs(kinematic_viscosity=1.0e-2),
            freestream_velocity=[0.0, 0.0, 0.0],
            backup=Backup(
                directory=str(SOLUTION_DIR),
                log_directory=str(SOLUTION_DIR),
            ),
        )
    )


def spanwise_downwash(solver: VPMSolver, backup: Path) -> pd.DataFrame:
    """Evaluate VPM velocity at each VLM collocation point."""
    solver.load_backup(str(backup))

    vlm = solver.vlm_solver
    if vlm is None:
        raise RuntimeError("The downwash diagnostic requires a VLM solver")

    ramp_time = 2.0 * RAMP_LENGTH * CHORD / FREESTREAM_SPEED
    displacement = np.array([-travelled_distance(solver.time, ramp_time), 0.0, 0.0])
    n_panels = vlm.lattice.n_panels
    collocation_point = vlm.lattice.collocation_point.to_numpy()[:n_panels] + displacement
    normal = vlm.lattice.normal.to_numpy()[:n_panels]
    bound_vortex_midpoint = vlm.lattice.bound_vortex_midpoint.to_numpy()[:n_panels]
    velocity = solver.compute_velocity_at_points(collocation_point, include_freestream=False)

    rows = []
    for block in VLMLoadingDistribution.build_surface_grid_index(vlm, "flat_plate"):
        indices = block["orig_idx"]
        if indices is None:
            continue
        for span_index in range(block["ns"]):
            panel_indices = indices[span_index]
            panel_downwash = np.einsum(
                "ij,ij->i",
                velocity[panel_indices],
                normal[panel_indices],
            )
            rows.append(
                {
                    "span_index": span_index,
                    "span_coordinate": float(np.mean(bound_vortex_midpoint[panel_indices, 1])),
                    "vpm_downwash_velocity": float(np.mean(panel_downwash)),
                    "standard_deviation_vpm_downwash_velocity": float(np.std(panel_downwash)),
                }
            )
    return pd.DataFrame(rows).sort_values("span_index").reset_index(drop=True)


def add_lifting_line_reference(data: pd.DataFrame) -> pd.DataFrame:
    """Add the downwash required by Prandtl lifting-line theory."""
    y = data["span_coordinate"].to_numpy()
    downwash = data["vpm_downwash_velocity"].to_numpy()
    reference = lifting_line_circulation(
        y,
        reference_span=SPAN,
        reference_chord=CHORD,
        angle_of_attack_radians=math.radians(ANGLE_OF_ATTACK),
        freestream_speed=FREESTREAM_SPEED,
    )
    effective_angle = reference["section_lift_coefficient"].to_numpy() / (2.0 * math.pi)
    required_angle = np.degrees(math.radians(ANGLE_OF_ATTACK) - effective_angle)
    measured_angle = np.degrees(np.arctan2(-downwash, FREESTREAM_SPEED))

    result = data.copy()
    result["span_coordinate_normalized"] = 2.0 * y / SPAN
    result["measured_induced_angle_degrees"] = measured_angle
    result["required_induced_angle_degrees"] = required_angle
    result["downwash_delivery_ratio"] = np.divide(
        measured_angle,
        required_angle,
        out=np.full_like(measured_angle, np.nan),
        where=np.abs(required_angle) > 0.01,
    )
    return result


def main() -> None:
    backup = latest_backup()
    solver = build_solver()
    result = add_lifting_line_reference(spanwise_downwash(solver, backup))
    output = TUTORIAL_DIR / "samples" / CASE_NAME / f"{CASE_NAME}_downwash.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    solver.reset_gpu()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
