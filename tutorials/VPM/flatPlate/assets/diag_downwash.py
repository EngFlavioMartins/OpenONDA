#!/usr/bin/env python3
"""Evaluate wake-induced downwash at the final flat-plate span stations.

The diagnostic uses the latest ``exp_moving_aoa05`` restart checkpoint and
writes ``samples/exp_moving_aoa05/exp_moving_aoa05_downwash.csv``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from generate_surface import create_flat_plate, save_surface
from openonda.vpm import (
    BackupSystem,
    ForceConfig,
    SmoothRampVLM,
    Solver,
    VLMLoadingDistribution,
    VLMMeshSetup,
    VLMSurfaceSetup,
    VLMSetup,
    VPMSetup,
)
from theoretical_model import liftingline_circulation


TUTORIAL_DIR = Path(__file__).resolve().parent.parent
SOLUTION_DIR = TUTORIAL_DIR / "solution"
CASE_NAME = "exp_moving_aoa05"

CHORD = 1.0
SPAN = 10.0
ANGLE_OF_ATTACK = 5.0
FREESTREAM_SPEED = 10.0
TIME_STEP = 0.0125
RAMP_LENGTH = 0.6
CHORDWISE_PANELS = 8
SPANWISE_PANELS = 14


def latest_checkpoint() -> Path:
    """Return the newest restart checkpoint for the diagnostic case."""
    files = sorted(
        SOLUTION_DIR.glob(f"vpm_{CASE_NAME}_*.h5"),
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )
    if not files:
        raise FileNotFoundError(f"No restart checkpoint found for {CASE_NAME}")
    return files[-1]


def travelled_distance(time: float, ramp_time: float) -> float:
    """Return the distance travelled by the smooth-ramp plate."""
    if time >= ramp_time:
        return 0.5 * FREESTREAM_SPEED * ramp_time + FREESTREAM_SPEED * (time - ramp_time)
    return (
        0.5 * FREESTREAM_SPEED * (time - ramp_time / math.pi * math.sin(math.pi * time / ramp_time))
    )


def build_solver() -> Solver:
    """Build the VLM geometry needed to query the saved wake."""
    surface_file = TUTORIAL_DIR / "assets" / "surfaces" / "diag_downwash.json"
    surface_file.parent.mkdir(parents=True, exist_ok=True)
    save_surface(
        create_flat_plate(
            chord=CHORD,
            span=SPAN,
            alpha=ANGLE_OF_ATTACK,
            n_chord=CHORDWISE_PANELS,
            n_span=SPANWISE_PANELS,
        ),
        str(surface_file),
    )

    ramp_time = 2.0 * RAMP_LENGTH * CHORD / FREESTREAM_SPEED
    vlm = VLMSetup(
        surfaces=(
            VLMSurfaceSetup(
                str(surface_file),
                kinematics=SmoothRampVLM(
                    U_final=[-FREESTREAM_SPEED, 0.0, 0.0],
                    acceleration_time=ramp_time,
                ),
            ),
        ),
        mesh=VLMMeshSetup.geometric(ratio=4.0, region="end"),
        density=1.0,
        viscosity=1.0e-2,
        freestream_velocity=(FREESTREAM_SPEED, 0.0, 0.0),
        force=ForceConfig.kutta_joukowski(),
        sigma_factor=2.5,
        sample_surface_forces=True,
    )
    return Solver(
        setup=VPMSetup.les_simulation(
            cs=0.30,
            time_step_size=TIME_STEP,
            vlm=vlm,
            background_velocity=[0.0, 0.0, 0.0],
            backup_directory=str(SOLUTION_DIR),
        )
    )


def spanwise_downwash(solver: Solver, checkpoint: Path) -> pd.DataFrame:
    """Evaluate VPM velocity at each VLM collocation point."""
    BackupSystem._load_numerical_data(solver, str(checkpoint))

    vlm = solver.vlm_solver
    if vlm is None:
        raise RuntimeError("The downwash diagnostic requires a VLM solver")

    ramp_time = 2.0 * RAMP_LENGTH * CHORD / FREESTREAM_SPEED
    displacement = np.array([-travelled_distance(solver.flow_time, ramp_time), 0.0, 0.0])
    number_of_panels = vlm.lattice.num_panels
    collocation = vlm.lattice.collocation.to_numpy()[:number_of_panels] + displacement
    normals = vlm.lattice.normals.to_numpy()[:number_of_panels]
    bound_midpoints = vlm.lattice.bound_midpoints.to_numpy()[:number_of_panels]
    velocity = solver.compute_target_velocities(collocation, include_freestream=False)

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
                normals[panel_indices],
            )
            rows.append(
                {
                    "span_index": span_index,
                    "y": float(np.mean(bound_midpoints[panel_indices, 1])),
                    "w_VPM": float(np.mean(panel_downwash)),
                    "w_VPM_std": float(np.std(panel_downwash)),
                }
            )
    return pd.DataFrame(rows).sort_values("span_index").reset_index(drop=True)


def add_lifting_line_reference(data: pd.DataFrame) -> pd.DataFrame:
    """Add the downwash required by Prandtl lifting-line theory."""
    y = data["y"].to_numpy()
    downwash = data["w_VPM"].to_numpy()
    reference = liftingline_circulation(
        y,
        b=SPAN,
        c=CHORD,
        alpha_rad=math.radians(ANGLE_OF_ATTACK),
        U_inf=FREESTREAM_SPEED,
    )
    effective_angle = reference["cl"].to_numpy() / (2.0 * math.pi)
    required_angle = np.degrees(math.radians(ANGLE_OF_ATTACK) - effective_angle)
    measured_angle = np.degrees(np.arctan2(-downwash, FREESTREAM_SPEED))

    result = data.copy()
    result["y_over_b"] = 2.0 * y / SPAN
    result["alpha_i_VPM_deg"] = measured_angle
    result["alpha_i_required_deg"] = required_angle
    result["delivery_ratio"] = np.divide(
        measured_angle,
        required_angle,
        out=np.full_like(measured_angle, np.nan),
        where=np.abs(required_angle) > 0.01,
    )
    return result


def main() -> None:
    checkpoint = latest_checkpoint()
    solver = build_solver()
    result = add_lifting_line_reference(spanwise_downwash(solver, checkpoint))
    output = TUTORIAL_DIR / "samples" / CASE_NAME / f"{CASE_NAME}_downwash.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    solver.reset_gpu()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
