"""Plotting inputs from native solver metadata, geometry and sampled rotor motion."""

from __future__ import annotations

import argparse
import json
import os
from functools import lru_cache
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd

from openonda import plotting as theme
from ..setup import STATION_REFERENCE_RADIUS

ASSETS_DIR = Path(__file__).resolve().parent
CASE_DIR = ASSETS_DIR.parent
FIGURES_DIR = CASE_DIR / "figures"
_OUTPUT_TAG = os.environ.get("ROTOR_OUTPUT_TAG", "completion")
SOLUTION_DIR = CASE_DIR / "solution" / _OUTPUT_TAG if _OUTPUT_TAG else CASE_DIR / "solution"

# A complete native window is five nominal rotor revolutions.  The field
# stationarity diagnostic compares the first two whole revolutions with the
# final three, so every portion of the averaged window is evaluated.
OPERATING_WINDOW_REVOLUTIONS = 5
FIELD_STATIONARITY_COMPARISON_REVOLUTIONS = 5
IMPULSE_WINDOW_REVOLUTIONS = 3
SIGNAL_ONSET_RELATIVE_THRESHOLD = 0.01
SIGNAL_ONSET_PERSISTENCE_FRAMES = 3


def blade_geometry(vlm):
    """Read the actual first blade's radial chord/pitch schedule, without regenerating it."""
    geometry = theme.read_vlm_surface(vlm["surfaces"][0], ASSETS_DIR)
    rows = []
    # The tutorial's first blade is unrotated; its rotor axis is global x.
    for wing in geometry["wings"]:
        for segment in wing["segments"]:
            a, b, c, d = (np.array(segment["vertex_position"][key]) for key in "abcd")
            q_root, q_tip = 0.75 * a + 0.25 * d, 0.75 * b + 0.25 * c
            r_root, r_tip = np.linalg.norm(q_root[1:]), np.linalg.norm(q_tip[1:])
            point = 0.5 * (q_root + q_tip)
            chord_vector = 0.5 * (c + d - a - b)
            tangent = np.cross([1.0, 0.0, 0.0], point)
            tangent /= np.linalg.norm(tangent)
            pitch = np.arctan2(chord_vector[0], chord_vector @ tangent)
            rows.append(
                dict(
                    radial_position=0.5 * (r_root + r_tip),
                    width=r_tip - r_root,
                    chord=np.linalg.norm(chord_vector),
                    pitch=pitch,
                    inner_radius=r_root,
                    outer_radius=r_tip,
                )
            )
    return pd.DataFrame(rows).sort_values("radial_position")


@lru_cache(maxsize=1)
def rotor_inputs():
    """Load recorded inputs only when a plot runs; --help needs no saved results."""
    metadata = json.loads((SOLUTION_DIR / "vpm_metadata.json").read_text())
    configuration = metadata["configuration"]
    vlm = configuration["numerics"]["vlm"]
    blade = blade_geometry(vlm)
    omega = abs(float(vlm["surfaces"][0]["kinematics"]["angular_speed"]))
    speed = float(np.linalg.norm(configuration["numerics"]["freestream_velocity"]))
    radius = float(blade.outer_radius.max())
    return SimpleNamespace(
        metadata=metadata,
        solution_dir=SOLUTION_DIR,
        blade=blade,
        density=vlm["density"],
        n_blades=len(vlm["surfaces"]),
        samples_dir=CASE_DIR / "samples" / configuration["samplers"]["directory"],
        freestream_speed=speed,
        angular_velocity=omega,
        rotor_radius=radius,
        station_radius=STATION_REFERENCE_RADIUS,
        station_diameter=2.0 * STATION_REFERENCE_RADIUS,
        hub_radius=float(blade.inner_radius.min()),
        rotation_period=2 * np.pi / omega,
        tip_speed_ratio=omega * radius / speed,
        time_step_size=configuration["numerics"]["time_step_size"],
    )


def bem_reference():
    from openonda.rotor_theory import solve_blade_element_momentum

    p = rotor_inputs()
    return solve_blade_element_momentum(
        p.blade.radial_position,
        p.blade.chord,
        p.blade.pitch,
        p.n_blades,
        p.rotor_radius,
        p.freestream_speed,
        p.angular_velocity,
        hub_radius=p.hub_radius,
        radial_widths=p.blade.width,
        density=p.density,
    )


def performance():
    """Actual shaft power includes the prescribed ramp and all individual blades."""
    p = rotor_inputs()
    data = pd.read_csv(p.samples_dir / "vlm_forces.csv")
    reference = 0.5 * p.density * p.freestream_speed**2 * np.pi * p.rotor_radius**2
    data["CT"] = data.force_x / reference
    data["CP"] = data.rotational_power / (reference * p.freestream_speed)
    data["nominal_revolutions"] = data.time / p.rotation_period
    return data


def read_operating_point(*, revolutions=OPERATING_WINDOW_REVOLUTIONS):
    data = performance()
    tail = data[data.time > data.time.max() - revolutions * rotor_inputs().rotation_period]
    return float(tail.CT.mean()), float(tail.CP.mean())


def read_time_step():
    return rotor_inputs().time_step_size


def load_theme():
    theme.set_thesis_style()
    return dict(theme.COLORS), theme


def build_arg_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--format", choices=theme.EXPORT_FORMATS, default="png")
    parser.add_argument("--dpi", type=int, default=theme.DEFAULT_DPI)
    return parser


def build_rotor_style_map(colors):
    return {name: dict(style) for name, style in theme.ROTOR_STYLE.items()}


def impulse_history(
    force,
    integrals,
    *,
    density,
    time_step_size,
    flow_interval_steps=None,
    flow_interval_time=None,
    start_time,
    end_time,
):
    """Compare native surface loads and coupled impulse on identical accepted clocks.

    Forces are fluid-on-body. Fluid impulse and the recorded relaxation transfer
    are per density. Keep the raw balance and numerical transfer separate: removing
    a stabilization source from an accounting residual does not validate that source.
    """
    force_fields = [f"{prefix}_{axis}" for prefix in ("force", "unsteady_force") for axis in "xyz"]
    flow_fields = [
        f"{prefix}_{axis}"
        for prefix in ("coupled_linear_impulse", "pedrizzetti_cumulative_linear_impulse_transfer")
        for axis in "xyz"
    ]
    for name, data, columns in (("force", force, force_fields), ("flow", integrals, flow_fields)):
        missing = set(["time", *columns]) - set(data.columns)
        if missing:
            raise ValueError(f"Native {name} history lacks required columns: {sorted(missing)}")
        if len(data) < 2 or not np.isfinite(data[["time", *columns]].to_numpy()).all():
            raise ValueError(f"Native {name} history is incomplete or non-finite")
        if np.any(np.diff(data.time) <= 0):
            raise ValueError(f"Native {name} clocks must increase strictly")
    clock = force.time.to_numpy()
    if not np.allclose(np.diff(clock), time_step_size, rtol=1e-7, atol=1e-10):
        raise ValueError("Impulse validation requires surface loads at every accepted time step")
    if (flow_interval_steps is None) == (flow_interval_time is None):
        raise ValueError("Specify exactly one native flow cadence: steps or physical time")
    flow_cadence = (
        flow_interval_steps * time_step_size
        if flow_interval_steps is not None
        else float(flow_interval_time)
    )
    if not np.isfinite(flow_cadence) or flow_cadence <= 0.0:
        raise ValueError("Native flow cadence must be finite and positive")
    if not np.allclose(np.diff(integrals.time), flow_cadence, rtol=1e-7, atol=1e-10):
        raise ValueError("Native flow samples have gaps in their configured cadence")
    if (
        integrals.time.iloc[0] > start_time + flow_cadence
        or integrals.time.iloc[-1] < end_time - flow_cadence - 1e-10
    ):
        raise ValueError("Native flow samples do not cover the requested impulse window")
    selected = integrals[(integrals.time >= start_time) & (integrals.time <= end_time)]
    times = selected.time.to_numpy()
    if len(times) < 2 or times[0] < clock[0] or times[-1] > clock[-1]:
        raise ValueError("Force and flow histories do not cover the requested impulse window")
    # Search both neighbours to tolerate CSV roundoff without interpolating an
    # unsteady pressure difference onto a different physical interval.
    indices = np.searchsorted(clock, times).clip(0, len(clock) - 1)
    left = np.maximum(indices - 1, 0)
    indices = np.where(abs(clock[left] - times) < abs(clock[indices] - times), left, indices)
    if not np.allclose(clock[indices], times, rtol=0, atol=1e-8 * time_step_size):
        raise ValueError("Flow samples do not coincide with accepted force clocks")
    total = force[[f"force_{a}" for a in "xyz"]].to_numpy()
    pressure = force[[f"unsteady_force_{a}" for a in "xyz"]].to_numpy()
    kj = total - pressure
    intervals = np.diff(clock)[:, None] * (0.5 * (kj[:-1] + kj[1:]) + pressure[1:])
    cumulative = np.vstack((np.zeros(3), np.cumsum(intervals, axis=0)))
    loads = cumulative[indices] - cumulative[indices[0]]
    fluid = -density * selected[[f"coupled_linear_impulse_{a}" for a in "xyz"]].to_numpy()
    relaxation = (
        -density
        * selected[
            [f"pedrizzetti_cumulative_linear_impulse_transfer_{a}" for a in "xyz"]
        ].to_numpy()
    )
    return times, loads, fluid - fluid[0], relaxation - relaxation[0]
