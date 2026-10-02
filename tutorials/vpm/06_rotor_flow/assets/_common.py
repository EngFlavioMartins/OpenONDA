"""Plotting inputs from native solver metadata, geometry and sampled rotor motion."""

from __future__ import annotations

import argparse
import json
import os
import warnings
from functools import lru_cache
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd

from openonda import plotting as theme
from ..setup import ROTOR_RADIUS, WAKE_PLANE_DIAMETERS

ASSETS_DIR = Path(__file__).resolve().parent
CASE_DIR = ASSETS_DIR.parent
FIGURES_DIR = CASE_DIR / "figures"
_OUTPUT_TAG = os.environ.get("ROTOR_OUTPUT_TAG", "")
SOLUTION_DIR = CASE_DIR / "solution" / _OUTPUT_TAG if _OUTPUT_TAG else CASE_DIR / "solution"

# A complete native window is five nominal rotor revolutions.  The field
# stationarity diagnostic compares the first two whole revolutions with the
# final three, so every portion of the averaged window is evaluated.
OPERATING_WINDOW_REVOLUTIONS = 5
FIELD_STATIONARITY_COMPARISON_REVOLUTIONS = 5
IMPULSE_WINDOW_REVOLUTIONS = 3
SIGNAL_ONSET_RELATIVE_THRESHOLD = 0.01
SIGNAL_ONSET_PERSISTENCE_FRAMES = 3
REQUIRED_PLANE_NAMES = tuple(f"wake_{distance}D" for distance in WAKE_PLANE_DIAMETERS)


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
        station_radius=ROTOR_RADIUS,
        station_diameter=2.0 * ROTOR_RADIUS,
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
    data = accepted_history(pd.read_csv(p.samples_dir / "vlm_forces.csv"), p.metadata)
    reference = 0.5 * p.density * p.freestream_speed**2 * np.pi * p.rotor_radius**2
    data["CT"] = data.force_x / reference
    data["CP"] = data.rotational_power / (reference * p.freestream_speed)
    data["nominal_revolutions"] = data.time / p.rotation_period
    return data


def accepted_history(data, metadata):
    """Select recorded accepted history while leaving live sampler tails untouched.

    Samplers can flush newer rows before the next metadata checkpoint. Such
    rows remain stored but cannot define a plot's accepted operating window.
    """
    state = metadata["state"]
    horizon = float(state["time"])
    dt = float(metadata["configuration"]["numerics"]["time_step_size"])
    tolerance = max(1e-10, dt * 1e-6)
    if not np.isfinite(horizon) or dt <= 0 or not np.isfinite(dt):
        raise ValueError("invalid recorded accepted horizon")
    times = data.time.to_numpy()
    if not np.isfinite(times).all():
        raise ValueError("native history has non-finite timestamps")
    if "step" in data:
        steps = data.step.to_numpy()
        current = steps >= state["initial_step"]
        expected = state["initial_time"] + (steps - state["initial_step"]) * dt
        if (
            not np.isfinite(steps).all()
            or np.any(steps < 0)
            or np.any(steps != np.rint(steps))
            or not np.allclose(times[current], expected[current], rtol=0, atol=tolerance)
            or np.any(times[~current] > state["initial_time"] + tolerance)
        ):
            raise ValueError("native history step/time clock disagrees with recorded solver clock")
        accepted = (steps <= state["step"]) & (times <= horizon + tolerance)
    else:
        accepted = times <= horizon + tolerance
    if not np.any(accepted):
        raise ValueError("native history has no rows within the accepted horizon")
    if not np.all(accepted):
        warnings.warn(
            f"Plotting accepted history through t={horizon:.9g} s; "
            "newer sampler rows are retained on disk and excluded from this plot.",
            stacklevel=2,
        )
    return data.loc[accepted].copy()


def read_operating_point(
    *, revolutions=OPERATING_WINDOW_REVOLUTIONS, window_start=None, window_end=None
):
    data = performance()
    if window_start is not None or window_end is not None:
        if window_start is None or window_end is None or window_start >= window_end:
            raise ValueError("Operating point requires an ordered complete window")
        times = data.time.to_numpy()
        if len(times) < 2 or np.any(np.diff(times) <= 0) or not np.isfinite(times).all():
            raise ValueError("Operating point requires finite increasing force clocks")
        if times[0] > window_start or times[-1] < window_end:
            raise ValueError("Force history does not bracket the native field window")
        selected = times[(times > window_start) & (times < window_end)]
        integration_times = np.concatenate(([window_start], selected, [window_end]))
        from scipy.integrate import trapezoid

        means = []
        for column in ("CT", "CP"):
            values = data[column].to_numpy()
            if not np.isfinite(values).all():
                raise ValueError("Operating point requires finite force coefficients")
            means.append(
                float(
                    trapezoid(np.interp(integration_times, times, values), integration_times)
                    / (window_end - window_start)
                )
            )
        return tuple(means)
    end = float(data.time.max())
    return read_operating_point(
        window_start=end - revolutions * rotor_inputs().rotation_period, window_end=end
    )


def run_status(p=None):
    p = rotor_inputs() if p is None else p
    state = p.metadata["state"]
    requested = (
        state["initial_time"] + p.metadata["configuration"]["run"]["steps"] * p.time_step_size
    )
    complete = (
        p.metadata["lifecycle"]["status"] == "completed"
        and state["step"] == state["initial_step"] + p.metadata["configuration"]["run"]["steps"]
    )
    return (
        f"Completed: {state['time']:.2g} s"
        if complete
        else f"Incomplete: {state['time']:.2g} / {requested:.2g} s"
    )


def read_time_step():
    return rotor_inputs().time_step_size


def load_theme():
    theme.set_thesis_style()
    return dict(theme.COLORS), theme


def rotor_subplots(nrows, *, height_cm, sharex=False):
    """Compact rotor panels using the authored OpenONDA Matplotlib style."""
    import matplotlib.pyplot as plt

    theme.set_thesis_style()
    fig, axes = plt.subplots(
        nrows,
        1,
        squeeze=False,
        sharex=sharex,
        figsize=(theme.MAX_FIGURE_WIDTH_CM * theme.CM, height_cm * theme.CM),
    )
    theme.centered_subplots_adjust(
        fig,
        outer=0.16,
        bottom=1.30 / height_cm,
        top=1 - 0.25 / height_cm,
        hspace=0.10 if sharex else 0.38,
    )
    return fig, axes[:, 0]


def save_rotor_figure(fig, path, *, figure_format="both", dpi=None):
    outer = {
        "rotor_performance": 0.100,
        "rotor_loading_validation": 0.129,
        "rotor_wake_planes": 0.113,
        "rotor_streamwise": 0.139,
    }[Path(path).stem]
    top_padding_cm = 0.17 if Path(path).stem == "rotor_loading_validation" else 0.11
    theme.centered_subplots_adjust(
        fig, outer=outer, top=1 - top_padding_cm / (fig.get_figheight() / theme.CM)
    )
    theme.validate_thesis_figure(fig, fig.axes)
    return theme.export_figure(fig, path, figure_format=figure_format, dpi=dpi)


def build_arg_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--format", choices=theme.FORMAT_CHOICES, default="both")
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
