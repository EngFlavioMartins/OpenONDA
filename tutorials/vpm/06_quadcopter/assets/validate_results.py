#!/usr/bin/env python3
"""Check native completion, rotor thrust/power, symmetry and periodic convergence."""

if not __package__:
    from pathlib import Path
    from openonda.tutorial_runner import case_package

    __package__ = case_package(Path(__file__).resolve().parents[1]) + ".assets"

import argparse
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from ._quadcopter_plots import FIGURES_DIR, bem_reference, performance, rotor_inputs, wake_windows


def validate_wake(p, end):
    """Require a complete final window and a stable induced velocity field."""
    config = p.metadata["configuration"]
    declared = {
        sampler["file_name"]: sampler
        for sampler in config["samplers"]["items"]
        if sampler["type"] == "SurfaceSampler"
    }
    failures = []
    try:
        planes = wake_windows(p.samples_dir, p.period)
    except (OSError, ValueError, KeyError) as error:
        return [f"invalid native wake samples: {error}"]
    if len(declared) != 2 or {plane["name"] for plane in planes} != set(declared):
        failures.append("missing declared downstream velocity planes")
    freestream = np.array(config["numerics"]["freestream_velocity"])
    for plane in planes:
        name, times = plane["name"], plane["times"]
        if name not in declared:
            continue
        cadence = declared[name]["schedule"]["interval"] * config["numerics"]["time_step_size"]
        tolerance = max(1e-10, cadence * 1e-6)
        if (
            len(times) < 4
            or end - times[-1] > cadence + tolerance
            or times[-1] > end + tolerance
            or times[0] > end - 6 * p.period + cadence + tolerance
            or np.any(np.diff(times) > cadence + tolerance)
        ):
            failures.append(f"{name}: incomplete final six-revolution velocity window")
            continue
        velocity = plane["velocity"]
        early = velocity[times <= end - 3 * p.period].mean(axis=0)
        late = velocity[times > end - 3 * p.period].mean(axis=0)
        # Subtract the freestream so a large uniform inflow cannot hide wake drift.
        scale = np.linalg.norm(velocity.mean(axis=0) - freestream)
        if scale <= 1e-12:
            failures.append(f"{name}: no resolved induced wake in the sampled field")
            continue
        drift = np.linalg.norm(late - early) / scale
        print(f"{name}: induced-velocity field drift={drift:.2%}")
        if not np.isfinite(drift) or drift > 0.03:
            failures.append(f"{name}: induced-velocity field drift exceeds3%")
    return failures


def validate_impulse(p, end, forces):
    """Check the final window's native bound-plus-wake axial impulse against thrust."""
    path = p.samples_dir / "flow_integrals.csv"
    flow = pd.read_csv(path)
    required = ["time", "coupled_linear_impulse_z"]
    if any(name not in flow for name in required):
        return ["native coupled-impulse samples are missing; momentum remains unqualified"]
    start = end - 6 * p.period
    times = flow.time.to_numpy()
    if (
        len(times) < 2
        or not np.isfinite(flow[required]).all().all()
        or np.any(np.diff(times) <= 0)
        or times[0] > start + 1e-10
        or times[-1] < end - 1e-10
    ):
        return ["incomplete or invalid final coupled-impulse history"]
    total = forces.groupby("time", sort=True).thrust.sum()
    if total.index.min() > start + 1e-10 or total.index.max() < end - 1e-10:
        return ["incomplete force history for the impulse comparison"]
    quadrature_times = np.r_[start, total.index[(total.index > start) & (total.index < end)], end]
    blade_impulse = trapezoid(np.interp(quadrature_times, total.index, total), quadrature_times)
    fluid_impulse = np.interp([start, end], times, flow.coupled_linear_impulse_z)
    ratio = -p.density * np.diff(fluid_impulse)[0] / max(blade_impulse, 1e-12)
    print(f"Bound-plus-wake impulse / integrated blade thrust: {ratio:.4f}")
    if blade_impulse <= 0 or not np.isfinite(ratio) or abs(ratio - 1) > 0.10:
        return [
            "coupled impulse / thrust differs from unity by more than10%; inspect wake health and boundary losses"
        ]
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-plot", action="store_true")
    args = parser.parse_args()
    p = rotor_inputs()
    config, state = p.metadata["configuration"], p.metadata["state"]
    steps = state["initial_step"] + config["run"]["steps"]
    end = state["initial_time"] + config["run"]["steps"] * config["numerics"]["time_step_size"]
    failures = []
    if p.metadata["lifecycle"]["status"] != "completed" or state["step"] != steps:
        print(
            f"[FAIL] Run incomplete; last recorded step {state['step']}/{steps}; convergence is unqualified"
        )
        return 1
    data = performance(p.samples_dir, p)
    if data.rotor.nunique() != p.n_rotors:
        failures.append("missing rotor force histories")
    bem = bem_reference(p)
    means = []
    for rotor, rows in data.groupby("rotor"):
        if (
            rows.time.duplicated().any()
            or not np.isfinite(rows.select_dtypes("number")).all().all()
        ):
            failures.append(f"{rotor}: duplicate or non-finite samples")
        if rows.step.max() < steps - config["numerics"]["vlm"]["logging_interval_steps"]:
            failures.append(f"{rotor}: incomplete force/power samples")
        tail = rows[rows.time > end - 6 * p.period]
        cadence = (
            config["numerics"]["vlm"]["logging_interval_steps"]
            * config["numerics"]["time_step_size"]
        )
        if len(tail) < 4 or tail.time.max() - tail.time.min() < 6 * p.period - 2 * cadence:
            failures.append(f"{rotor}: fewer than six revolutions of final force samples")
            continue
        means.append(tail.thrust.mean())
        for column, ref in (("thrust", bem.attrs["thrust"]), ("input_power", bem.attrs["power"])):
            values = tail[column].to_numpy()
            half = len(values) // 2
            drift = abs(values[:half].mean() - values[half:].mean()) / max(
                abs(values.mean()), 1e-12
            )
            difference = abs(values.mean() / ref - 1)
            print(
                f"{rotor} {column}: mean={values.mean():.6g}, BEM={ref:.6g}, difference={difference:.2%}, tail drift={drift:.2%}"
            )
            if values.mean() <= 0 or drift > 0.03 or difference > 0.20:
                failures.append(
                    f"{rotor} {column}: positive load, <=3% tail drift and <=20% isolated-BEM difference required"
                )
        ct, cp = tail.CT.mean(), tail.CP.mean()
        advance = p.climb / (p.omega * p.radius)
        ideal_cp = 0.5 * ct * (advance + np.sqrt(advance**2 + 2 * ct))
        if cp < 0.98 * ideal_cp:
            failures.append(f"{rotor}: power falls below the ideal axial-momentum requirement")
    if means:
        symmetry = np.ptp(means) / max(abs(np.mean(means)), 1e-12)
        print(f"Rotor mean-thrust spread={symmetry:.2%}")
        if symmetry > 0.01:
            failures.append("rotor thrust symmetry differs by more than1%")
    failures.extend(validate_wake(p, end))
    failures.extend(validate_impulse(p, end, data))
    if not args.pre_plot:
        for name in ("quadcopter_performance", "quadcopter_wake", "quadcopter_vorticity_history"):
            for extension in ("png", "pdf"):
                if not (FIGURES_DIR / f"{name}.{extension}").is_file():
                    failures.append(f"missing {name}.{extension}")
    print("\n".join(f"[FAIL] {item}" for item in failures) or "[OK] Quadcopter checks passed")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
