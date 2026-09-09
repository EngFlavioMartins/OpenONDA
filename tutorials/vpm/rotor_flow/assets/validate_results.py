#!/usr/bin/env python3
"""Check completion, native data, stationarity, impulse and matched rotor theory."""

if not __package__:
    from pathlib import Path
    from openonda.tutorial_runner import case_package

    __package__ = case_package(Path(__file__).resolve().parents[1]) + ".assets"

import argparse
import numpy as np
import pandas as pd

from ._common import FIGURES_DIR, bem_reference, performance, rotor_inputs
from .plot_rotor_wake_planes import plane_profiles, relative_drift


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-plot", action="store_true")
    args = parser.parse_args()
    p = rotor_inputs()
    config, state = p.metadata["configuration"], p.metadata["state"]
    final_step = state["initial_step"] + config["run"]["steps"]
    end = state["initial_time"] + config["run"]["steps"] * p.time_step_size
    failures = []
    if p.metadata["lifecycle"]["status"] != "completed" or state["step"] != final_step:
        print(
            f"[FAIL] Run incomplete; last recorded step {state['step']}/{final_step}; convergence is unqualified"
        )
        return 1
    data = performance()
    if not np.isfinite(data.select_dtypes("number")).all().all() or data.time.duplicated().any():
        failures.append("non-finite or duplicate force samples")
    cadence = config["numerics"]["vlm"]["logging_interval_steps"] * p.time_step_size
    if data.time.max() < end - cadence - 1e-9:
        failures.append("force samples do not cover the configured horizon")
    tail = data[data.time > end - 6 * p.rotation_period]
    if len(tail) < 4 or tail.time.max() - tail.time.min() < 6 * p.rotation_period - 2 * cadence:
        print("[FAIL] Fewer than six revolutions of final force samples")
        return 1
    bem = bem_reference()
    for key, reference_key in (("CT", "thrust_coefficient"), ("CP", "power_coefficient")):
        mean = tail[key].mean()
        reference = bem.attrs[reference_key]
        drift = relative_drift(tail[key])
        error = abs(mean / reference - 1)
        print(
            f"{key}: mean={mean:.6f}, BEM={reference:.6f}, difference={error:.2%}, window drift={drift:.2%}"
        )
        if mean <= 0 or drift > 0.02 or error > 0.15:
            failures.append(
                f"{key}: positive load, <=2% tail drift and <=15% BEM difference required"
            )
    integrals = pd.read_csv(p.samples_dir / "flow_integrals.csv")
    window = integrals[integrals.time > end - 3 * p.rotation_period]
    # Native impulse is per density. The entire wake remains inside this run's bounds.
    slope = np.polyfit(window.time, window.linear_impulse_x, 1)[0]
    mean_thrust = tail.force_x.mean()
    ratio = -p.density * slope / mean_thrust
    print(f"Wake impulse force / blade thrust: {ratio:.4f}")
    if not np.isfinite(ratio) or abs(ratio - 1) > 0.10:
        failures.append("wake impulse / thrust differs from unity by more than10%")
    for row in plane_profiles(p):
        drift = relative_drift(row["disc_means"])
        print(f"{row['name']}: disc-mean velocity drift={drift:.2%}")
        if drift > 0.01:
            failures.append(f"{row['name']}: >1% velocity drift; run longer")
    if not args.pre_plot:
        for name in ("rotor_performance", "rotor_loading_validation", "rotor_wake_planes"):
            for extension in ("png", "pdf"):
                if not (FIGURES_DIR / f"{name}.{extension}").is_file():
                    failures.append(f"missing {name}.{extension}")
    print("\n".join(f"[FAIL] {message}" for message in failures) or "[OK] Rotor checks passed")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
