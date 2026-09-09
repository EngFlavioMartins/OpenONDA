#!/usr/bin/env python3
"""Check completed native samples and repeatability of the final heave cycles."""

if not __package__:
    from pathlib import Path
    from openonda.tutorial_runner import case_package

    __package__ = case_package(Path(__file__).resolve().parents[1]) + ".assets"

import argparse
import json
import numpy as np
import pandas as pd
from ._delta_wing_plots import (
    CASE_DIR,
    FIGURES_DIR,
    SAMPLES_DIR,
    force_history,
    last_cycles,
    motion_period,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-plot", action="store_true")
    args = parser.parse_args()
    metadata = json.loads((CASE_DIR / "solution/vpm_metadata.json").read_text())
    config, state = metadata["configuration"], metadata["state"]
    steps = state["initial_step"] + config["run"]["steps"]
    dt = config["numerics"]["time_step_size"]
    failures = []
    if metadata["lifecycle"]["status"] != "completed" or state["step"] != steps:
        print(
            f"[FAIL] Run incomplete; last recorded step {state['step']}/{steps}; convergence is unqualified"
        )
        return 1
    data = force_history()
    period = motion_period(data)
    for surface, rows in data.groupby("surface"):
        if (
            rows.time.duplicated().any()
            or not np.isfinite(rows.select_dtypes("number")).all().all()
        ):
            failures.append(f"{surface}: duplicate or non-finite samples")
        if rows.step.max() < steps - config["numerics"]["vlm"]["logging_interval_steps"]:
            failures.append(f"{surface}: incomplete force history")
        cycles = list(last_cycles(rows, period))
        if len(cycles) < 3:
            failures.append(f"{surface}: fewer than three complete sampled cycles")
            continue
        phase = np.linspace(0.01, 1, 100)
        profiles = np.array([np.interp(phase, x, cycle.force_z) for _, x, cycle in cycles])
        scale = max(np.sqrt(np.mean(profiles**2)), 1e-12)
        drift = np.max(np.sqrt(np.mean(np.diff(profiles, axis=0) ** 2, axis=1))) / scale
        print(f"{surface}: period={period:.6g}s, phase-resolved force RMS drift={drift:.2%}")
        if not np.isfinite(drift) or drift > 0.05:
            failures.append(f"{surface}: final-cycle force drift exceeds5%; run longer")
    integrals = pd.read_csv(SAMPLES_DIR / "flow_integrals.csv")
    if not np.isfinite(integrals.select_dtypes("number")).all().all():
        failures.append("non-finite flow integrals")
    end = state["initial_time"] + config["run"]["steps"] * dt
    if integrals.time.max() < end - 0.05:
        failures.append("incomplete sampled flow history")
    if len(list(SAMPLES_DIR.glob("wake_*span.pvd"))) != 3:
        failures.append("missing published wake planes")
    if not args.pre_plot:
        for name in ("delta_wing_forces", "delta_wing_circulation_history", "delta_wing_wake"):
            for extension in ("png", "pdf"):
                if not (FIGURES_DIR / f"{name}.{extension}").is_file():
                    failures.append(f"missing {name}.{extension}")
    print("\n".join(f"[FAIL] {item}" for item in failures) or "[OK] Delta-wing checks passed")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
