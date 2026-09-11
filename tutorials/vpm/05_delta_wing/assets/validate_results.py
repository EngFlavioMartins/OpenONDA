#!/usr/bin/env python3
"""Check native Delta samples, including a sparse-to-dense continuation."""

if not __package__:
    from pathlib import Path as _CasePath
    from openonda.tutorial_runner import case_package

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ._delta_wing_plots import (
    CASE_DIR,
    FIGURES_DIR,
    SAMPLES_DIR,
    flow_integrals,
    force_history,
    last_cycles,
    motion_period,
)


def _directories(values, default: Path) -> list[Path]:
    paths = [Path(value) for value in values] if values else [default]
    return [path if path.is_absolute() else CASE_DIR / path for path in paths]


def _configured_flow_interval(metadata: dict) -> int | None:
    items = metadata["configuration"].get("samplers", {}).get("items", [])
    for item in items:
        if item.get("type") == "FlowIntegralsSampler":
            return item.get("schedule", {}).get("interval")
    return None


def _segment_metadata(solution_dir: Path) -> dict:
    path = solution_dir / "vpm_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _observed_cadence(steps: np.ndarray) -> int | None:
    unique = np.unique(steps.astype(int))
    if len(unique) < 2:
        return None
    differences = np.diff(unique)
    return int(np.gcd.reduce(differences))


def _check_segment(solution_dir: Path, samples_dir: Path, *, require_complete: bool) -> list[str]:
    """Check one segment while treating source lifecycle markers as archival."""
    failures = []
    metadata = _segment_metadata(solution_dir)
    config, state = metadata["configuration"], metadata["state"]
    expected_step = state["initial_step"] + config["run"]["steps"]
    label = solution_dir.name or str(solution_dir)
    status = metadata.get("lifecycle", {}).get("status")
    if status != "completed":
        print(
            f"[INFO] {label}: lifecycle marker {status!r} is archival metadata; "
            "this validator does not infer live-process state from it"
        )
    if require_complete:
        if status != "completed" or state["step"] != expected_step:
            failures.append(
                f"{label}: final segment incomplete ({status=}; step={state['step']}/{expected_step})"
            )

    force_path = samples_dir / "vlm_surface_forces.csv"
    force = pd.read_csv(force_path)
    duplicate_times = any(rows.time.duplicated().any() for _, rows in force.groupby("surface"))
    if duplicate_times or not np.isfinite(force.select_dtypes("number")).all().all():
        failures.append(f"{label}: duplicate or non-finite force samples")
    logging_interval = config["numerics"]["vlm"]["logging_interval_steps"]
    observed_force_cadence = _observed_cadence(force.step.to_numpy())
    print(
        f"[INFO] {label}: force cadence observed every {observed_force_cadence} steps; "
        f"configured {logging_interval}"
    )
    if require_complete and force.step.max() < expected_step - logging_interval:
        failures.append(f"{label}: incomplete force history")
    if require_complete and logging_interval == 1 and observed_force_cadence != 1:
        failures.append(f"{label}: dense continuation force cadence is not every accepted step")

    flow_path = samples_dir / "flow_integrals.csv"
    flow = pd.read_csv(flow_path)
    if not np.isfinite(flow.select_dtypes("number")).all().all():
        failures.append(f"{label}: non-finite flow integrals")
    flow_interval = _configured_flow_interval(metadata)
    observed_flow_cadence = _observed_cadence(flow.step.to_numpy())
    print(
        f"[INFO] {label}: flow cadence observed every {observed_flow_cadence} steps; "
        f"configured {flow_interval}"
    )
    if require_complete and flow_interval and observed_flow_cadence != flow_interval:
        failures.append(f"{label}: flow-integral cadence differs from its configured schedule")

    wake_planes = list(samples_dir.glob("wake_*span.pvd"))
    if require_complete and len(wake_planes) != 3:
        failures.append(f"{label}: expected three published wake planes")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-plot", action="store_true")
    parser.add_argument(
        "--solution",
        type=Path,
        action="append",
        help="solution directory; repeat in sparse-to-dense order",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        action="append",
        help="sample directory matching each --solution directory",
    )
    args = parser.parse_args()
    solution_dirs = _directories(args.solution, CASE_DIR / "solution")
    samples_dirs = _directories(args.samples, SAMPLES_DIR)
    if len(solution_dirs) != len(samples_dirs):
        parser.error("repeat --solution and --samples the same number of times")

    failures = []
    for index, (solution_dir, samples_dir) in enumerate(
        zip(solution_dirs, samples_dirs, strict=True)
    ):
        failures.extend(
            _check_segment(
                solution_dir,
                samples_dir,
                require_complete=index == len(solution_dirs) - 1,
            )
        )

    final_metadata = _segment_metadata(solution_dirs[-1])
    final_config, final_state = final_metadata["configuration"], final_metadata["state"]
    data = force_history(samples_dirs)
    period = motion_period(data)
    for surface, rows in data.groupby("surface"):
        if (
            rows.time.duplicated().any()
            or not np.isfinite(rows.select_dtypes("number")).all().all()
        ):
            failures.append(f"{surface}: duplicate or non-finite merged samples")
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

    integrals = flow_integrals(samples_dirs)
    if not np.isfinite(integrals.select_dtypes("number")).all().all():
        failures.append("non-finite merged flow integrals")
    dt = final_config["numerics"]["time_step_size"]
    end = final_state["initial_time"] + final_config["run"]["steps"] * dt
    if final_state["step"] == final_state["initial_step"] + final_config["run"]["steps"]:
        if integrals.time.max() < end - 0.05:
            failures.append("incomplete merged sampled flow history")
    if not args.pre_plot:
        for name in ("delta_wing_forces", "delta_wing_circulation_history", "delta_wing_wake"):
            for extension in ("png", "pdf"):
                if not (FIGURES_DIR / f"{name}.{extension}").is_file():
                    failures.append(f"missing {name}.{extension}")
    print("\n".join(f"[FAIL] {item}" for item in failures) or "[OK] Delta-wing checks passed")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
