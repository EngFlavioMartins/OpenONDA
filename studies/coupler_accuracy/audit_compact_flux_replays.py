"""Compare completed conservative-source controls using native 3D reference fields."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path: Path, expected_steps: int) -> dict:
    """Require a complete replay, preserving the source and input identities."""
    value = json.loads(path.read_text())
    if value["status"] != "completed" or value["steps_completed"] != expected_steps:
        raise ValueError(f"Incomplete replay: {path}")
    return value


def endpoint_comparison(native: dict, corrected: dict) -> dict:
    """Compare identical saved start states, targets, and physical endpoint times."""
    for key in ("checkpoint_sha256", "target_cache_sha256"):
        if native[key] != corrected[key]:
            raise ValueError(f"Mismatched comparison input: {key}")
    if native["observations"][-1]["time"] != corrected["observations"][-1]["time"]:
        raise ValueError("Endpoint physical times differ")
    a, b = native["observations"][-1], corrected["observations"][-1]
    return {
        "time": a["time"],
        "native_particles": a["particles"],
        "corrected_particles": b["particles"],
        "particle_count_ratio": b["particles"] / a["particles"],
        "regions": {
            name: {
                "native_error_rms": values["velocity_error_rms"],
                "corrected_error_rms": b["regions"][name]["velocity_error_rms"],
                "error_reduction": values["velocity_error_rms"]
                - b["regions"][name]["velocity_error_rms"],
                "relative_error_reduction": 1
                - b["regions"][name]["velocity_error_rms"] / values["velocity_error_rms"],
            }
            for name, values in a["regions"].items()
        },
    }


def source_balance(replay: dict) -> dict:
    """Separate the conservative source's moments from transport and renewal."""
    calls = replay["compact_flux_source"]["substeps"]
    stages = [stage for call in calls for stage in call["stages"]]
    impulse_closure = [
        np.asarray(call["after"]["linear_impulse_per_density"])
        - call["before"]["linear_impulse_per_density"]
        - call["integrated_source_impulse_per_density"]
        for call in calls
    ]
    phase_changes = {}
    for start, end, name in (
        ("before_native_evolution", "before_transfer", "native_transport_and_gbd"),
        ("before_transfer", "after_transfer", "renewal"),
    ):
        phase_changes[name] = {
            key: np.sum(
                [np.asarray(step[end][key]) - step[start][key] for step in replay["step_budgets"]],
                axis=0,
            ).tolist()
            for key in ("net_strength", "linear_impulse_per_density")
        }
    return {
        "evaluations": len(stages),
        "source_preserved_count": all(
            call["before"]["particles"] == call["after"]["particles"] for call in calls
        ),
        "max_source_net_rate": max(
            np.linalg.norm(stage["source_net_strength_rate"]) for stage in stages
        ),
        "max_storage_net_closure": max(
            np.linalg.norm(call["storage_net_strength_closure"]) for call in calls
        ),
        "max_storage_impulse_closure": max(np.linalg.norm(value) for value in impulse_closure),
        "total_applied_source_net_change": np.sum(
            [
                np.asarray(call["after"]["net_strength"]) - call["before"]["net_strength"]
                for call in calls
            ],
            axis=0,
        ).tolist(),
        "total_source_impulse_increment": np.sum(
            [call["integrated_source_impulse_per_density"] for call in calls], axis=0
        ).tolist(),
        "other_phase_changes": phase_changes,
    }


def timing(replay: dict) -> dict:
    """Report measured evolution/transfer/source work, including device synchronization."""
    steps = replay["step_budgets"]
    result = {
        "steps": len(steps),
        "mean_seconds_per_step": float(np.mean([step["wall_seconds"] for step in steps])),
        "mean_seconds_per_step_after_first_two": float(
            np.mean([step["wall_seconds"] for step in steps[2:]])
        ),
        "native_evolution_mean_seconds": float(
            np.mean([step["native_evolution_wall_seconds"] for step in steps])
        ),
        "renewal_calls": (
            sum(step["transfer_due"] for step in steps)
            if all("transfer_due" in step for step in steps)
            else None
        ),
    }
    if "compact_flux_source" in replay:
        result["source_mean_seconds_per_step"] = replay["compact_flux_source"][
            "wall_seconds"
        ] / len(steps)
    return result


def run(directory: Path, native_full: Path) -> None:
    """Audit the two dt controls, with a historical native full-step control."""
    paths = {
        "native_dt001": native_full,
        "native_dt001_pilot": directory / "native-dt001-pilot/replay.json",
        "flux_dt001_pilot": directory / "flux-dt001-pilot/replay.json",
        "flux_dt001": directory / "flux-dt001/replay.json",
        "native_dt0005": directory / "native-dt0005/replay.json",
        "flux_dt0005": directory / "flux-dt0005/replay.json",
    }
    expected_steps = {
        "native_dt001": 100,
        "native_dt001_pilot": 10,
        "flux_dt001_pilot": 10,
        "flux_dt001": 100,
        "native_dt0005": 200,
        "flux_dt0005": 200,
    }
    replays = {name: read(path, expected_steps[name]) for name, path in paths.items()}
    for key in ("time_step_size", "renewal_interval"):
        if replays["native_dt0005"][key] != replays["flux_dt0005"][key]:
            raise ValueError(f"Half-step comparison changed {key}")
    pairs = {
        "dt001": endpoint_comparison(replays["native_dt001"], replays["flux_dt001"]),
        "dt0005": endpoint_comparison(replays["native_dt0005"], replays["flux_dt0005"]),
        "pilot": endpoint_comparison(replays["native_dt001_pilot"], replays["flux_dt001_pilot"]),
    }
    temporal = {}
    for name, values in pairs["dt001"]["regions"].items():
        fine = pairs["dt0005"]["regions"][name]
        change = abs(fine["error_reduction"] - values["error_reduction"])
        temporal[name] = {
            "benefit_at_both_steps": min(values["error_reduction"], fine["error_reduction"]) > 0,
            "change_in_benefit": change,
            "change_smaller_than_benefit": change
            < min(values["error_reduction"], fine["error_reduction"]),
            "native_error_change": fine["native_error_rms"] - values["native_error_rms"],
            "corrected_error_change": fine["corrected_error_rms"] - values["corrected_error_rms"],
        }
    report = {
        "scope": __doc__,
        "pairs": pairs,
        "step_sensitivity": temporal,
        "source_balances": {
            key: source_balance(replays[key]) for key in ("flux_dt001", "flux_dt0005")
        },
        "timings": {
            key: timing(value) for key, value in replays.items() if "step_budgets" in value
        },
        "timing_limit": "The .01 pilots are sequential laptop runs with uncontrolled external load. The .005 pair runs concurrently for accuracy; exclude its timing from performance ratios. Historical native full-step timing is not a matched cost benchmark. No full-coupler timing claim.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()
        },
    }
    (directory / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--native-full", type=Path, required=True)
    args = parser.parse_args()
    run(args.directory, args.native_full)
