"""Summarize the three ten-step cylinder optimization pilots without running them.

Usage: python studies/summarize_cylinder_optimizations.py --root PATH --output PATH
       [--baseline-name baseline_repeat]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

DEFAULT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "tutorials/coupled_fvm_vpm"
    / "01_cylinder_shedding_flow/study_results/execution_pilots"
    / "optimization_comparison"
)
CASES = ("baseline", "optimized", "aitken")
PHASES = ("vpm", "vpm_boundary_condition", "fvm", "transfer", "total")
FIELDS = {
    "velocity": ("velocity_x", "velocity_y", "velocity_z"),
    "vorticity": ("vorticity_x", "vorticity_y", "vorticity_z"),
    "pressure": ("kinematic_pressure",),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path):
    require(path.is_file(), f"missing {path}")
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"missing {path}")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(bool(rows), f"empty {path}")
    return rows


def metric(values: list[float]) -> dict:
    require(bool(values), "empty metric")
    return {"mean": sum(values) / len(values), "maximum": max(values), "final": values[-1]}


def csv_key(row: dict[str, str]) -> tuple[float, float, float, float]:
    return tuple(
        round(float(row[name]), 10) for name in ("time", "position_x", "position_y", "position_z")
    )


def compare_samples(reference: Path, candidate: Path) -> dict:
    names = {path.name for path in reference.glob("*.csv") if path.name != "forces_history.csv"}
    require(bool(names), f"no sampled profiles in {reference}")
    require(
        names
        == {path.name for path in candidate.glob("*.csv") if path.name != "forces_history.csv"},
        f"profile file set differs: {candidate}",
    )
    result = {}
    for name in sorted(names):
        left, right = read_csv(reference / name), read_csv(candidate / name)
        require(len(left) == len(right), f"sample row count differs: {name}")
        a = {csv_key(row): row for row in left}
        b = {csv_key(row): row for row in right}
        require(
            len(a) == len(left) and len(b) == len(right) and a.keys() == b.keys(),
            f"sample times/positions differ or duplicate: {name}",
        )
        require(max(key[0] for key in a) >= 0.4 - 1e-9, f"sample horizon incomplete: {name}")
        errors = {}
        for group, columns in FIELDS.items():
            left_columns, right_columns = set(left[0]), set(right[0])
            left_present = all(column in left_columns for column in columns)
            right_present = all(column in right_columns for column in columns)
            require(left_present == right_present, f"sample field group differs: {name}:{group}")
            require(
                left_present
                or not any(column in left_columns | right_columns for column in columns),
                f"partial sample field group: {name}:{group}",
            )
            if not left_present:
                require(group != "velocity", f"velocity missing: {name}")
                continue
            differences, denominators = [], []
            for key in sorted(a):
                for column in columns:
                    x, y = float(a[key][column]), float(b[key][column])
                    require(math.isfinite(x) and math.isfinite(y), f"nonfinite {name}:{column}")
                    differences.append(y - x)
                    denominators.append(x)
            numerator = math.sqrt(sum(value * value for value in differences))
            denominator = math.sqrt(sum(value * value for value in denominators))
            errors[group] = {
                "max_abs": max(map(abs, differences)),
                "relative_l2": numerator / denominator
                if denominator
                else (0.0 if not numerator else None),
            }
        result[name] = {"aligned_rows": len(a), "fields": errors}
    return result


def summarize_case(root: Path, name: str) -> tuple[dict, dict, Path]:
    folder = root / name
    solution = folder / "solution"
    diagnostic_path = solution / "coupler_diagnostics.jsonl"
    require(diagnostic_path.is_file(), f"missing {diagnostic_path}")
    records = [
        json.loads(line) for line in diagnostic_path.read_text().splitlines() if line.strip()
    ]
    require(len(records) == 10, f"{name}: expected exactly 10 diagnostics, found {len(records)}")
    require(
        all(
            row["step"] == index and math.isclose(row["time"], index * 0.04, abs_tol=1e-8)
            for index, row in enumerate(records, 1)
        ),
        f"{name}: incomplete or mismatched step/time horizon",
    )
    manifest = read_json(folder / "campaign_manifest.json")
    config = manifest["config"]
    resolved = config["resolved"].copy()
    resolved["interface_acceleration"] = (config.get("overrides") or {}).get(
        "interface_acceleration", resolved.get("interface_acceleration", "none")
    )
    source_digest = manifest.get("software_fingerprint", {}).get("source_digest")
    require(source_digest, f"{name}: missing source digest")
    trial = read_json(root / "logs" / name / "trial.json")
    require(
        trial.get("returncode") == 0 and not trial.get("timed_out"),
        f"{name}: trial did not finish successfully",
    )
    require(manifest.get("status") == "complete", f"{name}: campaign is not complete")
    require(
        math.isclose(float(config["end_time"]), 0.4, abs_tol=1e-12),
        f"{name}: configured end time is not 0.4",
    )
    require(config.get("max_coupling_steps") in (None, 10), f"{name}: unexpected coupling-step cap")
    require(
        math.isclose(float(resolved["exchange_dt"]), 0.04, abs_tol=1e-12),
        f"{name}: exchange interval differs",
    )
    interval = [float(row["timing_seconds"]["total"]) for row in records]
    warm = records[3:]
    phases = {
        phase: {
            "all_seconds": sum(float(row["timing_seconds"][phase]) for row in records),
            "warm_4_to_10": metric([float(row["timing_seconds"][phase]) for row in warm]),
        }
        for phase in PHASES
    }
    accepted, factors, rejected = [], [], 0
    for row in records:
        iteration = row["interface_iteration"]
        sweep = int(iteration.get("accepted_sweep", iteration["sweeps"]))
        candidates = [item for item in iteration["residuals"] if int(item["sweep"]) == sweep]
        require(
            len(candidates) == 1 and not candidates[0].get("acceleration_rejected", False),
            f"{name}: invalid accepted sweep at step {row['step']}",
        )
        accepted.append(candidates[0])
        rejected += sum(bool(item.get("acceleration_rejected")) for item in iteration["residuals"])
        factors.extend(
            float(item["next_acceleration_alpha"])
            for item in iteration["residuals"]
            if item.get("next_acceleration_alpha") is not None
        )
    force_rows = read_csv(folder / "samples" / "forces_history.csv")
    final_forces = [
        row for row in force_rows if math.isclose(float(row["time"]), 0.4, abs_tol=1e-8)
    ]
    require(len(final_forces) == 1, f"{name}: missing or duplicate force sample at t=0.4")
    flux = [row["vpm_boundary_condition_flux"] for row in records]
    transfer = [row["transfer"] for row in records]
    images = [row.get("last_induction_image_call") or {} for row in records]
    report = {
        "raw_directory": str(folder.resolve()),
        "command": trial.get("command"),
        "wall_seconds": trial.get("wall_seconds"),
        "peak_process_tree_rss_bytes": trial.get("peak_process_tree_rss_bytes"),
        "source_digest": source_digest,
        "resolved": resolved,
        "interval_seconds": {"all": metric(interval), "warm_4_to_10": metric(interval[3:])},
        "phases": phases,
        "particle_count": metric([int(row["n_transfer_particles"]) for row in records]),
        "image_tail": {
            key: metric([float(item[key]) for item in images if item.get(key) is not None])
            for key in ("shell", "relative", "target_evaluations")
            if all(item.get(key) is not None for item in images)
        },
        "interface": {
            "accepted_normal_rms": metric(
                [float(item["normal_residual_rms"]) for item in accepted]
            ),
            "accepted_gradient_rms": metric(
                [float(item["gradient_residual_rms"]) for item in accepted]
            ),
            "converged_steps": sum(bool(item["converged"]) for item in accepted),
            "acceleration_rejected_trials": rejected,
            "acceleration_accepted_trials": sum(
                item.get("acceleration_alpha") is not None
                and not item.get("acceleration_rejected", False)
                for row in records
                for item in row["interface_iteration"]["residuals"]
            ),
            "proposed_factors": factors,
        },
        "conservation_flux": {
            "max_abs_raw_flux_mismatch": max(abs(float(item["raw_mismatch"])) for item in flux),
            "max_abs_corrected_flux_mismatch": max(
                abs(float(item["corrected_mismatch"])) for item in flux
            ),
            "max_raw_relative_flux_mismatch": max(float(item["raw_relative"]) for item in flux),
            "all_raw_flux_within_acceptance_limit": all(
                float(item["raw_relative"]) <= float(item["acceptance_limit"]) for item in flux
            ),
            "max_abs_renewal_conservation_error": max(
                abs(float(item["renewal_conservation_error"])) for item in transfer
            ),
            "all_renewal_within_tolerance": all(
                abs(float(item["renewal_conservation_error"]))
                <= float(item["renewal_vortex_strength_tolerance"])
                for item in transfer
            ),
        },
        "force_at_t_0_4": {
            key: float(final_forces[0][key])
            for key in ("drag_coefficient", "lift_coefficient", "side_force_coefficient")
        },
    }
    return report, resolved, folder / "samples"


def comparable_config(config: dict) -> dict:
    excluded = {"interface_acceleration", "source_hash", "software_fingerprint", "overrides"}
    return {key: value for key, value in config.items() if key not in excluded}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-name", default="baseline")
    args = parser.parse_args()
    require(
        args.baseline_name not in {"", ".", ".."}
        and Path(args.baseline_name).name == args.baseline_name
        and args.baseline_name not in CASES[1:],
        "baseline name must be one distinct directory name",
    )
    provenance = read_json(args.root / "provenance.json")
    reports, configs, samples = {}, {}, {}
    for name in CASES:
        source_name = args.baseline_name if name == "baseline" else name
        reports[name], configs[name], samples[name] = summarize_case(args.root, source_name)
    require(
        all(
            comparable_config(configs[name]) == comparable_config(configs["baseline"])
            for name in CASES[1:]
        ),
        "physical/numerical configuration differs across cases",
    )
    require(
        configs["baseline"].get("interface_acceleration", "none") == "none"
        and configs["optimized"].get("interface_acceleration", "none") == "none"
        and configs["aitken"].get("interface_acceleration") == "aitken",
        "unexpected interface acceleration modes",
    )
    output = {
        "schema": "openonda-cylinder-optimization-short-comparison/1",
        "qualification": "One run per case over ten short transient steps through t=0.4. Timing ratios carry single-run uncertainty from host load and startup effects; no confidence interval or repeatability claim is available. This is an execution and local comparison, not a 12-hour runtime, developed shedding, or grid-independence certificate.",
        "root": str(args.root.resolve()),
        "raw_case_directories": {
            name: str((args.root / (args.baseline_name if name == "baseline" else name)).resolve())
            for name in CASES
        },
        "provenance": provenance,
        "cases": reports,
        "comparisons": {
            name: {
                "sampled_fields_vs_baseline": compare_samples(samples["baseline"], samples[name]),
                "interpretation": "Aitken changes interface iterates; field differences do not establish numerical equivalence."
                if name == "aitken"
                else "Matched short-transient field difference; no equivalence threshold is implied.",
            }
            for name in CASES[1:]
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
