#!/usr/bin/env python3
"""Compare verified hybrid exchange intervals at identical FVM time indices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays

VELOCITY_METRICS = (
    "fvm_velocity_rms_over_Uinf", "fvm_near_body_velocity_rms_over_Uinf",
    "vpm_sampled_velocity_rms_over_Uinf",
)
FIXED_INPUTS = (
    "source_seed_time", "fvm_dt", "particle_spacing", "core_radius_ratio", "boundary_mode",
    "transfer_method", "eta_blend_width", "transfer_region_bounds", "initial_vorticity_cutoff",
    "transfer_vorticity_cutoff", "transfer_amplification_cap", "frozen_renewals",
    "experimental_residual_blend", "audit_pressure", "pressure_history", "mixed_boundary_convection",
    "full_cells", "small_cells", "requested_wall_spacing", "small_actual_bounds",
    "identical_native_shared_cells", "fvm_numerics", "vpm_numerics",
)


def load_pair(path, sources):
    verification = json.loads(path.read_text())
    assert verification["status"] == "complete" and verification["spatial_dimensions"] == 3
    assert verification["comparison_intervals"] > 1
    assert verification["one_sweep_control_bitwise_checkpoint_counts"] == {
        "fvm": 17, "vpm_boundary_condition": 11, "vpm_numeric_datasets": 11,
    }
    for row in verification["sources"]:
        assert hash_file(ROOT / row["path"]) == row
    sources.extend([hash_file(path), *verification["sources"]])
    control = next(row for row in verification["experiments"] if row["directory"] == verification["comparison_control_directory"])
    candidates = [row for row in verification["experiments"] if row["maximum_sweeps"] > 1]
    assert len(candidates) == 1 and control["maximum_sweeps"] == 0
    result = []
    for row in (control, candidates[0]):
        directory = ROOT / row["directory"]
        report = json.loads((directory / "interface-iteration-3d.json").read_text())
        child = json.loads((directory / "trial/cube-coupled-trial.json").read_text())
        assert row["auxiliary_panel_precision_mode"] == "f64_auxiliary_panel_queries"
        assert report["status"] == child["status"] == "complete"
        assert report["comparison"] == child["comparison"]
        fields = read_arrays(directory / "trial/latest-comparison-fields.npz")
        substeps = round(child["vpm_dt"] / child["fvm_dt"])
        assert substeps >= 1 and substeps * child["fvm_dt"] == child["vpm_dt"]
        indexed = {item["coupling_step"] * substeps: item for item in child["comparison"]}
        assert len(indexed) == len(child["comparison"])
        result.append({"verified": row, "report": report, "child": child, "fields": fields,
                       "indexed": indexed, "substeps": substeps})
    return result


def common_metrics(case, ticks):
    rows = [case["indexed"][tick] for tick in ticks]
    error = np.array([row["drag_coefficient_difference"] / row["full_drag_coefficient"] for row in rows])
    return {
        "drag_relative_difference_rms_percent": float(100 * np.sqrt(np.mean(error**2))),
        "drag_relative_difference_maximum_percent": float(100 * np.max(np.abs(error))),
        **{key: float(np.sqrt(np.mean([row[key]**2 for row in rows]))) for key in VELOCITY_METRICS},
    }


def run(args):
    if args.output.exists() or args.output.with_suffix(".png").exists():
        raise FileExistsError(args.output)
    sources = []
    coarse, fine = (load_pair(path, sources) for path in (args.coarse, args.fine))
    all_cases = [*coarse, *fine]
    base = coarse[0]
    for case in all_cases:
        for key in FIXED_INPUTS:
            assert case["child"][key] == base["child"][key], key
        for record in ("child", "report"):
            assert case[record]["sources"] == base[record]["sources"]
        assert case["report"]["execution_environment"] == base["report"]["execution_environment"]
        assert case["report"]["comparison"][0] == base["report"]["comparison"][0]
        for key in ("full_velocity", "position", "vpm_query_ids"):
            np.testing.assert_array_equal(case["fields"][key], base["fields"][key])
        assert max(case["indexed"]) == max(base["indexed"])
    assert coarse[0]["substeps"] == coarse[1]["substeps"] > fine[0]["substeps"] == fine[1]["substeps"]
    for key in ("maximum_sweeps", "relaxation", "normal_tolerance", "gradient_tolerance"):
        assert coarse[1]["report"][key] == fine[1]["report"][key]
    ticks = sorted(set.intersection(*(set(case["indexed"]) for case in all_cases)) - {0})
    assert ticks and ticks[-1] == max(base["indexed"])
    for tick in ticks:
        expected = base["indexed"][tick]
        for case in all_cases:
            actual = case["indexed"][tick]
            assert actual["full_drag_coefficient"] == expected["full_drag_coefficient"]
            np.testing.assert_allclose(actual["physical_time"], expected["physical_time"], rtol=0, atol=2e-14)
            np.testing.assert_allclose(actual["elapsed_flow_time"], tick * case["child"]["fvm_dt"], rtol=0, atol=2e-14)
    experiments = []
    for case in all_cases:
        report, child = case["report"], case["child"]
        last = {row["coupling_step"]: row for row in report["sweeps"]}
        final = child["comparison"][-1]
        experiments.append({
            "directory": case["verified"]["directory"], "exchange_dt": child["vpm_dt"],
            "fvm_substeps": case["substeps"], "maximum_sweeps": report["maximum_sweeps"],
            "common_endpoint_metrics": common_metrics(case, ticks), "final_comparison": final,
            "final_drag_relative_difference_percent": 100 * final["drag_coefficient_difference"] / final["full_drag_coefficient"],
            "converged_intervals": sum(row["converged"] for row in last.values()),
            "observed_intervals": len(last), "logical_sweeps": sum(row["sweep"] for row in last.values()),
            "capped_intervals": [step for step, row in last.items() if not row["converged"]],
        })
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    reference_rows = fine[0]["child"]["comparison"]
    axes[0, 0].plot([row["physical_time"] for row in reference_rows], [row["full_drag_coefficient"] for row in reference_rows], color="0.1", label="Full FVM")
    for index, case in enumerate(all_cases):
        child = case["child"]
        rows = child["comparison"]
        time = np.array([row["physical_time"] for row in rows])
        label = f"{'Iterated' if index % 2 else 'Original'}; exchange {child['vpm_dt']:g}"
        color = ("#718096", "#ad572f", "#739679", "#156d91")[index]
        style = "-" if index % 2 else "--"
        options = {"color": color, "linestyle": style, "label": label, "linewidth": 1.6}
        axes[0, 0].plot(time, [row["hybrid_drag_coefficient"] for row in rows], **options)
        axes[0, 1].plot(time, [100 * row["drag_coefficient_difference"] / row["full_drag_coefficient"] for row in rows], **options)
        axes[1, 0].plot(time, [row[VELOCITY_METRICS[0]] for row in rows], **options)
        axes[1, 1].plot(time, [row[VELOCITY_METRICS[1]] for row in rows], **options)
    axes[0, 0].set(title="Force at unchanged FVM time step", ylabel="Drag coefficient")
    axes[0, 1].set(title="Force agreement at the same physical time", ylabel="Relative drag difference (%)")
    axes[0, 1].axhline(0, color="0.5", linewidth=.8)
    axes[1, 0].set(title="Whole small FVM domain", ylabel="Velocity RMS / U∞")
    axes[1, 1].set(title="Near body", ylabel="Velocity RMS / U∞")
    for ax in axes.flat:
        ax.set_xlabel("Physical time")
        ax.grid(alpha=.18)
        ax.legend(fontsize=8)
    figure.suptitle("Fully 3D medium cube: exchange and VPM time-resolution comparison", fontsize=13)
    figure.savefig(args.output.with_suffix(".png"), dpi=170)
    plt.close(figure)
    own = Path(__file__).resolve()
    archive = args.output.parent / (args.output.stem + "-sources") / own.relative_to(ROOT)
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(own.read_bytes())
    sources.append(hash_file(own))
    result = {
        "schema": "openonda-interface-cadence-comparison-3d/1", "status": "complete", "spatial_dimensions": 3,
        "common_fvm_step_indices": ticks, "initial_state_excluded_from_statistics": True,
        "reference_force_at_common_times_bitwise_equal": True, "final_reference_velocity_bitwise_equal": True,
        "fixed_configuration_fields": list(FIXED_INPUTS), "experiments": experiments, "sources": sources,
        "figure": hash_file(args.output.with_suffix(".png")),
        "limitations": [
            "Common-endpoint statistics use identical FVM time indices, with no phase alignment or reference feedback.",
            "VPM time integration, exchange interval and renewal frequency change together; this does not separate their effects.",
            "The FVM dt and matched spatial mesh are fixed. This is not joint spatial/time convergence or developed-wake validation.",
            "Sampled VPM velocity is evaluated at matched FVM cell centres, not an exterior-wake profile.",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"common_endpoints": len(ticks), "experiments": experiments}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse", type=Path, required=True, help="Verified larger exchange interval")
    parser.add_argument("--fine", type=Path, required=True, help="Verified smaller exchange interval")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.coarse, args.fine, args.output = (path.resolve() for path in (args.coarse, args.fine, args.output))
    run(args)
