#!/usr/bin/env python3
"""Diagnose the direction and size of a frozen conservative source correction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_path(path):
    return str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


def response(error, change, weight):
    def inner(a, b):
        return float(np.average(np.sum(a*b, axis=-1), weights=weight))
    ee, ed, dd = inner(error, error), inner(error, change), inner(change, change)
    samples = []
    for scale in (0., .25, .5, .75, 1.):
        direct = inner(error+scale*change, error+scale*change)
        predicted = ee+2*scale*ed+scale*scale*dd
        np.testing.assert_allclose(direct, predicted, rtol=0, atol=1e-16)
        samples.append({"scale": scale, "error_rms": float(np.sqrt(direct)), "squared_error_identity_difference": abs(direct-predicted)})
    upper = min(1., -2*ed/dd) if ed < 0 and dd > 0 else None
    return {"baseline_error_rms": float(np.sqrt(ee)), "correction_rms": float(np.sqrt(dd)),
            "baseline_error_dot_correction": ed, "initial_squared_error_slope": 2*ed,
            "positive_scale_improvement_upper_bound": upper, "scale_samples": samples}


def run(args):
    sources, experiments = [], []
    for directory in args.induction:
        report_path = directory / "cube-shared-trace-induction-3d.json"
        report = json.loads(report_path.read_text())
        assert report["status"] == "complete" and report["spatial_dimensions"] == 3
        field_path = directory / "shared-trace-physical-fields.npz"
        source_path = ROOT / next(row["path"] for row in report["sources"] if row["path"].endswith("/shared-trace-source-fields.npz"))
        data, source = arrays(field_path), arrays(source_path)
        for path in (report_path, field_path, source_path):
            sources.append({"path": record_path(path), "sha256": digest(path)})
        names = data["source_names"].tolist()
        area = data["area"]
        families = []
        for kind, base, updated in (("point", 2, 4), ("cell_average", 5, 7)):
            base_name, updated_name = names[base], names[updated]
            u0, u1 = data[base_name+"__velocity"], data[updated_name+"__velocity"]
            observations = {}
            for group in ("near_body", "held_outer", "wake"):
                ids, target = data[group+"__cell_ids"], data["target__"+group]
                observations[group] = (u0[target]-source["full_velocity"][ids], u1[target]-u0[target], source["full_volume"][ids])
            for mode in ("point", "native"):
                key = "__"+mode+"_normal_velocity"
                observations[mode+"_normal_velocity"] = ((data[base_name+key]-data["reference_normal_velocity"])[:, None],
                                                         (data[updated_name+key]-data[base_name+key])[:, None], area)
            observations["native_tangential_derivative"] = (data[base_name+"__tangential_gradient"]-data["reference_tangential_gradient"],
                                                           data[updated_name+"__tangential_gradient"]-data[base_name+"__tangential_gradient"], area)
            rows = {name: response(*values) for name, values in observations.items()}
            for label, index in (("baseline", base), ("updated", updated)):
                sample = 0 if label == "baseline" else -1
                for group in ("near_body", "held_outer", "wake"):
                    np.testing.assert_allclose(rows[group]["scale_samples"][sample]["error_rms"],
                                               report["records"][index]["cell_velocity_rms_over_Uinf"][group], rtol=0, atol=1e-14)
                for observation, key in (("native_normal_velocity", "native_normal_velocity_error_rms"),
                                         ("point_normal_velocity", "point_normal_velocity_error_rms"),
                                         ("native_tangential_derivative", "native_gradient_error_rms")):
                    np.testing.assert_allclose(rows[observation]["scale_samples"][sample]["error_rms"],
                                               report["records"][index][key], rtol=0, atol=1e-14)
            gates = ("native_normal_velocity", "native_tangential_derivative")
            upper = [rows[name]["positive_scale_improvement_upper_bound"] for name in gates]
            joint = None if any(value is None for value in upper) else min(upper)
            outer_gates = (*gates, "held_outer", "wake")
            outer_upper = [rows[name]["positive_scale_improvement_upper_bound"] for name in outer_gates]
            outer_joint = None if any(value is None for value in outer_upper) else min(outer_upper)
            families.append({"input": kind, "baseline": base_name, "updated": updated_name, "observations": rows,
                             "joint_native_boundary_improvement_upper_bound": joint,
                             "joint_boundary_and_outer_velocity_improvement_upper_bound": outer_joint})
        experiments.append({"directory": record_path(directory), "source_sgs": report["source_sgs"],
                            "small_fvm_cells": report["small_fvm_cells"], "families": families})
    own = Path(__file__).resolve()
    sources.append({"path": record_path(own), "sha256": digest(own)})
    archive = args.output.parent / (args.output.stem+"-sources") / own.relative_to(ROOT)
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(own.read_bytes())
    result = {"schema": "openonda-shared-trace-response-3d/1", "status": "complete", "spatial_dimensions": 3,
              "experiments": experiments, "sources": sources,
              "interpretation": "For a positive scale lambda, squared error changes by 2 lambda <e,du> + lambda^2 ||du||^2. A missing upper bound means no positive scale reduces that error.",
              "limitations": ["This is a diagnostic using reference errors, not a reference-free damping rule or a production parameter choice.",
                              "The linear response holds for frozen source induction, the fixed constrained body solve and the linear mass correction. It does not predict an advancing nonlinear trajectory.",
                              "All scales interpolate the same source-update direction; changing the reconstruction or interface treatment is a different experiment.",
                              "Bounds are strict where equality gives the baseline error. A bound of one is also capped at the endpoint of the tested source segment."]}
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    for experiment in experiments:
        print(json.dumps({"directory": experiment["directory"], "families": [
            {"input": family["input"], "joint_native_boundary_improvement_upper_bound": family["joint_native_boundary_improvement_upper_bound"],
             "joint_boundary_and_outer_velocity_improvement_upper_bound": family["joint_boundary_and_outer_velocity_improvement_upper_bound"],
             "observations": {name: {key: row[key] for key in ("baseline_error_rms", "correction_rms", "baseline_error_dot_correction", "positive_scale_improvement_upper_bound")}
                              for name, row in family["observations"].items()}}
            for family in experiment["families"]]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--induction", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.induction = [path.resolve() for path in args.induction]
    args.output = args.output.resolve()
    run(args)
