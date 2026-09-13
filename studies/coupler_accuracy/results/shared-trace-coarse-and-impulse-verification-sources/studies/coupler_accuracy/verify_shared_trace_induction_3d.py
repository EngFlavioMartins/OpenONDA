#!/usr/bin/env python3
"""Independently check shared-source budgets and native physical comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from studies.coupler_accuracy.verify_sampled_native_face_3d import independent_gradient

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "studies/coupler_accuracy/results"


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label(path):
    return str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


def rms(value, weight=None):
    return float(np.sqrt(np.average(np.sum(np.asarray(value)**2, axis=-1), weights=weight)))


def tangent(value, normal):
    return value-np.sum(value*normal, axis=1)[:, None]*normal


def check_sources(directory, report):
    assert report["status"] == "complete" and report["spatial_dimensions"] == 3
    assert report["sources"] == json.loads((directory / "sources-at-start.json").read_text())
    for row in report["sources"]:
        path = ROOT / row["path"]
        archive = directory / "sources" / row["path"]
        assert digest(path) == row["sha256"], path
        if archive.exists():
            assert digest(archive) == row["sha256"], archive
    return len(report["sources"])


def budget_check(directory, check):
    report = json.loads((directory / "cube-shared-trace-sources-3d.json").read_text())
    source_count = check_sources(directory, report)
    data = arrays(directory / "shared-trace-source-fields.npz")
    names, c = data["source_names"].tolist(), data["centroid"]
    weight, gamma = data["volume_weight"], data["native_circulation"]
    rows = []
    for kind, base, cell, shared in (("point", 2, 3, 4), ("cell_average", 5, 6, 7)):
        np.testing.assert_array_equal(data["first_moment"][base], weight[:, None, None]*data[kind+"__base_moment"])
        np.testing.assert_array_equal(data["delta_circulation"][cell], weight[:, None]*(data[kind+"__new_circulation"]-gamma))
        np.testing.assert_array_equal(data["first_moment"][cell], weight[:, None, None]*data[kind+"__new_moment"])
        delta_integral = weight[:, None]*(data[kind+"__new_cell_integral"]-data[kind+"__base_cell_integral"])
        np.testing.assert_array_equal(delta_integral, data[kind+"__weighted_cell_integral_change"])
        if kind == "cell_average":
            np.testing.assert_array_equal(delta_integral, 0)
            np.testing.assert_array_equal(data[kind+"__base_cell_integral"], data["volume"][:, None]*data["cell_velocity"])
        integral = np.sum(delta_integral, axis=0)
        x, y, z = integral
        expected_h = np.array([[0., z, -y], [-z, 0., x], [y, -x, 0.]])
        for update, state in (("cell_weighted", cell), ("shared_trace", shared)):
            dg = data["delta_circulation"][state]
            dm = data["first_moment"][state]-data["first_moment"][base]
            h = np.sum(c[:, :, None]*dg[:, None, :]+dm, axis=0)
            local_impulse = .5*np.column_stack((dm[:, 1, 2]-dm[:, 2, 1], dm[:, 2, 0]-dm[:, 0, 2], dm[:, 0, 1]-dm[:, 1, 0]))
            impulse = np.sum(.5*np.cross(c, dg)+local_impulse, axis=0)
            claimed = next(row for row in report["update_budgets"] if row["input"] == kind and row["update"] == update)
            check(np.sum(dg, axis=0), claimed["circulation_change"])
            check(np.linalg.norm(dg, axis=1).sum(), claimed["circulation_change_L1"])
            check(h, claimed["first_spatial_moment_change"])
            check(impulse, claimed["impulse_change"])
            check(integral, claimed["weighted_cell_integral_change"])
            check(np.max(np.abs(h-expected_h)), claimed["first_moment_budget_maximum_difference"])
            if update == "shared_trace":
                np.testing.assert_allclose(dg.sum(axis=0), 0, rtol=0, atol=1e-12)
                np.testing.assert_allclose(h, expected_h, rtol=0, atol=1e-12)
                np.testing.assert_allclose(impulse, integral, rtol=0, atol=1e-12)
                np.testing.assert_array_equal(dg[weight == 0], 0)
                np.testing.assert_array_equal(dm[weight == 0], 0)
            rows.append({"input": kind, "update": update, "circulation_change_norm": float(np.linalg.norm(dg.sum(axis=0))),
                         "impulse_change": impulse.tolist(), "first_moment_budget_error": float(np.max(np.abs(h-expected_h)))})
    gaussian_h = np.sum(data["gaussian_position"][:, :, None]*data["gaussian_circulation"][:, None, :], axis=0)
    for state in range(len(names)):
        volume_gamma = weight[:, None]*gamma+data["delta_circulation"][state]
        total_gamma = volume_gamma.sum(axis=0)+data["gaussian_circulation"].sum(axis=0)
        total_h = np.sum(data["first_moment"][state]+c[:, :, None]*volume_gamma[:, None, :], axis=0)+gaussian_h
        check(total_gamma, data["total_source_circulation"][state])
        check(total_h, data["total_source_first_moment"][state])
    return data, report, {"directory": label(directory), "source_records": source_count, "updates": rows}


def run(args):
    metric_differences, array_differences, source_records, records, budget_records, artifacts = [], [], 0, [], [], []

    def check(actual, claimed, tolerance=2e-12):
        difference = float(np.max(np.abs(np.asarray(actual)-np.asarray(claimed))))
        metric_differences.append(difference)
        np.testing.assert_allclose(actual, claimed, rtol=0, atol=tolerance)

    def array_check(actual, expected):
        difference = float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
        array_differences.append(difference)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-12)

    for directory in args.induction:
        report = json.loads((directory / "cube-shared-trace-induction-3d.json").read_text())
        source_records += check_sources(directory, report)
        source_path = ROOT / next(row["path"] for row in report["sources"] if row["path"].endswith("/cube-shared-trace-sources-3d.json"))
        source, source_report, budgets = budget_check(source_path.parent, check)
        source_records += budgets["source_records"]
        budget_records.append(budgets)
        mesh_path = ROOT / next(row["path"] for row in report["sources"] if row["path"].endswith("/full-native-mesh.npz"))
        mesh = load_native_mesh(mesh_path)
        geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
        field = arrays(directory / "shared-trace-physical-fields.npz")
        induction = arrays(directory / "shared-trace-induction-checkpoint.npz")
        np.testing.assert_array_equal(induction["position"], field["position"])
        np.testing.assert_array_equal(field["source_names"], source["source_names"])
        assert str(induction["source_archive_sha256"]) == digest(source_path.parent / "shared-trace-source-fields.npz")
        normal, area = field["normal"], field["area"]
        np.testing.assert_allclose(np.linalg.norm(normal, axis=1), 1, rtol=0, atol=5e-16)
        np.testing.assert_array_equal(field["position"][field["target__sample"]], source["full_centres"][field["sample_cells"]])
        assert len(field["full_face_ids"]) == report["boundary_faces"]
        assert not np.any(np.isin(mesh["owners"][mesh["n_interior_faces"]:], field["gradient_cells"]))
        reference_gradient, reference_dn, reference_u = independent_gradient(mesh, geometry, field, source["full_velocity"][field["sample_cells"]])
        array_check(reference_gradient, field["reference_cell_gradient"])
        reference_gt, reference_un = tangent(reference_dn, normal), np.sum(reference_u*normal, axis=1)
        array_check(reference_gt, field["reference_tangential_gradient"])
        array_check(reference_un, field["reference_normal_velocity"])
        array_check(reference_u, field["reference_face_velocity"])
        sample_index = np.full(mesh["n_cells"], -1)
        sample_index[field["sample_cells"]] = np.arange(len(field["sample_cells"]))
        for state, row in enumerate(report["records"]):
            name = row["name"]
            actual, body = field[name+"__velocity"], field[name+"__body_velocity"]
            array_check(actual, induction["baseline_induction"][0]+induction["correction_induction"][state]+[1., 0., 0.]+body)
            sampled = actual[field["target__sample"]]
            gradient, dn, face_velocity = independent_gradient(mesh, geometry, field, sampled)
            gt = tangent(dn, normal)
            array_check(gradient, field[name+"__cell_gradient"])
            array_check(dn, field[name+"__normal_gradient"])
            array_check(gt, field[name+"__tangential_gradient"])
            array_check(face_velocity, field[name+"__face_velocity"])
            normals = {}
            for mode, u in (("point", actual[field["target__boundary"]]), ("native", face_velocity)):
                raw = np.sum(u*normal, axis=1)
                flux = float(raw @ area)
                corrected_velocity = u-(flux/area.sum())*normal
                normals[mode] = np.sum(corrected_velocity*normal, axis=1)
                array_check(normals[mode], field[name+"__"+mode+"_normal_velocity"])
                check(rms((normals[mode]-reference_un)[:, None], area), row[mode+"_normal_velocity_error_rms"])
                check(abs(flux), row["boundary_flux"][mode]["raw_mismatch"])
                check(abs(flux/area.sum()), row["boundary_flux"][mode]["applied_correction"])
                assert abs(normals[mode] @ area) < 1e-11
            check(rms(gt-reference_gt, area), row["native_gradient_error_rms"])
            check(rms(sampled[sample_index[field["gradient_cells"]]]-source["full_velocity"][field["gradient_cells"]],
                      source["full_volume"][field["gradient_cells"]]), row["gradient_input_cell_velocity_error_rms"])
            for group, claimed in row["cell_velocity_rms_over_Uinf"].items():
                ids = field[group+"__cell_ids"]
                check(rms(actual[field["target__"+group]]-source["full_velocity"][ids], source["full_volume"][ids]), claimed)
            wall = actual[field["target__wall"]]
            wall_un = np.sum(wall*field["wall_normal"], axis=1)
            check(rms(wall_un[:, None]), row["wall_normal_velocity_rms_over_Uinf"])
            check(rms(wall-wall_un[:, None]*field["wall_normal"]), row["wall_tangential_velocity_rms_over_Uinf"])
            check(field["panel_area"] @ field[name+"__panel_strength"], row["body_source_flux"])
            assert abs(row["body_source_flux"]) < 1e-10
            check(rms(field[name+"__collocation_residual"][:, None], field["panel_area"]), row["discrete_neumann_collocation_residual_rms"])
            for side, side_name in enumerate(("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")):
                mask = field["face_group"] == side
                side_row = row["face_groups"][side_name]
                assert int(mask.sum()) == side_row["faces"]
                check(rms((gt-reference_gt)[mask], area[mask]), side_row["native_gradient_error_rms"])
                for mode in ("point", "native"):
                    check(rms((normals[mode]-reference_un)[mask, None], area[mask]), side_row[mode+"_normal_velocity_error_rms"])
        records.append({"directory": label(directory), "source_sgs": source_report["source_sgs"],
                        "small_fvm_cells": report["small_fvm_cells"], "records": report["records"]})
        for path in (directory / "cube-shared-trace-induction-3d.json", directory / "shared-trace-physical-fields.npz",
                     directory / "shared-trace-induction-checkpoint.npz", source_path, source_path.parent / "shared-trace-source-fields.npz"):
            artifacts.append({"path": label(path), "sha256": digest(path)})
    impulse_endpoint_check = None
    if args.moment_impulse:
        path = args.moment_impulse / "native-moment-impulse-3d.json"
        endpoint = json.loads(path.read_text())
        assert endpoint["status"] == "complete" and endpoint["spatial_dimensions"] == 3
        for row in endpoint["sources"]:
            current = ROOT / row["path"]
            assert digest(current) == row["sha256"]
            archived = args.moment_impulse / "sources" / row["path"]
            if archived.exists():
                assert digest(archived) == row["sha256"]
        source_records += len(endpoint["sources"])
        data_path = args.moment_impulse / "native-moment-impulse-fields.npz"
        data = arrays(data_path)
        impulses = []
        for row in endpoint["states"]:
            name = row["state"]
            moment = np.sum(data[name+"__weighted_moment"], axis=0)
            value = .5*np.array([moment[1, 2]-moment[2, 1], moment[2, 0]-moment[0, 2], moment[0, 1]-moment[1, 0]])
            check(value, row["moment_representation_impulse_change"])
            check(np.max(np.abs(moment-data[name+"__independent_global_moment"])), row["independent_global_moment_maximum_difference"])
            array_check(moment, data[name+"__independent_global_moment"])
            impulses.append(value)
        check((impulses[1]-impulses[0])/endpoint["interval"], endpoint["mean_rate_of_representation_impulse_difference"])
        impulse_endpoint_check = {key: value for key, value in endpoint.items() if key != "sources"}
        artifacts += [{"path": label(p), "sha256": digest(p)} for p in (path, data_path)]
    test_file = RESULTS / "3d-shared-trace-update-regression.xml"
    cases = ET.parse(test_file).findall(".//testcase")
    assert len(cases) == 4 and all(case.find("failure") is None and case.find("error") is None and case.find("skipped") is None for case in cases)
    artifacts.append({"path": str(test_file.relative_to(ROOT)), "sha256": digest(test_file)})
    figure = args.output.with_suffix(".png")
    fig, axes = plt.subplots(len(records), 3, figsize=(13.2, 3.5*len(records)+.8), layout="constrained", squeeze=False)
    labels = ["P0", "Linear M", "Point base", "Point cell w", "Point shared", "Mean base", "Mean cell w", "Mean shared"]
    colors = ["#64748b", "#94a3b8", "#93c5fd", "#3b82f6", "#1d4ed8", "#a7f3d0", "#10b981", "#047857"]
    metrics = (("Near-body velocity RMS / U∞", lambda row: row["cell_velocity_rms_over_Uinf"]["near_body"]),
               ("Native normal-velocity error / U∞", lambda row: row["native_normal_velocity_error_rms"]),
               ("Native derivative error [U∞/D]", lambda row: row["native_gradient_error_rms"]))
    for row_axes, record in zip(axes, records, strict=True):
        for axis, (title, value) in zip(row_axes, metrics, strict=True):
            values = [value(row) for row in record["records"]]
            axis.bar(np.arange(8), values, color=colors)
            axis.set_xticks(np.arange(8), labels, rotation=55, ha="right", fontsize=8)
            axis.set_title(title, fontsize=10)
            axis.grid(axis="y", alpha=.2)
            axis.set_axisbelow(True)
            axis.set_ylim(0, max(values)*1.18)
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        seed = "Laminar seed" if record["source_sgs"] == "none" else "LES seed"
        row_axes[0].set_ylabel(f'{record["small_fvm_cells"]:,} small FVM cells\n{seed}', fontsize=10)
    subtitle = "Different coarse/medium physical seeds; compare variants within each row" if len(records) > 1 else "Compare source variants in the same frozen physical state"
    fig.suptitle("Conservative source updates: frozen fully 3D cube, fixed 1,728-panel body\n"+subtitle, fontsize=13)
    fig.savefig(figure, dpi=165)
    plt.close(fig)
    artifacts.append({"path": label(figure), "sha256": digest(figure)})
    dependencies = [Path(__file__).resolve(), ROOT / "studies/coupler_accuracy/verify_sampled_native_face_3d.py"]
    verification_sources = []
    for path in dependencies:
        target = args.output.parent / (args.output.stem+"-sources") / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        verification_sources.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path)})
    result = {"schema": "openonda-shared-trace-induction-verification-3d/1", "status": "passed", "spatial_dimensions": 3,
              "source_records": source_records, "metric_checks": len(metric_differences),
              "maximum_metric_difference": max(metric_differences), "array_checks": len(array_differences),
              "maximum_independent_array_difference": max(array_differences), "component_tests_passed": len(cases),
              "source_budgets": budget_records, "experiments": records, "impulse_endpoint_check": impulse_endpoint_check,
              "artifacts": artifacts, "verification_sources": verification_sources,
              "limitations": ["This verifies source budgets and frozen physical observations, not advancing force/profile accuracy.",
                              "Native gradients are independently accumulated from stored sampled velocities; the induction kernels retain their earlier component qualifications.",
                              "Impulse conservation is relative to each corresponding native-face-moment baseline, and concerns raw affine source vorticity."]}
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("source_budgets", "experiments", "impulse_endpoint_check", "artifacts", "verification_sources")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--induction", nargs="+", type=Path, default=[RESULTS / ("cube-3d-shared-trace-induction-"+name) for name in ("coarse", "medium-laminar")])
    parser.add_argument("--output", type=Path, default=RESULTS / "shared-trace-induction-verification.json")
    parser.add_argument("--moment-impulse", type=Path)
    args = parser.parse_args()
    args.induction = [path.resolve() for path in args.induction]
    args.output = args.output.resolve()
    if args.moment_impulse:
        args.moment_impulse = args.moment_impulse.resolve()
    run(args)
