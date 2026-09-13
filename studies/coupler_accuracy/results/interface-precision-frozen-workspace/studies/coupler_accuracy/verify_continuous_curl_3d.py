#!/usr/bin/env python3
"""Independently verify continuous-curl sources and frozen physical results."""

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
from studies.coupler_accuracy.continuous_velocity_curl_3d import ContinuousVelocityCurl
from studies.coupler_accuracy.verify_sampled_native_face_3d import independent_gradient
from studies.coupler_accuracy.verify_shared_trace_induction_3d import (
    arrays,
    check_sources,
    digest,
    label,
    rms,
    tangent,
)

ROOT = Path(__file__).resolve().parents[2]


def check_source(directory, check):
    report = json.loads((directory / "cube-continuous-curl-sources-3d.json").read_text())
    count = check_sources(directory, report)
    data = arrays(directory / "continuous-curl-source-fields.npz")
    mesh_path = ROOT / next(row["path"] for row in report["sources"] if row["path"].endswith("/small-native-mesh.npz"))
    mesh = load_native_mesh(mesh_path)
    c = ContinuousVelocityCurl.from_mesh(mesh, data["cell_centroid"])
    np.testing.assert_array_equal(c.position, data["node_position"])
    np.testing.assert_array_equal(c.tetrahedra, data["tetrahedron_nodes"])
    u, omega = data["nodal_velocity"], data["tetrahedron_vorticity"]
    # Stokes on every tetrahedral face uses edge velocity integrals, independent
    # of the reconstruction's inverse-edge gradient and curl implementation.
    vertices = c.position[c.face_nodes]
    sf = np.cross(vertices[:, 1]-vertices[:, 0], vertices[:, 2]-vertices[:, 0])/2
    edge = np.roll(vertices, -1, axis=1)-vertices
    normal = sf/np.linalg.norm(sf, axis=1)[:, None]
    rows = []
    for state in range(len(u)):
        face_u = u[state, c.face_nodes]
        stokes = np.sum(.5*(face_u+np.roll(face_u, -1, axis=1))*edge, axis=(1, 2))
        own_flux = np.sum(omega[state, c.source.owners]*sf, axis=1)
        check(own_flux, stokes, "edge_stokes_flux", 2e-14)
        inner = c.source.neighbours >= 0
        other_flux = np.sum(omega[state, c.source.neighbours[inner]]*sf[inner], axis=1)
        check(other_flux, stokes[inner], "edge_stokes_flux", 2e-14)
        jump = omega[state, c.source.owners].copy()
        jump[inner] -= omega[state, c.source.neighbours[inner]]
        raw_jump = np.sum(jump*normal, axis=1)
        check(raw_jump, data["normal_curl_jump"][state], "normal_jump")
        check(np.max(np.abs(raw_jump)), report["records"][state]["maximum_normal_curl_jump"], "reported_metric")
        q = np.column_stack([np.bincount(c.parent, weights=c.volume*u[state, c.tetrahedra, axis].mean(axis=1), minlength=mesh["n_cells"])
                             for axis in range(3)])
        check(q, data["cell_velocity_integral_change"][state], "cell_integral", 2e-16)
        g = np.column_stack([np.bincount(c.parent, weights=c.volume*omega[state, :, axis], minlength=mesh["n_cells"])
                             for axis in range(3)])
        relative = c.centroid-c.cell_centroid[c.parent]
        m = np.empty((mesh["n_cells"], 3, 3))
        for i in range(3):
            for j in range(3):
                m[:, i, j] = np.bincount(c.parent, weights=c.volume*relative[:, i]*omega[state, :, j], minlength=mesh["n_cells"])
        check(g, data["cell_circulation"][state], "cell_circulation")
        check(m, data["first_moment"][state], "cell_first_moment")
        total_h = np.einsum("t,ti,tj->ij", c.volume, c.centroid, omega[state])
        total_q = data["cell_velocity_integral_change"][state].sum(axis=0)
        x, y, z = total_q
        expected_h = np.array([[0., z, -y], [-z, 0., x], [y, -x, 0.]])
        check(total_h, expected_h, "global_first_moment")
        check(total_h, report["records"][state]["total_first_moment"], "reported_metric")
        check(np.sum(c.volume[:, None]*omega[state], axis=0), 0, "global_circulation")
        np.testing.assert_array_equal(u[state, c.boundary_nodes], 0)
        np.testing.assert_array_equal(omega[state, data["volume_weight"][c.parent] == 0], 0)
        rows.append({"name": str(data["source_names"][state]), "maximum_edge_stokes_difference": float(np.max(np.abs(own_flux-stokes))),
                     "maximum_normal_curl_jump": float(np.max(np.abs(raw_jump))),
                     "maximum_cell_integral_difference": float(np.max(np.abs(q-data["cell_velocity_integral_change"][state]))),
                     "maximum_global_first_moment_difference": float(np.max(np.abs(total_h-expected_h)))})
    return {"directory": label(directory), "source_records": count, "records": rows}


def run(args):
    differences, source_records, source_results, experiments, artifacts = [], 0, [], [], []

    def check(actual, expected, category="physical_metric", tolerance=2e-12):
        value = float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)
        differences.append({"category": category, "maximum_difference": value})

    for directory in args.source:
        result = check_source(directory, check)
        source_results.append(result)
        source_records += result["source_records"]
        for name in ("cube-continuous-curl-sources-3d.json", "continuous-curl-source-fields.npz"):
            path = directory / name
            artifacts.append({"path": label(path), "sha256": digest(path)})
    for directory in args.induction:
        report_path = directory / "cube-continuous-curl-induction-3d.json"
        field_path = directory / "continuous-curl-physical-fields.npz"
        checkpoint_path = directory / "continuous-curl-induction-checkpoint.npz"
        report = json.loads(report_path.read_text())
        source_records += check_sources(directory, report)
        for path in (report_path, field_path, checkpoint_path):
            artifacts.append({"path": label(path), "sha256": digest(path)})
        base_path = ROOT / next(row["path"] for row in report["sources"] if row["path"].endswith("/shared-trace-physical-fields.npz"))
        donor_path = ROOT / next(row["path"] for row in report["sources"] if row["path"].endswith("/shared-trace-source-fields.npz"))
        mesh_path = ROOT / next(row["path"] for row in report["sources"] if row["path"].endswith("/full-native-mesh.npz"))
        base_report = json.loads((base_path.parent / "cube-shared-trace-induction-3d.json").read_text())
        base, donor, fields, checkpoint = arrays(base_path), arrays(donor_path), arrays(field_path), arrays(checkpoint_path)
        mesh = load_native_mesh(mesh_path)
        geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
        np.testing.assert_array_equal(fields["position"], base["position"])
        np.testing.assert_array_equal(checkpoint["position"][checkpoint["physical_target_indices"]], base["position"])
        normal, area = base["normal"], base["area"]
        v, step = fields["curl_values"], float(fields["curl_step"])
        low, high = np.empty((4, 24, 3)), np.empty((4, 24, 3))
        for axis, (i, j) in enumerate(((1, 2), (2, 0), (0, 1))):
            low[:, :, axis] = (v[:, :, i, 0, j]-v[:, :, i, 1, j]-v[:, :, j, 0, i]+v[:, :, j, 1, i])/(2*step)
            high[:, :, axis] = (v[:, :, i, 2, j]-v[:, :, i, 3, j]-v[:, :, j, 2, i]+v[:, :, j, 3, i])/step
        check(low, fields["curl"], "exterior_curl")
        check(high, fields["curl_half"], "exterior_curl")
        np.testing.assert_allclose(high[[1, 3]], 0, rtol=0, atol=1e-6)
        for state, row in enumerate(report["records"]):
            name, baseline_name = row["name"], row["baseline"]
            actual = fields[name+"__velocity"]
            expected = (base[baseline_name+"__velocity"]+checkpoint["correction_induction"][state, checkpoint["physical_target_indices"]]
                        +fields[name+"__body_velocity_change"])
            check(actual, expected, "field_identity")
            gradient, dn, face_u = independent_gradient(mesh, geometry, base, actual[base["target__sample"]])
            gt = tangent(dn, normal)
            for key, value in (("cell_gradient", gradient), ("normal_gradient", dn), ("tangential_gradient", gt), ("face_velocity", face_u)):
                check(value, fields[name+"__"+key], "independent_native_gradient")
            gt_error = gt-base["reference_tangential_gradient"]
            check(rms(gt_error, area), row["native_gradient_error_rms"])
            normals = {}
            for mode, velocity in (("point", actual[base["target__boundary"]]), ("native", face_u)):
                raw = np.sum(velocity*normal, axis=1)
                flux = raw @ area
                un = np.sum((velocity-(flux/area.sum())*normal)*normal, axis=1)
                normals[mode] = un
                check(un, fields[name+"__"+mode+"_normal_velocity"], "independent_normal_velocity")
                check(rms((un-base["reference_normal_velocity"])[:, None], area), row[mode+"_normal_velocity_error_rms"])
                check(abs(flux), row["boundary_flux"][mode]["raw_mismatch"])
                check(abs(flux/area.sum()), row["boundary_flux"][mode]["applied_correction"])
                assert abs(un @ area) < 1e-11
            for group in ("near_body", "held_outer", "wake"):
                ids = base[group+"__cell_ids"]
                check(rms(actual[base["target__"+group]]-donor["full_velocity"][ids], donor["full_volume"][ids]),
                      row["cell_velocity_rms_over_Uinf"][group])
            wall = actual[base["target__wall"]]
            wall_n = np.sum(wall*base["wall_normal"], axis=1)
            check(rms(wall_n[:, None]), row["wall_normal_velocity_rms_over_Uinf"])
            check(rms(wall-wall_n[:, None]*base["wall_normal"]), row["wall_tangential_velocity_rms_over_Uinf"])
            check(base["panel_area"] @ fields[name+"__body_strength_change"], row["body_source_flux_change"])
            check(rms(high[state]), row["exterior_curl_rms"])
            check(rms(high[state]-low[state]), row["exterior_curl_step_halving_difference_rms"])
            for i, side in enumerate(("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")):
                selected = base["face_group"] == i
                check(rms(gt_error[selected], area[selected]), row["face_groups"][side]["native_gradient_error_rms"])
                check(rms((normals["native"]-base["reference_normal_velocity"])[selected, None], area[selected]),
                      row["face_groups"][side]["native_normal_velocity_error_rms"])
        experiments.append({"directory": label(directory), "small_fvm_cells": report["small_fvm_cells"],
                            "records": report["records"], "baselines": [base_report["records"][i] for i in (2, 5)]})
    test_path = ROOT / "studies/coupler_accuracy/results/3d-continuous-velocity-curl-regression.xml"
    suites = ET.parse(test_path).getroot().findall(".//testsuite")
    tests = sum(int(suite.get("tests", 0)) for suite in suites)
    assert tests == 5 and all(int(suite.get(key, 0)) == 0 for suite in suites for key in ("failures", "errors", "skipped"))
    artifacts.append({"path": label(test_path), "sha256": digest(test_path)})
    own = Path(__file__).resolve()
    for path in (own, ROOT / "studies/coupler_accuracy/verify_sampled_native_face_3d.py",
                 ROOT / "studies/coupler_accuracy/verify_shared_trace_induction_3d.py"):
        archive = args.output.parent / (args.output.stem+"-sources") / path.relative_to(ROOT)
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(path.read_bytes())
        artifacts.append({"path": label(archive), "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()})
    if experiments:
        figure, axes = plt.subplots(len(experiments), 4, figsize=(18, 4.8*len(experiments)), squeeze=False)
        for row, experiment in enumerate(experiments):
            records = experiment["records"]
            bars = [experiment["baselines"][0], *records[:2], experiment["baselines"][1], *records[2:]]
            labels = ["Point base", "Point affine", "Point curl", "Mean base", "Mean affine", "Mean curl"]
            colors = ["#64748b", "#93c5fd", "#1d4ed8", "#64748b", "#a7f3d0", "#047857"]
            values = [[r["cell_velocity_rms_over_Uinf"]["near_body"] for r in bars],
                      [r["native_normal_velocity_error_rms"] for r in bars], [r["native_gradient_error_rms"] for r in bars]]
            for col, (value, title) in enumerate(zip(values, ("Near-body velocity error / U∞", "Native normal-velocity error / U∞", "Native derivative error [U∞/D]"), strict=True)):
                axes[row, col].bar(labels, value, color=colors)
                axes[row, col].set_title(title)
                axes[row, col].tick_params(axis="x", labelrotation=55)
                axes[row, col].grid(axis="y", alpha=.2)
                axes[row, col].set_axisbelow(True)
            axes[row, 0].set_ylabel(f"{experiment['small_fvm_cells']:,} small-FVM cells")
            axes[row, 3].bar(["Point affine", "Point curl", "Mean affine", "Mean curl"],
                             [r["exterior_curl_rms"] for r in records], color=[colors[i] for i in (1, 2, 4, 5)])
            axes[row, 3].set_yscale("log")
            axes[row, 3].set_title("Exterior correction curl [U∞/D]\n24 diagnostic points")
            axes[row, 3].tick_params(axis="x", labelrotation=55)
        figure.suptitle("Frozen fully 3D cube: each affine/curl pair has identical cell circulation and first moments", fontsize=15)
        figure.tight_layout(rect=(0, 0, 1, .91))
        image_path = args.output.with_suffix(".png")
        figure.savefig(image_path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        artifacts.append({"path": label(image_path), "sha256": digest(image_path)})
    result = {"schema": "openonda-continuous-curl-verification-3d/1", "status": "passed", "spatial_dimensions": 3,
              "source_records": source_records, "checks": len(differences), "maximum_difference": max(row["maximum_difference"] for row in differences),
              "component_tests_passed": tests, "source_results": source_results, "experiments": experiments,
              "checks_by_category": {category: {"count": sum(row["category"] == category for row in differences),
                                                   "maximum_difference": max(row["maximum_difference"] for row in differences if row["category"] == category)}
                                     for category in sorted({row["category"] for row in differences})},
              "artifacts": artifacts, "limitations": ["This qualifies frozen sources and observations, not advancing particle transfer, forces or velocity profiles.",
                                                       "Induction retains its component volume-quadrature qualification; full physical induction is not rerun by this verifier."]}
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: result[key] for key in ("status", "source_records", "checks", "maximum_difference", "component_tests_passed", "checks_by_category")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--induction", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.source = [path.resolve() for path in args.source]
    args.induction = [path.resolve() for path in args.induction]
    args.output = args.output.resolve()
    run(args)
