#!/usr/bin/env python3
"""Recompute and compare the body-resolution and quadratic-moment studies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "studies/coupler_accuracy/results"


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def rms(value, weight=None):
    return float(np.sqrt(np.average(np.sum(np.asarray(value)**2, axis=-1), weights=weight)))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def body_metrics(row):
    return [row["cell_velocity_rms_over_Uinf"]["near_body"], row["cell_velocity_rms_over_Uinf"]["held_outer"],
            row["boundary_normal_velocity_rms_error_over_Uinf"], row["boundary_native_tangential_gradient_rms_error"]["centred_half"],
            row["wall_normal_velocity_rms_over_Uinf"], row["wall_tangential_velocity_rms_over_Uinf"]]


def run():
    sources, differences, derivative_differences = [], [], []

    def verify_sources(directory, report):
        assert report["status"] == "complete" and report["spatial_dimensions"] == 3
        assert report["sources"] == json.loads((directory / "sources-at-start.json").read_text())
        for row in report["sources"]:
            path = ROOT / row["path"]
            archive = directory / "sources" / row["path"]
            assert digest(archive if archive.exists() else path) == row["sha256"]
            assert digest(path) == row["sha256"], f"Post-run source change: {path}"
            sources.append({"result_directory": directory.name, **row})

    full = arrays(RESULTS / "cube-3d-native-volume-induction/volume-induction-comparison-fields.npz")
    boundary = arrays(RESULTS / "cube-3d-integrated-velocity-curl-reconstruction-boundary/boundary-fields.npz")
    previous = arrays(RESULTS / "cube-3d-reconstructed-moment-overlap/reconstructed-moment-overlap-fields.npz")
    body_reports, body_fields = {}, {}
    for count in (108, 432, 1728, 6912):
        directory = RESULTS / ("cube-3d-moment-panels-108-qualified" if count == 108 else f"cube-3d-moment-panels-{count}")
        report = json.loads((directory / "cube-moment-panel-resolution-3d.json").read_text())
        verify_sources(directory, report)
        field = arrays(directory / "moment-panel-resolution-fields.npz")
        assert report["panels"] == len(field["panel_area"]) == count
        assert report["panel_velocity_evaluation"].endswith("far_field_min_panels=panel_count+1")
        np.testing.assert_allclose(field["panel_area"].sum(), 6, rtol=0, atol=2e-12)
        np.testing.assert_array_equal(field["position"], previous["position"])
        if count > 108:
            np.testing.assert_array_equal(field["incident_velocity"], body_fields[108]["incident_velocity"])
        selections = {key: slice(*limits) for key, limits in report["target_slices"].items()}
        h = report["derivative_steps"][0]
        stencils = {"centred": (["plus", "minus"], [1, -1], 2*h),
                    "centred_half": (["plus_half", "minus_half"], [1, -1], h),
                    "exterior": (["boundary", "plus", "plus2"], [-3, 4, -1], 2*h),
                    "interior": (["boundary", "minus", "minus2"], [3, -4, 1], 2*h),
                    "exterior_half": (["boundary", "plus_half", "plus"], [-3, 4, -1], h),
                    "interior_half": (["boundary", "minus_half", "minus"], [3, -4, 1], h)}
        for state, row in enumerate(report["records"]):
            name, velocity = row["name"], field[row["name"]+"__velocity"]
            np.testing.assert_array_equal(velocity, field["incident_velocity"][state]+[1, 0, 0]+field[name+"__body_velocity"])
            if count == 108:
                np.testing.assert_allclose(velocity, previous[name+"__velocity"], rtol=0, atol=1e-12)
            for group, recorded in row["cell_velocity_rms_over_Uinf"].items():
                ids = full[group+"__cell_ids"]
                differences.append(abs(rms(velocity[selections[group]]-full["native_cell_velocity"][ids], full["native_cell_volume"][ids])-recorded))
            raw = velocity[selections["boundary"]]
            raw_un = np.sum(raw*boundary["normal"], axis=1)
            correction = np.dot(raw_un, boundary["area"])/boundary["area"].sum()
            un = np.sum((raw-correction*boundary["normal"])*boundary["normal"], axis=1)
            np.testing.assert_allclose(un, field[name+"__boundary_normal_velocity"], rtol=0, atol=4e-16)
            differences.append(abs(rms((un-boundary["fvm_normal_velocity"])[:, None], boundary["area"])-row["boundary_normal_velocity_rms_error_over_Uinf"]))
            for side, (keys, coefficient, denominator) in stencils.items():
                derivative = np.einsum("s,snd->nd", coefficient, np.stack([velocity[selections[key]] for key in keys]))/denominator
                derivative /= np.linalg.norm(boundary["normal"], axis=1)[:, None]
                normal = boundary["native_unit_normal"]
                derivative -= np.sum(derivative*normal, axis=1)[:, None]*normal
                derivative_differences.append(float(np.max(np.abs(derivative-field[name+"__gradient_"+side]))))
                differences.append(abs(rms(derivative-boundary["fvm_native_flux_tangential_normal_gradient"], boundary["native_vector_area"])
                                       - row["boundary_native_tangential_gradient_rms_error"][side]))
            wall = velocity[selections["wall"]]
            wall_un = np.sum(wall*full["wall_normal"], axis=1)
            differences += [abs(float(np.sqrt(np.mean(wall_un**2)))-row["wall_normal_velocity_rms_over_Uinf"]),
                            abs(rms(wall-wall_un[:, None]*full["wall_normal"])-row["wall_tangential_velocity_rms_over_Uinf"]),
                            abs(rms(field[name+"__collocation_residual"][:, None], field["panel_area"])-row["discrete_neumann_collocation_residual_rms"]),
                            abs(float(field["panel_area"] @ field["panel_strength"][state])-row["body_source_flux"])]
            assert abs(row["body_source_flux"]) < 1e-10 and max(row["derivative_step_halving_maximum_difference"].values()) < 1e-6
        body_reports[count], body_fields[count] = report, field
    refinement = []
    for name in body_reports[108]["source_names"]:
        row = {"name": name, "field_change_1728_to_6912": {}}
        difference = body_fields[6912][name+"__velocity"]-body_fields[1728][name+"__velocity"]
        for group in ("near_body", "held_outer", "wake"):
            row["field_change_1728_to_6912"][group] = rms(difference[selections[group]], full["native_cell_volume"][full[group+"__cell_ids"]])
        row["field_change_1728_to_6912"]["boundary_normal"] = rms((body_fields[6912][name+"__boundary_normal_velocity"]-body_fields[1728][name+"__boundary_normal_velocity"])[:, None], boundary["area"])
        row["field_change_1728_to_6912"]["boundary_gradient"] = rms(body_fields[6912][name+"__gradient_centred_half"]-body_fields[1728][name+"__gradient_centred_half"], boundary["native_vector_area"])
        refinement.append(row)
    quadratic_reports = {}
    for resolution in ("coarse", "medium"):
        directory = RESULTS / f"cube-3d-quadratic-moment-reconstruction-{resolution}"
        report = json.loads((directory / "cube-quadratic-moment-reconstruction-3d.json").read_text())
        verify_sources(directory, report)
        source, field = [arrays(directory / name) for name in ("quadratic-moment-source-fields.npz", "quadratic-moment-induction-fields.npz")]
        old = arrays(RESULTS / f"cube-3d-manufactured-induction-{resolution}/manufactured-source-fields.npz")
        exact = arrays(RESULTS / f"cube-3d-manufactured-linear-induction-{resolution}/linear-source-fields.npz")
        targets = arrays(RESULTS / f"cube-3d-manufactured-induction-{resolution}/manufactured-induction-fields.npz")
        for key in ("position", "exact_velocity"):
            np.testing.assert_array_equal(field[key], targets[key])
        for key in ("volume", "centroid", "covariance"):
            np.testing.assert_array_equal(source[key], exact[key])
        for kind, diagnostic in report["stencil_diagnostics"].items():
            assert diagnostic["maximum_condition"] == float(source[kind+"__condition"].max())
            assert diagnostic["minimum_neighbour_count"] == int(source[kind+"__neighbour_count"].min())
            assert diagnostic["maximum_neighbour_count"] == int(source[kind+"__neighbour_count"].max())
            assert diagnostic["maximum_rings"] == int(source[kind+"__rings"].max())
            assert diagnostic["cells_requiring_third_ring"] == int(np.sum(source[kind+"__rings"] > 2))
        radius = np.max(np.abs(source["centres"]), axis=1)
        regions = {"all": np.ones(len(radius), dtype=bool), "near_body": radius < .8, "outer_layer": (radius > 1.1) & (radius < 1.5)}
        for state, row in enumerate(report["records"]):
            index = (0.35, 0.15).index(row["width"])
            gamma, moment = source["circulation"][state], source["first_moment"][state]
            gradient = np.linalg.solve(source["covariance"], moment)
            exact_gradient = np.linalg.solve(source["covariance"], exact["first_moment"][index])
            delta = gradient-exact_gradient
            squared = np.einsum("nij,nik,nkj->n", delta, source["covariance"], delta)
            norm2 = np.einsum("nij,nik,nkj->n", exact_gradient, source["covariance"], exact_gradient)
            for region, selected in regions.items():
                volume = source["volume"][selected]
                error = rms((gamma-old["vorticity_integral"][index])[selected]/volume[:, None], volume)
                norm = rms(old["vorticity_integral"][index, selected]/volume[:, None], volume)
                computed = {"circulation_density_rms": error, "circulation_relative_error": error/norm,
                            "moment_variation_density_rms": float(np.sqrt(squared[selected].sum()/volume.sum())),
                            "moment_variation_relative_error": float(np.sqrt(squared[selected].sum()/norm2[selected].sum()))}
                differences.extend(abs(value-row["source_error"][region][key]) for key, value in computed.items())
            if row["method"].startswith("weak_"):
                kind = "average" if "average_input" in row["method"] else "point"
                np.testing.assert_allclose(gamma.sum(axis=0), 0, rtol=0, atol=2e-13)
                global_moment = moment.sum(axis=0)+np.einsum("ni,nj->ij", source["centroid"], gamma)
                expected = -np.cross(np.eye(3), source[kind+"__cell_velocity_integral"][index].sum(axis=0))
                np.testing.assert_allclose(global_moment, expected, rtol=0, atol=2e-12)
            for group, limits in report["target_slices"].items():
                delta = field["velocity"][state, slice(*limits)]-field["exact_velocity"][index, slice(*limits)]
                differences += [abs(rms(delta)-row["velocity_error"][group]["rms_over_reference_speed"]),
                                abs(float(np.linalg.norm(delta, axis=1).max())-row["velocity_error"][group]["maximum_over_reference_speed"])]
        quadratic_reports[resolution] = report
    tests = ET.parse(RESULTS / "3d-quadratic-moments-regression.xml")
    assert len(list(tests.iter("testcase"))) == 5 and not any(list(tests.iter(tag)) for tag in ("failure", "error", "skipped"))
    assert max(differences) < 1e-12 and max(derivative_differences) < 1e-8
    plot(body_reports, quadratic_reports)
    destination = RESULTS / "moment-panels-and-quadratic-verification-sources" / Path(__file__).relative_to(ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, destination)
    output = {"status": "complete", "spatial_dimensions": 3, "source_records_verified": len(sources), "source_checks": sources,
              "metrics_recomputed": len(differences), "maximum_metric_difference": max(differences),
              "maximum_independent_derivative_replay_difference": max(derivative_differences),
              "baseline_velocity_maximum_difference": body_reports[108]["baseline_velocity_maximum_difference"],
              "new_passing_tests": 5, "panel_refinement": refinement,
              "verification_artifacts": [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in
                                         (Path(__file__), RESULTS / "cube-3d-moment-panel-resolution.png", RESULTS / "cube-3d-quadratic-moment-reconstruction.png",
                                          RESULTS / "3d-quadratic-moments-regression.xml",
                                          RESULTS / "cube-3d-native-moment-reconstruction-coarse/cube-native-moment-reconstruction-3d.json",
                                          RESULTS / "cube-3d-native-moment-reconstruction-medium/cube-native-moment-reconstruction-3d.json")]}
    (RESULTS / "moment-panels-and-quadratic-verification.json").write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps({key: value for key, value in output.items() if key not in ("source_checks", "verification_artifacts")}, indent=2))


def plot(body_reports, quadratic_reports):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4))
    labels = ["Constant volume", "Circulation-gradient moments", "Native-face moments", "Linear-face moments"]
    colors = ["#718397", "#bf9445", "#558aaa", "#237d7a"]
    counts = list(body_reports)
    titles = ["Near-body velocity", "Unused outer-layer velocity", "Boundary normal velocity", "Boundary tangential derivative", "Wall normal velocity", "Wall tangential velocity"]
    for metric, (ax, title) in enumerate(zip(axes.ravel(), titles, strict=True)):
        for state, (label, color) in enumerate(zip(labels, colors, strict=True)):
            ax.plot(range(len(counts)), [body_metrics(body_reports[n]["records"][state])[metric] for n in counts], "o-", color=color, label=label)
        ax.set_xticks(range(len(counts)), [f"{n:,}" for n in counts])
        ax.set_xlabel("Body source panels")
        ax.set_ylabel("RMS error (U∞/D)" if metric == 3 else "RMS error / U∞")
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.2)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .95), ncols=4, fontsize=9)
    fig.suptitle("Frozen 3D cube: refine the body with all circulation and moments fixed", fontsize=14)
    fig.text(.5, .012, "Identical 2,840-cell small domain and common targets · Exact source-panel velocity kernel at every resolution", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .89), h_pad=2)
    fig.savefig(RESULTS / "cube-3d-moment-panel-resolution.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.7))
    labels = ["P1 faces\npoint input", "P2 faces\npoint input", "P1 faces\naverage input", "P2 faces\naverage input", "Native Γ\nP2 point M", "Exact Γ,M\ncontrol"]
    for column, width in enumerate((.35, .15)):
        for resolution, color in (("coarse", "#2374ab"), ("medium", "#c95627")):
            old = json.loads((RESULTS / f"cube-3d-native-moment-reconstruction-{resolution}/cube-native-moment-reconstruction-3d.json").read_text())
            exact = json.loads((RESULTS / f"cube-3d-manufactured-linear-induction-{resolution}/cube-manufactured-linear-induction-3d.json").read_text())
            def find(report, method, selected_width=width):
                return next(row for row in report["records"] if row["width"] == selected_width and row.get("method") == method)
            rows = [find(old, "weak_linear_faces_point_input"), find(quadratic_reports[resolution], "weak_quadratic_faces_point_input"),
                    find(old, "weak_linear_faces_average_input"), find(quadratic_reports[resolution], "weak_quadratic_faces_average_input"),
                    find(quadratic_reports[resolution], "native_circulation_quadratic_point_moments")]
            exact_row = next(row for row in exact["records"] if row["width"] == width)
            for index, group in enumerate(("near_body", "wall")):
                values = [row["velocity_error"][group]["rms_over_reference_speed"] for row in rows]
                values.append(exact_row["velocity_errors"][group]["rms_over_reference_speed"])
                axes[index, column].plot(range(len(labels)), values, "o-", color=color, label=resolution.capitalize())
        axes[0, column].set_title(f"{'Broad' if width == .35 else 'Thin'} 3D field · width = {width} D")
        for index, ax in enumerate(axes[:, column]):
            ax.set_ylabel(("Near-body velocity" if index == 0 else "Wall velocity")+" RMS error / Uref")
            ax.set_xticks(range(len(labels)), labels, fontsize=8)
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=.2)
            ax.legend()
    fig.suptitle("Quadratic velocity reconstruction supplies affine vorticity moments", fontsize=14)
    fig.text(.5, .012, "Common fully 3D meshes and targets · Point and exact-average inputs are separate · No target fitting", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .94), h_pad=2)
    fig.savefig(RESULTS / "cube-3d-quadratic-moment-reconstruction.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    run()
