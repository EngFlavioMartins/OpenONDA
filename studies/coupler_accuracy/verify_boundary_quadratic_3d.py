#!/usr/bin/env python3
"""Independently verify boundary-informed reconstruction and physical overlap."""

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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rms(value, weight=None):
    return float(np.sqrt(np.average(np.sum(np.asarray(value)**2, axis=-1), weights=weight)))


def run():
    sources, metric_differences, derivative_differences = [], [], []

    def verify_sources(directory, report):
        assert report["status"] == "complete" and report["spatial_dimensions"] == 3
        assert report["sources"] == json.loads((directory / "sources-at-start.json").read_text())
        for row in report["sources"]:
            path = ROOT / row["path"]
            archive = directory / "sources" / row["path"]
            assert digest(archive if archive.exists() else path) == row["sha256"]
            assert digest(path) == row["sha256"]
            sources.append({"result_directory": directory.name, **row})

    manufactured = {}
    for resolution in ("coarse", "medium"):
        directory = RESULTS / f"cube-3d-boundary-quadratic-{resolution}"
        report = json.loads((directory / "cube-boundary-quadratic-moments-3d.json").read_text())
        verify_sources(directory, report)
        source, field = [arrays(directory / name) for name in ("boundary-quadratic-source-fields.npz", "boundary-quadratic-induction-fields.npz")]
        previous_dir = RESULTS / f"cube-3d-quadratic-moment-reconstruction-{resolution}"
        previous = arrays(previous_dir / "quadratic-moment-source-fields.npz")
        truth = arrays(RESULTS / f"cube-3d-manufactured-induction-{resolution}/manufactured-source-fields.npz")
        target_truth = arrays(RESULTS / f"cube-3d-manufactured-induction-{resolution}/manufactured-induction-fields.npz")
        exact = arrays(RESULTS / f"cube-3d-manufactured-linear-induction-{resolution}/linear-source-fields.npz")
        for key in ("position", "exact_velocity"):
            np.testing.assert_array_equal(field[key], target_truth[key])
        for key in ("centres", "volume", "centroid", "covariance"):
            np.testing.assert_array_equal(source[key], previous[key])
        for kind, stats in report["stencil_diagnostics"].items():
            assert stats["maximum_condition"] == float(source[kind+"__condition"].max())
            assert stats["cells_with_wall_observations"] == int(np.sum(source[kind+"__boundary_observation_count"] > 0))
            assert stats["potentially_changed_source_cells"] == int(source[kind+"__source_change_mask"].sum())
            assert stats["boundary_faces"] == len(np.unique(source[kind+"__boundary_face_ids"]))
            assert stats["boundary_points"] == len(source[kind+"__boundary_position"])
            np.testing.assert_allclose(source[kind+"__boundary_area_weight"].sum(), 6, rtol=0, atol=2e-14)
            np.testing.assert_allclose(np.max(np.abs(source[kind+"__boundary_position"]), axis=1), .5, rtol=0, atol=2e-15)
        radius = np.max(np.abs(source["centres"]), axis=1)
        regions = {"all": np.ones(len(radius), dtype=bool), "near_body": radius < .8, "outer_layer": (radius > 1.1) & (radius < 1.5)}
        for state, row in enumerate(report["records"]):
            index = (.35, .15).index(row["width"])
            gamma, moment = source["circulation"][state], source["first_moment"][state]
            kind = "average" if "average_input" in row["method"] else "point"
            changed = source[kind+"__source_change_mask"]
            np.testing.assert_array_equal(gamma[~changed], previous["circulation"][state, ~changed])
            np.testing.assert_array_equal(moment[~changed], previous["first_moment"][state, ~changed])
            if row["method"].startswith("native_circulation_"):
                np.testing.assert_array_equal(gamma, previous["circulation"][state])
            else:
                np.testing.assert_allclose(gamma.sum(axis=0), 0, rtol=0, atol=3e-13)
                global_moment = moment.sum(axis=0)+np.einsum("ni,nj->ij", source["centroid"], gamma)
                expected = -np.cross(np.eye(3), source[kind+"__cell_velocity_integral"][index].sum(axis=0))
                np.testing.assert_allclose(global_moment, expected, rtol=0, atol=3e-12)
            gradient = np.linalg.solve(source["covariance"], moment)
            exact_gradient = np.linalg.solve(source["covariance"], exact["first_moment"][index])
            delta = gradient-exact_gradient
            squared = np.einsum("nij,nik,nkj->n", delta, source["covariance"], delta)
            norm2 = np.einsum("nij,nik,nkj->n", exact_gradient, source["covariance"], exact_gradient)
            for region, selected in regions.items():
                volume = source["volume"][selected]
                error = rms((gamma-truth["vorticity_integral"][index])[selected]/volume[:, None], volume)
                norm = rms(truth["vorticity_integral"][index, selected]/volume[:, None], volume)
                computed = {"circulation_density_rms": error, "circulation_relative_error": error/norm,
                            "moment_variation_density_rms": float(np.sqrt(squared[selected].sum()/volume.sum())),
                            "moment_variation_relative_error": float(np.sqrt(squared[selected].sum()/norm2[selected].sum()))}
                metric_differences += [abs(value-row["source_error"][region][key]) for key, value in computed.items()]
            for group, limits in report["target_slices"].items():
                delta = field["velocity"][state, slice(*limits)]-field["exact_velocity"][index, slice(*limits)]
                metric_differences += [abs(rms(delta)-row["velocity_error"][group]["rms_over_reference_speed"]),
                                      abs(float(np.linalg.norm(delta, axis=1).max())-row["velocity_error"][group]["maximum_over_reference_speed"])]
        manufactured[resolution] = report
    full = arrays(RESULTS / "cube-3d-native-volume-induction/volume-induction-comparison-fields.npz")
    boundary = arrays(RESULTS / "cube-3d-integrated-velocity-curl-reconstruction-boundary/boundary-fields.npz")
    old_moments = arrays(RESULTS / "cube-3d-reconstructed-moment-overlap/moment-overlap-induction-checkpoint.npz")
    physical, checkpoint_by_count, baseline_difference = {}, {}, []
    for count in (108, 1728, 6912):
        directory = RESULTS / f"cube-3d-boundary-quadratic-overlap-{count}-qualified"
        report = json.loads((directory / "cube-boundary-quadratic-overlap-3d.json").read_text())
        verify_sources(directory, report)
        checkpoint, field = [arrays(directory / name) for name in ("physical-boundary-quadratic-checkpoint.npz", "physical-boundary-quadratic-fields.npz")]
        baseline_dir = RESULTS / ("cube-3d-moment-panels-108-qualified" if count == 108 else f"cube-3d-moment-panels-{count}")
        base = arrays(baseline_dir / "moment-panel-resolution-fields.npz")
        np.testing.assert_array_equal(checkpoint["position"], base["position"])
        np.testing.assert_array_equal(field["position"], base["position"])
        np.testing.assert_array_equal(checkpoint["panel_centre"], base["panel_centre"])
        np.testing.assert_array_equal(checkpoint["delta_circulation"][:4], 0)
        np.testing.assert_array_equal(checkpoint["first_moment"][0], 0)
        np.testing.assert_array_equal(checkpoint["first_moment"][1], old_moments["weighted_first_moment"][3])
        np.testing.assert_array_equal(checkpoint["first_moment"][3], checkpoint["first_moment"][4])
        expected_delta = checkpoint["volume_weights"][:, None]*(checkpoint["wall_observations__unweighted_circulation"]-checkpoint["native_circulation"])
        np.testing.assert_array_equal(checkpoint["delta_circulation"][4], expected_delta)
        for state, kind in ((2, "cell_only"), (3, "wall_observations")):
            np.testing.assert_array_equal(checkpoint["first_moment"][state], checkpoint["volume_weights"][:, None, None]*checkpoint[kind+"__unweighted_moment"])
        if count > 108:
            for key in ("delta_circulation", "first_moment", "native_circulation", "volume_weights", "correction_induction"):
                np.testing.assert_array_equal(checkpoint[key], checkpoint_by_count[108][key])
        checkpoint_by_count[count] = checkpoint
        selections = {key: slice(*limits) for key, limits in report["target_slices"].items()}
        h = report["derivative_steps"][0]
        stencils = {"centred": (["plus", "minus"], [1, -1], 2*h), "centred_half": (["plus_half", "minus_half"], [1, -1], h),
                    "exterior": (["boundary", "plus", "plus2"], [-3, 4, -1], 2*h), "interior": (["boundary", "minus", "minus2"], [3, -4, 1], 2*h),
                    "exterior_half": (["boundary", "plus_half", "plus"], [-3, 4, -1], h), "interior_half": (["boundary", "minus_half", "minus"], [3, -4, 1], h)}
        for state, row in enumerate(report["records"]):
            name, velocity = row["name"], field[row["name"]+"__velocity"]
            np.testing.assert_array_equal(velocity, base["constant_volume_control__velocity"]+checkpoint["correction_induction"][state]+field[name+"__body_velocity_change"])
            if state < 2:
                control = "constant_volume_control" if state == 0 else "linear_face_moments"
                baseline_difference.append(float(np.max(np.abs(velocity-base[control+"__velocity"]))))
                assert baseline_difference[-1] < 1e-12
            for group, recorded in row["cell_velocity_rms_over_Uinf"].items():
                ids = full[group+"__cell_ids"]
                metric_differences.append(abs(rms(velocity[selections[group]]-full["native_cell_velocity"][ids], full["native_cell_volume"][ids])-recorded))
            raw = velocity[selections["boundary"]]
            flux = np.dot(np.sum(raw*boundary["normal"], axis=1), boundary["area"])
            cut = raw-(flux/boundary["area"].sum())*boundary["normal"]
            un = np.sum(cut*boundary["normal"], axis=1)
            np.testing.assert_allclose(un, field[name+"__boundary_normal_velocity"], rtol=0, atol=3e-16)
            metric_differences.append(abs(rms((un-boundary["fvm_normal_velocity"])[:, None], boundary["area"])-row["boundary_normal_velocity_rms_error_over_Uinf"]))
            for side, (keys, coefficients, denominator) in stencils.items():
                derivative = np.einsum("s,snd->nd", coefficients, np.stack([velocity[selections[key]] for key in keys]))/denominator
                derivative /= np.linalg.norm(boundary["normal"], axis=1)[:, None]
                n = boundary["native_unit_normal"]
                derivative -= np.sum(derivative*n, axis=1)[:, None]*n
                derivative_differences.append(float(np.max(np.abs(derivative-field[name+"__gradient_"+side]))))
                metric_differences.append(abs(rms(derivative-boundary["fvm_native_flux_tangential_normal_gradient"], boundary["native_vector_area"])
                                              -row["boundary_native_tangential_gradient_rms_error"][side]))
            wall = velocity[selections["wall"]]
            wn = np.sum(wall*full["wall_normal"], axis=1)
            metric_differences += [abs(rms(wn[:, None])-row["wall_normal_velocity_rms_over_Uinf"]),
                                  abs(rms(wall-wn[:, None]*full["wall_normal"])-row["wall_tangential_velocity_rms_over_Uinf"]),
                                  abs(float(base["panel_area"] @ field[name+"__panel_strength"])-row["body_source_flux"]),
                                  abs(float(np.linalg.norm(checkpoint["delta_circulation"][state], axis=1).sum())-row["circulation_change_L1"])]
            np.testing.assert_allclose(checkpoint["delta_circulation"][state].sum(axis=0), row["total_circulation_change"], rtol=0, atol=1e-15)
            assert abs(row["body_source_flux"]) < 1e-10 and max(row["derivative_step_halving_maximum_difference"].values()) < 1e-6
        physical[count] = report
    tests = ET.parse(RESULTS / "3d-boundary-quadratic-regression.xml")
    assert len(list(tests.iter("testcase"))) == 5 and not any(list(tests.iter(tag)) for tag in ("failure", "error", "skipped"))
    assert max(metric_differences) < 1e-12 and max(derivative_differences) < 1e-8
    plot(manufactured, physical)
    archive = RESULTS / "boundary-quadratic-verification-sources" / Path(__file__).relative_to(ROOT)
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, archive)
    artifacts = [Path(__file__), RESULTS / "cube-3d-boundary-quadratic-manufactured.png", RESULTS / "cube-3d-boundary-quadratic-physical.png",
                 RESULTS / "3d-boundary-quadratic-regression.xml"]
    for resolution in ("coarse", "medium"):
        artifacts += [RESULTS / f"cube-3d-native-moment-reconstruction-{resolution}/cube-native-moment-reconstruction-3d.json",
                      RESULTS / f"cube-3d-quadratic-moment-reconstruction-{resolution}/cube-quadratic-moment-reconstruction-3d.json"]
    output = {"status": "complete", "spatial_dimensions": 3, "source_records_verified": len(sources), "source_checks": sources,
              "metrics_recomputed": len(metric_differences), "maximum_metric_difference": max(metric_differences),
              "maximum_independent_derivative_replay_difference": max(derivative_differences),
              "maximum_physical_control_velocity_difference": max(baseline_difference), "new_passing_tests": 5,
              "verification_artifacts": [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in artifacts]}
    (RESULTS / "boundary-quadratic-verification.json").write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps({key: value for key, value in output.items() if key not in ("source_checks", "verification_artifacts")}, indent=2))


def plot(manufactured, physical):
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.6))
    labels = ["Linear\npoint", "Quadratic\npoint", "Quadratic + wall\npoint", "Linear\naverage", "Quadratic\naverage", "Quadratic + wall\naverage"]
    for column, width in enumerate((.35, .15)):
        for resolution, color in (("coarse", "#2374ab"), ("medium", "#c95627")):
            old = json.loads((RESULTS / f"cube-3d-native-moment-reconstruction-{resolution}/cube-native-moment-reconstruction-3d.json").read_text())
            quadratic = json.loads((RESULTS / f"cube-3d-quadratic-moment-reconstruction-{resolution}/cube-quadratic-moment-reconstruction-3d.json").read_text())
            rows = []
            for kind in ("point", "average"):
                for report, prefix in ((old, "weak_linear_faces_"), (quadratic, "weak_quadratic_faces_"), (manufactured[resolution], "weak_boundary_quadratic_faces_")):
                    rows.append(next(row for row in report["records"] if row["width"] == width and row["method"] == prefix+kind+"_input"))
            for index, group in enumerate(("near_body", "wall")):
                axes[index, column].plot(range(6), [row["velocity_error"][group]["rms_over_reference_speed"] for row in rows], "o-", color=color, label=resolution.capitalize())
        axes[0, column].set_title(f"{'Broad' if width == .35 else 'Thin'} 3D field · width = {width} D")
        for index, ax in enumerate(axes[:, column]):
            ax.set_ylabel(("Near-body velocity" if index == 0 else "Wall velocity")+" RMS error / Uref")
            ax.set_xticks(range(6), labels, fontsize=8)
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=.2)
            ax.legend()
    fig.suptitle("Using prescribed wall velocity inside the quadratic cell fit", fontsize=14)
    fig.text(.5, .012, "Identical fully 3D meshes and targets · Point and exact-average inputs are separate · No target fitting", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .94), h_pad=2)
    fig.savefig(RESULTS / "cube-3d-boundary-quadratic-manufactured.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    labels = ["Constant\nvolume", "Linear\nM", "Quadratic\nM\ncells", "Quadratic\nM\ncells + wall", "Quadratic\nΓ and M\ncells + wall"]
    titles = ["Near-body velocity", "Boundary normal velocity", "Boundary tangential derivative", "Wall normal velocity", "Wall tangential velocity", "Unused outer-layer velocity"]
    for count, color in ((108, "#2374ab"), (1728, "#c95627"), (6912, "#237d7a")):
        for metric, ax in enumerate(axes.ravel()):
            values = []
            for row in physical[count]["records"]:
                values.append([row["cell_velocity_rms_over_Uinf"]["near_body"], row["boundary_normal_velocity_rms_error_over_Uinf"],
                               row["boundary_native_tangential_gradient_rms_error"]["centred_half"], row["wall_normal_velocity_rms_over_Uinf"],
                               row["wall_tangential_velocity_rms_over_Uinf"], row["cell_velocity_rms_over_Uinf"]["held_outer"]][metric])
            ax.plot(range(5), values, "o-", color=color, label=f"{count:,} panels")
    for metric, ax in enumerate(axes.ravel()):
        ax.set_xticks(range(5), labels, fontsize=8)
        ax.set_title(titles[metric])
        ax.set_ylabel("RMS error (U∞/D)" if metric == 2 else "RMS error / U∞")
        ax.set_ylim(0, ax.get_ylim()[1]*1.07)
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle("Frozen physical cube: separate first-moment and circulation changes", fontsize=14)
    fig.text(.5, .012, "Same 2,840-cell small FVM box and exterior particles · First four states preserve native Γ · Last state changes weighted Γ", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .94), h_pad=2)
    fig.savefig(RESULTS / "cube-3d-boundary-quadratic-physical.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    run()
