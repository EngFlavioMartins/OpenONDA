#!/usr/bin/env python3
"""Verify native moment acquisition and its frozen physical overlap test."""

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


def run():
    metric_difference, source_checks, native_reports = [], [], {}

    def verify_sources(directory, report):
        assert report["status"] == "complete" and report["spatial_dimensions"] == 3
        assert report["sources"] == json.loads((directory / "sources-at-start.json").read_text())
        for row in report["sources"]:
            path = ROOT / row["path"]
            archive = directory / "sources" / row["path"]
            assert digest(archive if archive.exists() else path) == row["sha256"]
            assert digest(path) == row["sha256"], f"Post-run source change: {path}"
            source_checks.append({"result_directory": directory.name, **row})

    for resolution in ("coarse", "medium"):
        directory = RESULTS / f"cube-3d-native-moment-reconstruction-{resolution}"
        report = json.loads((directory / "cube-native-moment-reconstruction-3d.json").read_text())
        verify_sources(directory, report)
        source = arrays(directory / "reconstructed-moment-source-fields.npz")
        field = arrays(directory / "reconstructed-moment-induction-fields.npz")
        old_dir = RESULTS / f"cube-3d-manufactured-induction-{resolution}"
        parent = json.loads((old_dir / "cube-manufactured-induction-3d.json").read_text())
        old = arrays(old_dir / "manufactured-source-fields.npz")
        old_field = arrays(old_dir / "manufactured-induction-fields.npz")
        exact_dir = RESULTS / f"cube-3d-manufactured-linear-induction-{resolution}"
        exact = arrays(exact_dir / "linear-source-fields.npz")
        for key in ("position", "exact_velocity"):
            np.testing.assert_array_equal(field[key], old_field[key])
        for key in ("volume", "centroid", "covariance"):
            np.testing.assert_array_equal(source[key], exact[key])
        np.testing.assert_array_equal(source["source_names"], report["source_names"])
        widths = [row["width"] for row in parent["manufactured_fields"]]
        radius = np.max(np.abs(source["centres"]), axis=1)
        regions = {"all": np.ones(len(radius), dtype=bool), "near_body": radius < .8,
                   "outer_layer": (radius > 1.1) & (radius < 1.5)}
        for state, row in enumerate(report["records"]):
            index = widths.index(row["width"])
            gamma, moment = source["circulation"][state], source["first_moment"][state]
            reference_gamma = old["vorticity_integral"][index]
            gradient = np.linalg.solve(source["covariance"], moment)
            exact_gradient = np.linalg.solve(source["covariance"], exact["first_moment"][index])
            difference = gradient-exact_gradient
            squared = np.einsum("nij,nik,nkj->n", difference, source["covariance"], difference)
            norm2 = np.einsum("nij,nik,nkj->n", exact_gradient, source["covariance"], exact_gradient)
            for region, selected in regions.items():
                volume = source["volume"][selected]
                error = rms((gamma-reference_gamma)[selected]/volume[:, None], volume)
                norm = rms(reference_gamma[selected]/volume[:, None], volume)
                recomputed = {"circulation_density_rms": error, "circulation_relative_error": error/norm,
                              "moment_variation_density_rms": float(np.sqrt(squared[selected].sum()/volume.sum())),
                              "moment_variation_relative_error": float(np.sqrt(squared[selected].sum()/norm2[selected].sum()))}
                metric_difference += [abs(value-row["source_error"][region][key]) for key, value in recomputed.items()]
            np.testing.assert_allclose(gamma.sum(axis=0), row["total_circulation"], rtol=0, atol=1e-14)
            global_moment = moment.sum(axis=0)+np.einsum("ni,nj->ij", source["centroid"], gamma)
            np.testing.assert_allclose(global_moment, row["global_first_vorticity_moment"], rtol=0, atol=1e-14)
            if row["method"].startswith("weak_"):
                key = "true_cell_velocity_integral" if row["method"].endswith("average_input") else "estimated_cell_velocity_integral"
                expected = -np.cross(np.eye(3), source[key][index].sum(axis=0))
                np.testing.assert_allclose(global_moment, expected, rtol=0, atol=1e-11)
            for group, limits in report["target_slices"].items():
                selected = slice(*limits)
                difference = field["velocity"][state, selected]-field["exact_velocity"][index, selected]
                metric_difference.extend((abs(rms(difference)-row["velocity_error"][group]["rms_over_reference_speed"]),
                                          abs(np.linalg.norm(difference, axis=1).max()-row["velocity_error"][group]["maximum_over_reference_speed"])))
        native_reports[resolution] = report
    directory = RESULTS / "cube-3d-reconstructed-moment-overlap"
    physical = json.loads((directory / "cube-reconstructed-moment-overlap-3d.json").read_text())
    verify_sources(directory, physical)
    fields = arrays(directory / "reconstructed-moment-overlap-fields.npz")
    checkpoint = arrays(directory / "moment-overlap-induction-checkpoint.npz")
    previous = arrays(RESULTS / "cube-3d-volume-overlap/volume-overlap-fields.npz")
    full = arrays(RESULTS / "cube-3d-native-volume-induction/volume-induction-comparison-fields.npz")
    omega = arrays(RESULTS / "cube-3d-native-volume-induction/native-induction-fields.npz")["cell_vorticity"]
    boundary = arrays(RESULTS / "cube-3d-integrated-velocity-curl-reconstruction-boundary/boundary-fields.npz")
    np.testing.assert_array_equal(fields["position"], previous["position"])
    ids = checkpoint["inside_cell_ids"]
    np.testing.assert_array_equal(checkpoint["native_circulation"], omega[ids]*full["native_cell_volume"][ids, None])
    np.testing.assert_array_equal(checkpoint["moment_induction"][0], 0)
    baseline = fields["constant_volume_control__velocity"]
    baseline_difference = float(np.max(np.abs(baseline-previous["taper_0.75_to_1.25__velocity"])))
    assert baseline_difference < 1e-12
    selections = {key: slice(*limits) for key, limits in physical["target_slices"].items()}
    epsilon = physical["derivative_steps"][0]
    stencils = {"centred": (["plus", "minus"], [1, -1], 2*epsilon),
                "centred_half": (["plus_half", "minus_half"], [1, -1], epsilon),
                "exterior": (["boundary", "plus", "plus2"], [-3, 4, -1], 2*epsilon),
                "interior": (["boundary", "minus", "minus2"], [3, -4, 1], 2*epsilon),
                "exterior_half": (["boundary", "plus_half", "plus"], [-3, 4, -1], epsilon),
                "interior_half": (["boundary", "minus_half", "minus"], [3, -4, 1], epsilon)}
    derivative_differences = []
    for row in physical["records"]:
        name = row["name"]
        velocity = fields[name+"__velocity"]
        for group, recorded in row["cell_velocity_rms_over_Uinf"].items():
            selected = full[group+"__cell_ids"]
            computed = rms(velocity[selections[group]]-full["native_cell_velocity"][selected], full["native_cell_volume"][selected])
            metric_difference.append(abs(computed-recorded))
        un = fields[name+"__boundary_normal_velocity"]
        metric_difference.append(abs(rms((un-boundary["fvm_normal_velocity"])[:, None], boundary["area"])-row["boundary_normal_velocity_rms_error_over_Uinf"]))
        for side, (keys, coefficient, denominator) in stencils.items():
            derivative = np.einsum("s,snd->nd", coefficient, np.stack([velocity[selections[key]] for key in keys]))/denominator
            derivative /= np.linalg.norm(boundary["normal"], axis=1)[:, None]
            normal = boundary["native_unit_normal"]
            derivative -= np.sum(derivative*normal, axis=1)[:, None]*normal
            derivative_differences.append(float(np.max(np.abs(derivative-fields[name+"__gradient_"+side]))))
            metric_difference.append(abs(rms(fields[name+"__gradient_"+side]-boundary["fvm_native_flux_tangential_normal_gradient"],
                                               boundary["native_vector_area"])-row["boundary_native_tangential_gradient_rms_error"][side]))
        assert max(row["derivative_step_halving_maximum_difference"].values()) < 1e-6
    assert max(metric_difference) < 1e-12 and max(derivative_differences) < 1e-8
    test_counts = {}
    for filename, count in (("3d-manufactured-regression.xml", 44), ("3d-linear-volume-regression.xml", 7),
                            ("3d-moment-reconstruction-regression.xml", 5), ("3d-induction-derivatives-regression.xml", 1)):
        tree = ET.parse(RESULTS / filename)
        assert len(list(tree.iter("testcase"))) == count
        assert not any(list(tree.iter(tag)) for tag in ("failure", "error", "skipped"))
        test_counts[filename] = count
    invalid = RESULTS / "cube-3d-reconstructed-moment-overlap-initial-attempt/validation-status.json"
    assert json.loads(invalid.read_text())["status"] == "invalid_derivative_measurement"
    plot(native_reports, physical)
    artifacts = [Path(__file__), RESULTS / "cube-3d-native-moment-reconstruction.png", RESULTS / "cube-3d-reconstructed-moment-overlap.png", invalid]
    artifacts += [RESULTS / filename for filename in test_counts]
    destination = RESULTS / "reconstructed-moments-verification-sources" / Path(__file__).relative_to(ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, destination)
    output = {"status": "complete", "spatial_dimensions": 3, "source_records_verified": len(source_checks), "source_checks": source_checks,
              "metrics_recomputed": len(metric_difference), "maximum_metric_difference": max(metric_difference),
              "baseline_velocity_maximum_difference_from_previous_overlap": baseline_difference,
              "maximum_derivative_difference_from_independent_stencil_evaluation": max(derivative_differences),
              "passing_tests": test_counts, "total_passing_tests": sum(test_counts.values()),
              "physical_source_cells": physical["small_fvm_cells"], "exterior_particles": physical["exterior_particles"],
              "discarded_attempt": str(invalid.parent.relative_to(ROOT)),
              "verification_artifacts": [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in artifacts]}
    (RESULTS / "reconstructed-moments-verification.json").write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps({k: v for k, v in output.items() if k not in ("source_checks", "verification_artifacts")}, indent=2))


def plot(reports, physical):
    methods = ["lsq_exact_circulation", "lsq_native_curl", "weak_native_faces", "weak_linear_faces_point_input", "weak_linear_faces_average_input"]
    labels = ["LSQ\nexact Γ", "LSQ\nnative Γ", "Weak\nnative faces", "Linear faces\npoint input", "Linear faces\naverage input"]
    colors = {"coarse": "#2374ab", "medium": "#c95627"}
    fig, axes = plt.subplots(2, 2, figsize=(12.7, 7.4))
    for column, width in enumerate((.35, .15)):
        for resolution, report in reports.items():
            rows = [next(r for r in report["records"] if r["method"] == method and r["width"] == width) for method in methods]
            axes[0, column].plot(range(5), [100*r["source_error"]["near_body"]["moment_variation_relative_error"] for r in rows],
                                 "o-", color=colors[resolution], label=resolution.capitalize())
            axes[1, column].plot(range(5), [r["velocity_error"]["near_body"]["rms_over_reference_speed"] for r in rows],
                                 "o-", color=colors[resolution], label=resolution.capitalize())
        axes[0, column].set_title(f"{'Broad' if width == .35 else 'Thin'} 3D field · width = {width} D")
        axes[0, column].set_ylabel("Moment-induced density variation error (%)")
        axes[1, column].set_ylabel("Near-body velocity RMS error / Uref")
        for ax in axes[:, column]:
            ax.set_xticks(range(5), labels, fontsize=8)
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=.2)
    fig.suptitle("Recovering first moments from neighbouring circulation or native velocity", fontsize=14)
    fig.text(.5, .012, "All methods use the same 3D meshes and targets; no method receives exact first moments or target velocities", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .94), h_pad=2)
    fig.savefig(RESULTS / "cube-3d-native-moment-reconstruction.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4))
    labels = ["Constant\nvolume", "Circulation\nLSQ moments", "Native-face\nmoments", "Linear-face\nmoments"]
    for ax, key, title, unit in zip(axes.ravel(),
                                   ("near_body", "held_outer", "boundary_normal", "boundary_gradient", "wall_normal", "wall_tangent"),
                                   ("Near-body velocity", "Unused outer-layer velocity", "Boundary normal velocity", "Boundary tangential derivative", "Wall normal velocity", "Wall tangential velocity"),
                                   ("U∞", "U∞", "U∞", "U∞/D", "U∞", "U∞"), strict=True):
        values = []
        for row in physical["records"]:
            mapping = {"near_body": row["cell_velocity_rms_over_Uinf"]["near_body"], "held_outer": row["cell_velocity_rms_over_Uinf"]["held_outer"],
                       "boundary_normal": row["boundary_normal_velocity_rms_error_over_Uinf"],
                       "boundary_gradient": row["boundary_native_tangential_gradient_rms_error"]["centred_half"],
                       "wall_normal": row["wall_normal_velocity_rms_over_Uinf"], "wall_tangent": row["wall_tangential_velocity_rms_over_Uinf"]}
            values.append(mapping[key])
        ax.bar(range(4), values, color=["#718397", "#bf9445", "#558aaa", "#237d7a"])
        ax.set_xticks(range(4), labels, fontsize=8)
        ax.set_title(title)
        ax.set_ylabel(f"RMS error ({unit})")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Frozen physical cube: better near-body velocity does not improve every boundary component", fontsize=14)
    fig.text(.5, .012, "Same 2,840-cell small FVM domain, native circulations, Gaussian complements, 68 exterior particles and 108 body panels", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .94), h_pad=2)
    fig.savefig(RESULTS / "cube-3d-reconstructed-moment-overlap.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    run()
