#!/usr/bin/env python3
"""Verify the exact-moment control against the qualified manufactured baseline."""

from __future__ import annotations

import ast
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


def read_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    comparisons, checks, source_checks, changed = [], [], [], []
    for resolution in ("coarse", "medium"):
        directory = RESULTS / f"cube-3d-manufactured-linear-induction-{resolution}"
        parent_dir = RESULTS / f"cube-3d-manufactured-induction-{resolution}"
        report = json.loads((directory / "cube-manufactured-linear-induction-3d.json").read_text())
        parent = json.loads((parent_dir / "cube-manufactured-induction-3d.json").read_text())
        assert report["status"] == "complete" and report["spatial_dimensions"] == 3
        assert report["sources"] == json.loads((directory / "sources-at-start.json").read_text())
        for record in report["sources"]:
            path = ROOT / record["path"]
            archive = directory / "sources" / record["path"]
            checked = archive if archive.exists() else path
            assert digest(checked) == record["sha256"], checked
            source_checks.append({"resolution": resolution, **record})
            if digest(path) != record["sha256"]:
                same_ast = ast.dump(ast.parse(path.read_text())) == ast.dump(ast.parse(archive.read_text()))
                assert same_ast, f"A post-run source change requires qualification: {path}"
                changed.append({"resolution": resolution, **record, "current_sha256": digest(path),
                                "same_python_syntax_tree": same_ast})
        source = read_arrays(directory / "linear-source-fields.npz")
        target = read_arrays(directory / "linear-induction-fields.npz")
        old_source = read_arrays(parent_dir / "manufactured-source-fields.npz")
        old_target = read_arrays(parent_dir / "manufactured-induction-fields.npz")
        for key in ("position", "exact_velocity"):
            np.testing.assert_array_equal(target[key], old_target[key])
        np.testing.assert_array_equal(source["circulation"], old_source["vorticity_integral"])
        np.testing.assert_array_equal(source["volume"], old_source["polyhedron_volume"])
        np.testing.assert_array_equal(source["centroid"], old_source["polyhedron_centroid"])
        reconstructed = source["covariance"][None] @ source["gradient"]
        recovery = float(np.max(np.abs(reconstructed-source["first_moment"])/source["volume"][None, :, None, None]**(4/3)))
        assert recovery < 1e-10
        assert np.all(np.isfinite(target["velocity"]))
        assert report["target_slices"] == parent["target_slices"]
        first = np.einsum("ni,snj->sij", source["centroid"], source["circulation"])
        global_moment = first+source["first_moment"].sum(axis=1)
        native_gamma_diagnostic = []
        for index, row in enumerate(report["records"]):
            exact_index = parent["source_names"].index(f"width_{row['width']}__exact_cell_circulation")
            native_index = parent["source_names"].index(f"width_{row['width']}__native_curl_of_point_velocity")
            # P1 is linear in Gamma and M. Changing only Gamma changes the
            # constant-volume term, which was already evaluated independently.
            diagnostic = (target["velocity"][index]+old_target["native_volume__velocity"][native_index]
                          - old_target["native_volume__velocity"][exact_index])
            native_gamma_diagnostic.append(diagnostic)
            result = {"resolution": resolution, "width": row["width"], "groups": {},
                      "global_circulation": source["circulation"][index].sum(axis=0).tolist(),
                      "constant_cell_global_first_moment": first[index].tolist(),
                      "linear_cell_global_first_moment": global_moment[index].tolist()}
            for group, limits in report["target_slices"].items():
                rows = slice(*limits)
                difference = target["velocity"][index, rows]-target["exact_velocity"][index, rows]
                norm = np.linalg.norm(difference, axis=1)
                rms = float(np.sqrt(np.mean(norm**2)))
                checks += [abs(rms-row["velocity_errors"][group]["rms_over_reference_speed"]),
                           abs(float(norm.max())-row["velocity_errors"][group]["maximum_over_reference_speed"])]
                volume = next(r for r in parent["records"] if r["width"] == row["width"] and r["kernel"] == "native_volume"
                              and r["source"] == "exact_cell_circulation")["velocity_errors"][group]["rms_over_reference_speed"]
                gaussian = next(r for r in parent["records"] if r["width"] == row["width"] and r["kernel"] == "gaussian_nominal_spacing"
                                and r["source"] == "exact_cell_circulation")["velocity_errors"][group]["rms_over_reference_speed"]
                result["groups"][group] = {"linear_rms": rms, "constant_volume_rms": volume, "gaussian_rms": gaussian,
                                           "native_curl_gamma_with_exact_first_moment_rms": float(np.sqrt(np.mean(np.sum(
                                               (diagnostic[rows]-target["exact_velocity"][index, rows])**2, axis=1)))),
                                           "linear_relative_improvement_over_constant_percent": 100*(1-rms/volume),
                                           "worst_sample_position": target["position"][rows][int(norm.argmax())].tolist(),
                                           "linear_maximum_error": float(norm.max())}
            comparisons.append(result)
        np.savez_compressed(RESULTS / f"manufactured-linear-native-gamma-control-{resolution}.npz", position=target["position"],
                            velocity=np.stack(native_gamma_diagnostic), exact_velocity=target["exact_velocity"])
    assert max(checks) < 1e-14
    test_counts = {}
    for name, expected in (("3d-manufactured-regression.xml", 44), ("3d-linear-volume-regression.xml", 7)):
        tree = ET.parse(RESULTS / name)
        tests = list(tree.iter("testcase"))
        assert len(tests) == expected and not list(tree.iter("failure")) and not list(tree.iter("error")) and not list(tree.iter("skipped"))
        test_counts[name] = len(tests)
    plot(comparisons)
    artifacts = [Path(__file__), ROOT / "studies/coupler_accuracy/compare_manufactured_induction_3d.py",
                 RESULTS / "manufactured-induction-verification.json", RESULTS / "cube-3d-manufactured-induction.png",
                 RESULTS / "cube-3d-manufactured-error-components.png", RESULTS / "cube-3d-manufactured-linear-induction.png"]
    artifacts += [RESULTS / name for name in test_counts]
    artifacts += [RESULTS / f"manufactured-linear-native-gamma-control-{resolution}.npz" for resolution in ("coarse", "medium")]
    for path in artifacts:
        if path.suffix == ".py":
            archive = RESULTS / "manufactured-linear-verification-sources" / path.relative_to(ROOT)
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, archive)
    output = {"status": "complete", "spatial_dimensions": 3, "source_records_verified": len(source_checks),
              "source_checks": source_checks, "current_sources_differing_from_archive": changed,
              "same_circulations_meshes_and_targets_as_parent": True, "metrics_recomputed": len(checks),
              "native_gamma_control": "P1(native point-input curl Gamma, exact M) is obtained by adding the already evaluated constant-volume induction of the Gamma difference to P1(exact Gamma, exact M). This retains oracle first moments and is not an available physical export.",
              "maximum_metric_difference": max(checks), "passing_tests": test_counts,
              "comparisons": comparisons,
              "verification_artifacts": [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in artifacts]}
    (RESULTS / "manufactured-linear-induction-verification.json").write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps({k: v for k, v in output.items() if k not in ("source_checks", "verification_artifacts")}, indent=2))


def plot(comparisons):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    representations = [("constant_volume_rms", "Constant native volume", "#718397"),
                       ("gaussian_rms", "Gaussian σ = h", "#d39a42"),
                       ("linear_rms", "Affine native volume", "#237d7a")]
    for row, width in enumerate((.35, .15)):
        for column, group in enumerate(("near_body", "outer_layer", "wall")):
            ax = axes[row, column]
            for offset, (key, label, color) in zip((-.25, 0, .25), representations, strict=True):
                values = [next(r for r in comparisons if r["width"] == width and r["resolution"] == resolution)["groups"][group][key]
                          for resolution in ("coarse", "medium")]
                ax.bar(np.arange(2)+offset, values, width=.23, color=color, label=label)
            ax.set_xticks([0, 1], ["Coarse", "Medium"])
            ax.set_yscale("log")
            ax.set_title(f"{'Broad' if width == .35 else 'Thin'} field · {group.replace('_', ' ')}")
            ax.set_ylabel("Velocity RMS error / Uref")
            ax.grid(axis="y", alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .94), ncol=3, frameon=False)
    fig.suptitle("Preserving first moments reduces induction error with the same cell circulations", fontsize=14)
    fig.text(.5, .012, "Exact-moment 3D control · all source cells and target points unchanged · no time advance or fitting", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .035, 1, .88), h_pad=2)
    fig.savefig(RESULTS / "cube-3d-manufactured-linear-induction.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    run()
