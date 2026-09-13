#!/usr/bin/env python3
"""Verify saved 3D manufactured experiments and decompose their vector errors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "studies/coupler_accuracy/results"


def read_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def rms(value, weight=None):
    return float(np.sqrt(np.average(np.sum(value**2, axis=-1), weights=weight)))


def run():
    reports, decompositions, metric_differences, sources, changed = {}, [], [], [], []
    common = None
    for resolution in ("coarse", "medium"):
        directory = RESULTS / f"cube-3d-manufactured-induction-{resolution}"
        report = json.loads((directory / "cube-manufactured-induction-3d.json").read_text())
        assert report["status"] == "complete" and report["spatial_dimensions"] == 3
        assert report["sources"] == json.loads((directory / "sources-at-start.json").read_text())
        for record in report["sources"]:
            path = ROOT / record["path"]
            archive = directory / "sources" / record["path"]
            checked = archive if archive.exists() else path
            assert hashlib.sha256(checked.read_bytes()).hexdigest() == record["sha256"], checked
            sources.append({"resolution": resolution, **record})
            current = hashlib.sha256(path.read_bytes()).hexdigest()
            if current != record["sha256"]:
                changed.append({"resolution": resolution, **record, "current_sha256": current})
        field = read_arrays(directory / "manufactured-induction-fields.npz")
        source = read_arrays(directory / "manufactured-source-fields.npz")
        checkpoint = read_arrays(directory / "native-velocity-checkpoint.npz")
        np.testing.assert_array_equal(source["source_names"], report["source_names"])
        np.testing.assert_array_equal(checkpoint["position"], field["position"])
        np.testing.assert_array_equal(checkpoint["velocity"], field["native_volume__velocity"])
        if common is None:
            common = field
        else:
            for name in ("position", "exact_velocity", "wall_normal"):
                np.testing.assert_array_equal(common[name], field[name])
        slices = {k: slice(*v) for k, v in report["target_slices"].items()}
        assert np.all(np.isfinite(source["gamma"])) and np.all(source["polyhedron_volume"] > 0)
        widths = [f["width"] for f in report["manufactured_fields"]]
        radius = np.max(np.abs(source["centres"]), axis=1)
        groups = {"all": np.ones(len(radius), dtype=bool), "near_body": radius < .8,
                  "outer_layer": (radius > 1.1) & (radius < 1.5)}
        for index, row in enumerate(report["source_records"]):
            reference = source["vorticity_integral"][widths.index(row["width"])]
            gamma = source["gamma"][index]
            np.testing.assert_allclose(gamma.sum(axis=0), row["total_gamma"], rtol=0, atol=1e-14)
            metric_differences.append(abs(np.linalg.norm(gamma, axis=1).sum()-row["l1_gamma"]))
            for group, selected in groups.items():
                volume = source["polyhedron_volume"][selected]
                error = rms((gamma-reference)[selected]/volume[:, None], volume)
                norm = rms(reference[selected]/volume[:, None], volume)
                recorded = row["circulation_error"][group]
                metric_differences.extend((abs(error-recorded["density_rms_error"]),
                                           abs(error/norm-recorded["relative_to_exact_cell_circulation"])))
        for row in report["records"]:
            index = report["source_names"].index(row["name"])
            velocity = field[row["kernel"]+"__velocity"][index]
            expected = field["exact_velocity"][widths.index(row["width"])]
            assert np.all(np.isfinite(velocity))
            for group, selected in slices.items():
                difference = velocity[selected]-expected[selected]
                recomputed = {"rms_over_reference_speed": rms(difference), "exact_velocity_rms": rms(expected[selected]),
                              "maximum_over_reference_speed": float(np.linalg.norm(difference, axis=1).max())}
                metric_differences += [abs(value-row["velocity_errors"][group][key]) for key, value in recomputed.items()]
        for width in widths:
            def volume(name, report=report, field=field, width=width):
                i = report["source_names"].index(f"width_{width}__{name}")
                return field["native_volume__velocity"][i]

            expected = field["exact_velocity"][widths.index(width)]
            exact = volume("exact_cell_circulation")
            midpoint = volume("exact_face_centre_velocity")
            for native_input in ("native_curl_of_point_velocity", "native_curl_of_cell_average_velocity"):
                native = volume(native_input)
                index = report["source_names"].index(f"width_{width}__{native_input}")
                for kernel in ("gaussian_nominal_spacing", "gaussian_cell_volume_scale"):
                    gaussian = field[kernel+"__velocity"][index]
                    errors = np.stack((exact-expected, midpoint-exact, native-midpoint, gaussian-native))
                    identity_error = float(np.max(np.abs(errors.sum(axis=0)-(gaussian-expected))))
                    assert identity_error < 1e-14
                    for group, selected in slices.items():
                        e = errors[:, selected]
                        gram = np.einsum("ind,jnd->ij", e, e)/e.shape[1]
                        total_squared = rms((gaussian-expected)[selected])**2
                        assert abs(gram.sum()-total_squared) < 1e-14
                        decompositions.append({"resolution": resolution, "width": width, "native_input": native_input,
                                               "kernel": kernel, "target_group": group,
                                               "component_names": ["constant_cell_source", "face_centre_quadrature",
                                                                   "native_face_interpolation", "gaussian_replacement"],
                                               "component_rms": np.sqrt(np.diag(gram)).tolist(),
                                               "component_gram_matrix": gram.tolist(), "total_rms": np.sqrt(total_squared),
                                               "sum_of_individual_squared_norms": float(np.trace(gram)),
                                               "sum_of_cross_terms": float(gram.sum()-np.trace(gram)),
                                               "vector_identity_maximum_difference": identity_error})
        reports[resolution] = report
    assert max(metric_differences) < 1e-13
    summary = {"status": "complete", "spatial_dimensions": 3, "source_records_verified": len(sources),
               "source_checks": sources, "current_sources_differing_from_archive": changed,
               "metrics_recomputed": len(metric_differences), "maximum_metric_difference": max(metric_differences),
               "same_targets_and_exact_velocity_across_resolutions": True, "decompositions": decompositions,
               "interpretation": "Components telescope as vectors. Their squared norms are coupled by the recorded Gram cross terms; RMS magnitudes are not additive fractions of cause."}
    (RESULTS / "manufactured-induction-verification.json").write_text(json.dumps(summary, indent=2)+"\n")
    plot(reports, decompositions)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("source_checks", "decompositions")}, indent=2))
    for row in decompositions:
        if row["target_group"] == "near_body" and row["native_input"] == "native_curl_of_point_velocity" and row["kernel"] == "gaussian_nominal_spacing":
            print(json.dumps(row))


def plot(reports, decompositions):
    names = ["exact_cell_circulation", "point_vorticity_times_fvm_volume", "exact_face_centre_velocity",
             "exact_face_average_velocity", "native_curl_of_point_velocity", "native_curl_of_cell_average_velocity"]
    labels = ["Exact cell\nΓ", "Point ω\n× FVM V", "Exact face\ncentre u", "Exact face\naverage u", "FVM curl\npoint u", "FVM curl\naverage u"]
    colors = {"coarse": "#2374ab", "medium": "#c95627"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2))
    for column, width in enumerate((.35, .15)):
        ax = axes[0, column]
        for resolution, report in reports.items():
            values = [next(r for r in report["source_records"] if r["width"] == width and r["source"] == name)
                      ["circulation_error"]["near_body"]["relative_to_exact_cell_circulation"] for name in names]
            ax.plot(np.arange(1, 6), np.array(values[1:])*100, "o-", color=colors[resolution], label=resolution.capitalize())
        ax.set_yscale("log")
        ax.set_title(f"{'Broad' if width == .35 else 'Thin'} 3D field, width = {width} D")
        ax.set_ylabel("Cell-circulation relative error (%)")
        ax.set_xticks(np.arange(1, 6), labels[1:], fontsize=8)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=.2)
        ax = axes[1, column]
        for resolution, report in reports.items():
            for kernel, style, title in (("native_volume", "o-", "Native volume"),
                                          ("gaussian_nominal_spacing", "s--", "Gaussian σ = h")):
                values = [next(r for r in report["records"] if r["width"] == width and r["source"] == name and r["kernel"] == kernel)
                          ["velocity_errors"]["near_body"]["rms_over_reference_speed"] for name in names]
                ax.plot(range(6), values, style, color=colors[resolution], markersize=4, label=f"{resolution.capitalize()} · {title}")
        ax.set_xticks(range(6), labels, fontsize=8)
        ax.set_ylabel("Near-body velocity RMS error / Uref")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Exact circulation alone does not recover the within-cell vorticity distribution", fontsize=14)
    fig.text(.5, .015, "Kinematic 3D verification · same 256 off-grid near-body targets · no time advance or fitted strengths", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .045, 1, .95), h_pad=2)
    fig.savefig(RESULTS / "cube-3d-manufactured-induction.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    for ax, width in zip(axes, (.35, .15), strict=True):
        for offset, resolution in ((-.18, "coarse"), (.18, "medium")):
            row = next(r for r in decompositions if r["width"] == width and r["resolution"] == resolution
                       and r["target_group"] == "near_body" and r["native_input"] == "native_curl_of_point_velocity"
                       and r["kernel"] == "gaussian_nominal_spacing")
            ax.bar(np.arange(5)+offset, row["component_rms"]+[row["total_rms"]], width=.34,
                   color=colors[resolution], label=resolution.capitalize())
        ax.set_xticks(range(5), ["Constant\ncell", "Face\nquadrature", "Native\ninterpolation", "Gaussian\nreplacement", "Total"], fontsize=9)
        ax.set_title(f"{'Broad' if width == .35 else 'Thin'} field, width = {width} D")
        ax.set_ylabel("Near-body vector error RMS / Uref")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Errors combine as vectors; these component magnitudes must not be added", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(RESULTS / "cube-3d-manufactured-error-components.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    run()
