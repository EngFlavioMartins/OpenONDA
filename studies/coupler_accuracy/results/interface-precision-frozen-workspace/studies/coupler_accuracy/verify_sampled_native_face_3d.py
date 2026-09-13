#!/usr/bin/env python3
"""Replay sampled-face measurements and qualify their numerical comparisons."""

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

from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "studies/coupler_accuracy/results"


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rms(value, weight=None):
    return float(np.sqrt(np.average(np.sum(np.asarray(value)**2, axis=-1), weights=weight)))


def tangent(value, normal):
    return value-np.sum(value*normal, axis=1)[:, None]*normal


def independent_gradient(mesh, geometry, source, cell_velocity):
    """Accumulate only complete gradient-cell face stencils using add.at."""
    sample_cells, gradient_cells = source["sample_cells"], source["gradient_cells"]
    sample_index = np.full(mesh["n_cells"], -1)
    sample_index[sample_cells] = np.arange(len(sample_cells))
    grad_index = np.full(mesh["n_cells"], -1)
    grad_index[gradient_cells] = np.arange(len(gradient_cells))
    faces = source["gradient_faces"]
    owner, neighbour = mesh["owners"][faces], mesh["neighbours"][faces]
    assert np.all(sample_index[owner] >= 0) and np.all(sample_index[neighbour] >= 0)
    weight = geometry["face_interpolation_weight"][faces]
    u_face = cell_velocity[sample_index[owner]]*(1-weight[:, None])+cell_velocity[sample_index[neighbour]]*weight[:, None]
    contribution = geometry["face_area_vector"][faces, :, None]*u_face[:, None, :]
    gradient = np.zeros((len(gradient_cells), 3, 3))
    for cells, sign in ((owner, 1), (neighbour, -1)):
        keep = grad_index[cells] >= 0
        np.add.at(gradient, grad_index[cells[keep]], sign*contribution[keep])
    gradient /= geometry["cell_volume"][gradient_cells, None, None]
    cut = source["full_face_ids"]
    owner, neighbour = mesh["owners"][cut], mesh["neighbours"][cut]
    sign = source["signs"]
    inward, outward = np.where(sign > 0, owner, neighbour), np.where(sign > 0, neighbour, owner)
    sf = geometry["face_area_vector"][cut]
    normal = sf*sign[:, None]/np.linalg.norm(sf, axis=1)[:, None]
    edge = geometry["cell_centre"][outward]-geometry["cell_centre"][inward]
    normal_distance = np.sum(edge*normal, axis=1)
    assert np.all(normal_distance > 0)
    w = geometry["face_interpolation_weight"][cut]
    grad_face = gradient[grad_index[owner]]*(1-w[:, None, None])+gradient[grad_index[neighbour]]*w[:, None, None]
    derivative = ((cell_velocity[sample_index[outward]]-cell_velocity[sample_index[inward]])/normal_distance[:, None]
                  +np.einsum("nd,ndc->nc", normal-edge/normal_distance[:, None], grad_face))
    face_velocity = cell_velocity[sample_index[owner]]*(1-w[:, None])+cell_velocity[sample_index[neighbour]]*w[:, None]
    return gradient, derivative, face_velocity


def run():
    directory = RESULTS / "cube-3d-sampled-native-face"
    report = json.loads((directory / "cube-sampled-native-face-3d.json").read_text())
    assert report["status"] == "complete" and report["spatial_dimensions"] == 3
    assert report["sources"] == json.loads((directory / "sources-at-start.json").read_text())
    for source in report["sources"]:
        path = ROOT / source["path"]
        archive = directory / "sources" / source["path"]
        assert digest(path) == source["sha256"]
        assert digest(archive if archive.exists() else path) == source["sha256"]
    source = arrays(directory / "sampled-face-source-induction.npz")
    field = arrays(directory / "sampled-native-face-fields.npz")
    boundary = arrays(RESULTS / "cube-3d-integrated-velocity-curl-reconstruction-boundary/boundary-fields.npz")
    truth = arrays(RESULTS / "cube-3d-native-volume-induction/volume-induction-comparison-fields.npz")
    mesh = load_native_mesh(RESULTS / "cube-3d-oracle/full-native-mesh.npz")
    geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    n = report["sample_cells"]
    assert n == len(source["sample_cells"])
    assert report["gradient_cells"] == len(source["gradient_cells"])
    np.testing.assert_array_equal(source["position"], field["position"])
    np.testing.assert_array_equal(source["position"][:n], geometry["cell_centre"][source["sample_cells"]])
    np.testing.assert_array_equal(source["position"][n:], boundary["position"])
    np.testing.assert_array_equal(source["incident_velocity"], source["baseline_induction"]+source["correction_induction"])
    assert not np.any(np.isin(mesh["owners"][mesh["n_interior_faces"]:], source["gradient_cells"]))
    normal, area, side = field["normal"], field["area"], field["face_group"]
    sample_index = np.full(mesh["n_cells"], -1)
    sample_index[source["sample_cells"]] = np.arange(n)
    grad_input = sample_index[source["gradient_cells"]]
    metric_diff, gradient_diff, velocity_replay = [], [], []

    def check(actual, claimed):
        metric_diff.append(abs(actual-claimed))
        np.testing.assert_allclose(actual, claimed, rtol=0, atol=2e-13)

    ref_g, ref_dn, ref_u = independent_gradient(mesh, geometry, source, truth["native_cell_velocity"][source["sample_cells"]])
    np.testing.assert_allclose(ref_g, field["reference_cell_gradient"], rtol=0, atol=2e-13)
    np.testing.assert_allclose(tangent(ref_dn, normal), field["reference_tangential_gradient"], rtol=0, atol=2e-13)
    np.testing.assert_allclose(ref_u, field["reference_face_velocity"], rtol=0, atol=2e-15)
    normal_ref = field["reference_normal_velocity"]
    gradient_ref = field["reference_tangential_gradient"]
    for row in report["records"]:
        count, name = row["panels"], row["name"]
        prefix = f"panel{count}__{name}"
        index = report["source_names"].index(name)
        old_dir = RESULTS / f"cube-3d-boundary-quadratic-overlap-{count}-qualified"
        old_report = json.loads((old_dir / "cube-boundary-quadratic-overlap-3d.json").read_text())
        old = arrays(old_dir / "physical-boundary-quadratic-fields.npz")
        checkpoint = arrays(old_dir / "physical-boundary-quadratic-checkpoint.npz")
        cuts = {key: slice(*limits) for key, limits in old_report["target_slices"].items()}
        u = field[prefix+"__velocity"]
        np.testing.assert_array_equal(u, source["incident_velocity"][index]+[1., 0., 0.]+field[prefix+"__body_velocity"])
        replay = float(np.max(np.abs(u[n:]-old[name+"__velocity"][cuts["boundary"]])))
        check(replay, row["boundary_velocity_replay_maximum_difference"])
        velocity_replay.append(replay)
        np.testing.assert_allclose(source["correction_induction"][index, n:], checkpoint["correction_induction"][index, cuts["boundary"]], rtol=0, atol=1e-13)
        g, dn, face_u = independent_gradient(mesh, geometry, source, u[:n])
        gradient_diff.append(float(np.max(np.abs(g-field[prefix+"__cell_gradient"]))))
        gradient_diff.append(float(np.max(np.abs(dn-field[prefix+"__normal_gradient"]))))
        np.testing.assert_allclose(g, field[prefix+"__cell_gradient"], rtol=0, atol=2e-13)
        np.testing.assert_allclose(dn, field[prefix+"__normal_gradient"], rtol=0, atol=2e-13)
        np.testing.assert_allclose(face_u, field[prefix+"__face_velocity"], rtol=0, atol=2e-15)
        # Replay the previously step-qualified centred derivative directly from
        # its two saved velocity samples, including the original normal length.
        h = old_report["derivative_steps"][0]
        continuous = (old[name+"__velocity"][cuts["plus_half"]]-old[name+"__velocity"][cuts["minus_half"]])/h
        continuous = tangent(continuous/np.linalg.norm(boundary["normal"], axis=1)[:, None], normal)
        np.testing.assert_array_equal(continuous, field[prefix+"__continuous_tangential_gradient"])
        native = field[prefix+"__tangential_gradient"]
        ec, en, delta = continuous-gradient_ref, native-gradient_ref, native-continuous
        check(rms(ec, area), row["continuous_gradient_error_rms"])
        check(rms(en, area), row["native_gradient_error_rms"])
        check(rms(delta, area), row["gradient_observation_change_rms"])
        check(float(np.average(np.sum(ec*delta, axis=1), weights=area)), row["error_dot_observation_change"])
        check(rms(field[prefix+"__value_tangential_gradient"]-native, area), row["value_vs_flux_gradient_rms"])
        cut = source["full_face_ids"]
        owner, neighbour = mesh["owners"][cut], mesh["neighbours"][cut]
        distance = np.sum(geometry["cell_connection_vector"][cut]*normal, axis=1)
        two = (u[sample_index[neighbour]]-u[sample_index[owner]])/distance[:, None]
        check(rms(tangent(field[prefix+"__normal_gradient"]-two, normal), area), row["nonorthogonal_gradient_correction_rms"])
        normal_values = {}
        for observation, velocity in (("point", u[n:]), ("native", face_u)):
            raw_flux = np.einsum("ij,ij->i", velocity, boundary["normal"]) @ boundary["area"]
            corrected = velocity-(raw_flux/boundary["area"].sum())*boundary["normal"]
            normal_values[observation] = np.sum(corrected*boundary["normal"], axis=1)
            np.testing.assert_allclose(normal_values[observation], field[prefix+"__"+observation+"_normal_velocity"], rtol=0, atol=2e-15)
            check(rms((normal_values[observation]-normal_ref)[:, None], boundary["area"]), row[observation+"_normal_velocity_error_rms"])
            check(abs(float(raw_flux)), row["boundary_flux"][observation]["raw_mismatch"])
            check(abs(float(raw_flux/boundary["area"].sum())), row["boundary_flux"][observation]["applied_correction"])
            check(abs(float(normal_values[observation] @ boundary["area"])), row["boundary_flux"][observation]["corrected_mismatch"])
        check(rms((normal_values["native"]-normal_values["point"])[:, None], boundary["area"]), row["normal_velocity_observation_change_rms"])
        check(rms(face_u-u[n:], area), row["face_velocity_observation_change_rms"])
        check(rms(u[grad_input]-truth["native_cell_velocity"][source["gradient_cells"]], geometry["cell_volume"][source["gradient_cells"]]), row["gradient_input_cell_velocity_error_rms"])
        for number, side_name in enumerate(("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")):
            group, mask = row["face_groups"][side_name], side == number
            assert group["faces"] == int(mask.sum())
            for value, key in ((ec, "continuous_gradient_error_rms"), (en, "native_gradient_error_rms"), (delta, "gradient_observation_change_rms")):
                check(rms(value[mask], area[mask]), group[key])
            for observation in ("point", "native"):
                check(rms((normal_values[observation]-normal_ref)[mask, None], boundary["area"][mask]), group[observation+"_normal_velocity_error_rms"])
    check(max(velocity_replay), report["boundary_velocity_replay_maximum_difference"])
    test_path = RESULTS / "3d-face-velocity-sampling-regression.xml"
    tests = ET.parse(test_path).getroot().findall(".//testcase")
    assert len(tests) == 4 and not any(test.find(tag) is not None for test in tests for tag in ("failure", "error", "skipped"))

    finest = [row for row in report["records"] if row["panels"] == max(report["body_counts"])]
    labels = ["Constant\nvolume", "Linear-face M\nnative Γ", "Quadratic cell M\nnative Γ", "Quadratic wall M\nnative Γ", "Quadratic wall\nΓ and M"]
    x = np.arange(len(finest))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), constrained_layout=True)
    for ax, point, native, label in ((axes[0], "continuous_gradient_error_rms", "native_gradient_error_rms", "Tangential derivative RMS error [U∞/D]"),
                                   (axes[1], "point_normal_velocity_error_rms", "native_normal_velocity_error_rms", "Normal velocity RMS error / U∞")):
        ax.bar(x-.18, [row[point] for row in finest], .36, label="Continuous derivative / point velocity", color="#527a9c")
        ax.bar(x+.18, [row[native] for row in finest], .36, label="Native FVM observation of sampled velocity", color="#cb7544")
        ax.set_xticks(x, labels)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.set_ylim(0, ax.get_ylim()[1]*1.25)
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Same frozen 3D velocity, different boundary measurements\n6,912 body panels; unchanged small FVM domain and source field", fontsize=14)
    figure = RESULTS / "cube-3d-sampled-native-face.png"
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    result = {"status": "complete", "source_records_checked": len(report["sources"]), "metrics_recomputed": len(metric_diff),
              "maximum_metric_difference": max(metric_diff), "independent_gauss_and_flux_replay_maximum_difference": max(gradient_diff),
              "previous_boundary_velocity_replay_maximum_difference": max(velocity_replay), "new_tests_passed": len(tests),
              "physical_records": len(report["records"]), "figure": {"path": str(figure.relative_to(ROOT)), "sha256": digest(figure)},
              "tests": {"path": str(test_path.relative_to(ROOT)), "sha256": digest(test_path)}}
    archive = RESULTS / "sampled-native-face-verification-sources" / Path(__file__).relative_to(ROOT)
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, archive)
    result["verifier"] = {"path": str(Path(__file__).relative_to(ROOT)), "sha256": digest(Path(__file__))}
    (RESULTS / "sampled-native-face-verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
