#!/usr/bin/env python3
"""Quantify the normal-trace change needed to preserve saved cell velocities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from source.solvers.fvm.io.mesh_storage import load_native_mesh
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays
from studies.coupler_accuracy.native_normal_moment_lift_3d import NormalMomentLift
from studies.coupler_accuracy.native_volume_induction_3d import NativeVolumeSources


def independent_surface_observation(mesh, native, centre, gradient):
    """Direct triangle quadrature independent of covariance/eigenmode formulas."""
    nf = mesh["n_faces"]
    flux, first, energy, maximum = np.zeros(nf), np.zeros((nf, 3)), np.zeros(nf), np.zeros(nf)
    barycentric = (np.ones((3, 3)) + 3 * np.eye(3)) / 6
    for start in range(0, len(native.triangles), 32768):
        triangles = native.triangles[start:start + 32768]
        ids = native.face_ids[start:start + 32768]
        relative_vertices = triangles - centre[ids, None]
        relative = np.einsum("qv,tvi->tqi", barycentric, relative_vertices)
        sf = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]) / 2
        area = np.linalg.norm(sf, axis=1)
        value = np.einsum("tqi,ti->tq", relative, gradient[ids])
        np.add.at(flux, ids, area * value.mean(axis=1))
        np.add.at(first, ids, area[:, None] * (relative * value[:, :, None]).mean(axis=1))
        np.add.at(energy, ids, area * np.mean(value**2, axis=1))
        vertex_value = np.einsum("tvi,ti->tv", relative_vertices, gradient[ids])
        np.maximum.at(maximum, ids, np.max(np.abs(vertex_value), axis=1))
    cell = np.column_stack([np.bincount(mesh["owners"], weights=first[:, j], minlength=mesh["n_cells"])
                            - np.bincount(mesh["neighbours"], weights=first[:mesh["n_interior_faces"], j], minlength=mesh["n_cells"])
                            for j in range(3)])
    return flux, first, energy, maximum, cell


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    audit_path = args.audit / "flux-moment-compatibility-3d.json"
    audit = json.loads(audit_path.read_text())
    verification = json.loads(args.verification.read_text())
    assert audit["status"] == verification["status"] == "complete"
    assert audit["spatial_dimensions"] == verification["spatial_dimensions"] == 3
    assert hash_file(audit_path) in verification["sources"]
    row = next(r for r in audit["records"] if r["name"] == args.name and r["fvm_step"] == args.step)
    assert args.step > 0 and not row["reset_initial_state"]
    field_path = ROOT / row["fields"]["path"]
    assert hash_file(field_path) == row["fields"]
    fields = read_arrays(field_path)
    mesh_path = ROOT / next(r["path"] for r in audit["sources"] if r["path"].endswith(f"trial/{args.name}/solution/mesh.npz"))
    mesh = load_native_mesh(mesh_path)
    native = NativeVolumeSources.from_mesh(mesh)
    paths = [Path(__file__).resolve(), audit_path, args.verification, field_path, mesh_path]
    paths += [ROOT / name for name in (
        "studies/coupler_accuracy/native_normal_moment_lift_3d.py", "studies/coupler_accuracy/native_flux_moments_3d.py",
        "studies/coupler_accuracy/native_volume_induction_3d.py", "studies/coupler_accuracy/cube_native_moment_reconstruction_3d.py",
        "studies/coupler_accuracy/cube_boundary_oracle.py", "source/solvers/fvm/io/mesh_storage.py",
        "tests/coupler/test_native_normal_moment_lift_3d.py")]
    for source in audit["sources"]:
        assert hash_file(ROOT / source["path"]) == source
    sources = [hash_file(path) for path in paths]
    for path in paths:
        if path.suffix == ".py":
            target = args.output / "sources" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    (args.output / "sources-at-start.json").write_text(json.dumps(sources, indent=2) + "\n")
    wall = next(p for p in mesh["boundary"] if p["name"] == "cube")
    locked = np.arange(wall["start_face"], wall["start_face"] + wall["n_faces"])
    free = np.setdiff1d(np.arange(mesh["n_faces"]), locked)
    exterior = free[free >= mesh["n_interior_faces"]]
    volume = fields["polyhedron_volume"]
    baseline = fields["implied_velocity"][2] * volume[:, None]
    assert fields["variant_names"][2] == "native_affine_velocity_trace"
    assert np.max(np.abs(fields["net_flux"] / volume)) < 1e-10
    records = []
    for dimensions in (2, 3):
        lift = NormalMomentLift.from_mesh(mesh, native, fields["face_origin"], volume, free, dimensions=dimensions)
        for target_name, target in (("stored_cell_velocity", fields["stored_velocity"] * volume[:, None]),
                                   ("stored_volume_momentum", fields["stored_velocity"] * fields["fvm_volume"][:, None])):
            required = target - baseline
            answer = lift.solve(required)
            diagnostic = answer["diagnostics"]
            assert diagnostic["stop"] in (1, 2) and diagnostic["dual_stop"] in (1, 2)
            assert diagnostic["velocity_constraint_rms"] < 1e-8
            assert abs(diagnostic["relative_primal_dual_energy_gap"]) < 1e-6
            flux, moment, energy, maximum, cell = independent_surface_observation(mesh, native, lift.centre, answer["face_gradient"])
            flux_difference = float(np.max(np.abs(flux) / lift.area))
            moment_difference = float(np.max(np.linalg.norm(moment - answer["face_first_moment"], axis=1) / lift.area**1.5))
            mean_difference = float(np.max(np.abs((cell - required) / volume[:, None] - answer["cell_velocity_residual"])))
            energy_difference = abs(float(energy.sum()) - diagnostic["normal_change_integrated_squared"]) / diagnostic["normal_change_integrated_squared"]
            assert flux_difference < 2e-11 and moment_difference < 2e-10 and mean_difference < 2e-10 and energy_difference < 1e-8
            np.testing.assert_array_equal(answer["face_gradient"][locked], 0)
            label = f"{target_name}-{dimensions}-modes"
            out = args.output / (label + ".npz")
            np.savez_compressed(out, normal_gradient=answer["face_gradient"], face_centroid=lift.centre, face_area=lift.area,
                                face_first_moment=answer["face_first_moment"], required_cell_integral=required,
                                cell_velocity_residual=answer["cell_velocity_residual"], normal_change_energy_by_face=energy,
                                maximum_normal_change_by_face=maximum, coefficients=answer["coefficients"],
                                dual_variables=answer["dual_variables"], locked_faces=locked, free_faces=free)
            record = {"name": label, "retained_modes": len(lift.mode_faces), "discarded_eigenvalues": lift.discarded_eigenvalues,
                      **diagnostic, "normal_change_rms_over_all_free_faces": float(np.sqrt(energy[free].sum() / lift.area[free].sum())),
                      "normal_change_rms_on_outer_faces": float(np.sqrt(energy[exterior].sum() / lift.area[exterior].sum())),
                      "maximum_normal_change": float(maximum.max()), "maximum_outer_normal_change": float(maximum[exterior].max()),
                      "locked_all_boundaries_global_velocity_error_lower_bound": float(np.linalg.norm(required.sum(axis=0)) / volume.sum()),
                      "independent_quadrature_flux_error_as_velocity": flux_difference,
                      "independent_quadrature_moment_error_over_area_to_three_halves": moment_difference,
                      "independent_quadrature_cell_velocity_difference": mean_difference,
                      "independent_quadrature_relative_energy_difference": energy_difference, "fields": hash_file(out)}
            records.append(record)
            print(json.dumps(record), flush=True)
    for source in sources:
        assert hash_file(ROOT / source["path"]) == source
    result = {"schema": "openonda-normal-moment-lift-3d/1", "status": "complete", "spatial_dimensions": 3,
              "snapshot": {"name": args.name, "fvm_step": args.step, "physical_time": row["physical_time"]},
              "sources": sources, "records": records, "elapsed_seconds": time.perf_counter() - start,
              "limitations": [
                  "An equality fit and its primal/dual energy check qualify only the chosen normal-trace space, not a continuous volume velocity, vorticity transfer or advancing flow.",
                  "Cube-wall traces and every total face flux are locked. Within-face normal variation on other outer faces is allowed; this is a diagnostic freedom, not a proposed boundary condition.",
                  "Two-mode traces use the dominant covariance directions. Three-mode traces retain all eigenvalues above sixteen machine epsilons times the face's largest eigenvalue.",
                  "The energy minimum is a minimum change relative to the specified native affine trace, and is not a velocity error against a continuum oracle.",
                  "A positive global lower bound with all boundary moments locked rules out exact cell-mean preservation for those fixed traces, irrespective of the interior normal corrections.",
                  "Large fitted normal variation must not be hidden by the small moment residual. No production solver or particle field is changed.",
              ]}
    (args.output / "normal-moment-lift-3d.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--name", choices=("full", "hybrid"), default="hybrid")
    parser.add_argument("--step", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.audit, args.verification, args.output = args.audit.resolve(), args.verification.resolve(), args.output.resolve()
    run(args)
