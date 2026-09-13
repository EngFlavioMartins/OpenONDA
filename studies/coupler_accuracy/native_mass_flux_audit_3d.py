#!/usr/bin/env python3
"""Compare conservative FVM flux and velocity interpolation on identical cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from source.solvers.fvm.fields.diagnostics import compute_continuity_error
from source.solvers.fvm.io.backup import decode_state
from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, field_rms, hash_file
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays


def net_flux(value, mesh, geo):
    n, ni = mesh["n_cells"], mesh["n_interior_faces"]
    actual = (np.bincount(mesh["owners"], weights=value, minlength=n)
              -np.bincount(mesh["neighbours"], weights=value[:ni], minlength=n))
    expected = compute_continuity_error(value, mesh, geo)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-14)
    np.testing.assert_allclose(actual.sum(), value[ni:].sum(), rtol=0, atol=2e-13)
    return actual


def circulation(face_velocity, mesh, geo):
    n, ni = mesh["n_cells"], mesh["n_interior_faces"]
    flux = np.cross(geo["face_area_vector"], face_velocity)
    return np.column_stack([np.bincount(mesh["owners"], weights=flux[:, j], minlength=n)
                            -np.bincount(mesh["neighbours"], weights=flux[:ni, j], minlength=n) for j in range(3)])


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    small_path = args.oracle / "small-native-mesh.npz"
    full_path = args.oracle / "full-native-mesh.npz"
    map_path = args.oracle / "cell-and-face-map.npz"
    small = load_native_mesh(small_path)
    small_geo = compute_mesh_geometry(small, gradient_scheme="gauss", compute_lsq=False)
    mapping = read_arrays(map_path)
    entries, paths = [], [Path(__file__).resolve(), small_path, full_path, map_path]
    for directory in args.trial:
        name = directory.parent.name if directory.name == "trial" else directory.name
        mesh_path = directory / "hybrid/solution/mesh.npz"
        backup_path = directory / "hybrid/solution/backups/fvm_000020.npz"
        comparison_path = directory / "latest-comparison-fields.npz"
        entries.append((name, mesh_path, backup_path, np.arange(small["n_cells"]), comparison_path, None))
        paths += [mesh_path, backup_path, comparison_path]
    if args.reference:
        reference_path = args.reference / "reference-flux-replay-3d.json"
        reference = json.loads(reference_path.read_text())
        assert reference["status"] == "complete" and reference["spatial_dimensions"] == 3
        for row in reference["sources"]:
            assert hash_file(ROOT / row["path"]) == row
        paths.append(reference_path)
        for row in reference["records"]:
            backup = ROOT / row["backup"]["path"]
            assert hash_file(backup) == row["backup"]
            entries.append(("reference-"+row["name"], full_path, backup, mapping["cell_ids"], None, row["physical_time"]))
            paths.append(backup)
    if not entries or len({entry[0] for entry in entries}) != len(entries):
        raise ValueError("At least one uniquely named trial or reference snapshot is required")
    paths += [ROOT / name for name in (
        "source/solvers/fvm/fields/diagnostics.py", "source/solvers/fvm/io/backup.py", "source/solvers/fvm/io/mesh_storage.py",
        "source/solvers/fvm/mesh/geometry.py", "studies/coupler_accuracy/cube_boundary_oracle.py",
        "studies/coupler_accuracy/cube_native_moment_reconstruction_3d.py")]
    paths = list(dict.fromkeys(paths))
    sources = [hash_file(path) for path in paths]
    for path in paths:
        if path.suffix == ".py":
            target = args.output / "sources" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    (args.output / "sources-at-start.json").write_text(json.dumps(sources, indent=2)+"\n")
    radius = np.max(np.abs(small_geo["cell_centre"]), axis=1)
    masks = {"all_shared": np.ones(small["n_cells"], dtype=bool), "near_body": radius < .8,
             "overlap": (radius >= .75) & (radius < 1.25), "outer_shared": radius >= 1.25}
    records = []
    for name, mesh_path, backup_path, selected_cells, comparison_path, physical_time in entries:
        mesh = load_native_mesh(mesh_path)
        geo = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
        state = decode_state(read_arrays(backup_path))
        u, phi = state["velocity"], state["volumetric_face_flux"]
        n, ni, nf = mesh["n_cells"], mesh["n_interior_faces"], mesh["n_faces"]
        np.testing.assert_allclose(geo["cell_centre"][selected_cells], small_geo["cell_centre"], rtol=0, atol=1e-13)
        np.testing.assert_allclose(geo["cell_volume"][selected_cells], small_geo["cell_volume"], rtol=0, atol=1e-14)
        if comparison_path:
            comparison = read_arrays(comparison_path)
            np.testing.assert_array_equal(u[:n], comparison["hybrid_velocity"])
        assert u.shape == (n+nf-ni, 3) and phi.shape == (nf,)
        own, nei = mesh["owners"], mesh["neighbours"]
        w = geo["face_interpolation_weight"][:ni, None]
        sf = geo["face_area_vector"]
        area = np.linalg.norm(sf, axis=1)
        normal = sf/area[:, None]
        face_u = np.empty((nf, 3))
        face_u[:ni] = (1-w)*u[own[:ni]]+w*u[nei]
        face_u[ni:] = u[n:]
        velocity_flux = np.sum(face_u*sf, axis=1)
        interior_replacement = phi.copy()
        interior_replacement[:ni] = velocity_flux[:ni]

        actual_div = net_flux(phi, mesh, geo)/geo["cell_volume"]
        replaced_div = net_flux(interior_replacement, mesh, geo)/geo["cell_volume"]
        corrected_face_u = face_u+((phi-velocity_flux)/area)[:, None]*normal
        np.testing.assert_allclose(np.sum(corrected_face_u*sf, axis=1), phi, rtol=0, atol=2e-14)

        gamma = circulation(face_u, mesh, geo)
        corrected_gamma = circulation(corrected_face_u, mesh, geo)
        curl_change = (corrected_gamma-gamma)/geo["cell_volume"][:, None]
        np.testing.assert_allclose(curl_change, 0, rtol=0, atol=2e-12)
        in_shared = np.zeros(n, dtype=bool)
        in_shared[selected_cells] = True
        shared_faces = in_shared[own[:ni]] & in_shared[nei]
        delta_un = (velocity_flux[:ni]-phi[:ni])/area[:ni]
        regions = {}
        for region, mask in masks.items():
            ids = selected_cells[mask]
            regions[region] = {"cells": len(ids), "stored_flux_divergence_rms": field_rms(actual_div[ids, None], geo["cell_volume"][ids]),
                               "stored_flux_divergence_maximum": float(np.max(np.abs(actual_div[ids]))),
                               "interior_velocity_flux_divergence_rms": field_rms(replaced_div[ids, None], geo["cell_volume"][ids]),
                               "interior_velocity_flux_divergence_maximum": float(np.max(np.abs(replaced_div[ids])))}
        snapshot_path = args.output / (name+"-flux-fields.npz")
        np.savez_compressed(snapshot_path, selected_cell_ids=selected_cells, cell_volume=geo["cell_volume"][selected_cells],
                            stored_flux_divergence=actual_div[selected_cells], interior_velocity_flux_divergence=replaced_div[selected_cells],
                            shared_interior_face_ids=np.flatnonzero(shared_faces), interior_face_area=area[:ni][shared_faces],
                            interior_normal_velocity_difference=delta_un[shared_faces],
                            circulation=gamma[selected_cells], corrected_normal_circulation=corrected_gamma[selected_cells],
                            vorticity_change_from_normal_flux_correction=curl_change[selected_cells],
                            **{region+"__cell_rows": np.flatnonzero(mask) for region, mask in masks.items()})
        row = {"name": name, "solver_time": float(state["time"]), "solver_step": int(state["step"]),
               "physical_time": physical_time, "mesh_cells": n, "shared_cells": len(selected_cells),
               "shared_interior_faces": int(shared_faces.sum()), "regions": regions,
               "interior_normal_velocity_difference_rms": field_rms(delta_un[shared_faces, None], area[:ni][shared_faces]),
               "interior_normal_velocity_difference_maximum": float(np.max(np.abs(delta_un[shared_faces]))),
               "maximum_vorticity_change_from_normal_flux_correction": float(np.max(np.abs(curl_change[selected_cells]))),
               "global_stored_flux_imbalance": float(phi[ni:].sum()), "fields": hash_file(snapshot_path)}
        records.append(row)
        print(json.dumps(row, indent=2), flush=True)
    result = {"schema": "openonda-native-mass-flux-audit-3d/1", "status": "complete", "spatial_dimensions": 3,
              "records": records, "sources": sources,
              "limitations": [
                  "Every reported region uses the same mapped small-domain native cells and cell volumes, including for the fully meshed reference.",
                  "The comparison replaces only interior face fluxes by linear interpolation of cell velocity. Stored physical-boundary fluxes remain fixed.",
                  "Nonzero divergence of that interpolated field is a discrete representation difference, not compressible flow or proof of a pressure-solver error.",
                  "Correcting a face velocity only along its native normal enforces the stored flux without changing its native Gauss circulation. This does not define a unique continuous velocity field.",
                  "The reset-initial reference snapshot reproduces an initial-state reset, not an accepted pressure-corrected time step.",
                  "These observations do not establish which representation difference dominates the advancing force error."]}
    (args.output / "native-mass-flux-audit-3d.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--trial", type=Path, nargs="*", default=[])
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.oracle, args.output = args.oracle.resolve(), args.output.resolve()
    args.trial = [path.resolve() for path in args.trial]
    if args.reference:
        args.reference = args.reference.resolve()
    run(args)
