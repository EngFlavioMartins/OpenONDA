"""Frozen native planar wall GBD replay, without accepting a solver time step.

Use numerical sources from the explicit repository and one complete, immutable
native checkpoint bundle. Decode host arrays and geometry without initializing
a solver/device. Compare remeshing and molecular diffusion using the actual
stored particle coordinates, retaining physical parameters and strength guards.
The measured boundary-field changes do not establish force-amplitude recovery.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import h5py
import numpy as np


def audit(repository, directory):
    """Save one frozen operator report and induced fields under the input bundle."""
    sys.path.insert(0, str(repository))
    spec = importlib.util.spec_from_file_location(
        "wall_audit",
        Path(__file__).with_name("audit_saved_wall_circulation.py"),
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    from source.coupler.stable_renewal import vortex_invariants
    from source.solvers.vpm.physics.diffusion import planar
    from source.solvers.vpm.physics.diffusion.planar import planar_gbd

    if (
        Path(planar.__file__).resolve()
        != (repository / "source/solvers/vpm/physics/diffusion/planar.py").resolve()
    ):
        raise RuntimeError("Planar numerical source does not belong to the supplied repository")

    metadata_path = directory / "checkpoint/checkpoint_info.json"
    metadata = json.loads(metadata_path.read_text())
    state_path = metadata_path.parent / metadata["checkpoint_files"]["fvm"]
    vpm_path = metadata_path.parent / metadata["checkpoint_files"]["vpm"]
    _, _, _, _, boundary, wall, _ = helper.load_donor_geometry(directory, state_path)
    assert wall.revision in metadata["config"]["solid_geometry"]["wall_revisions"]
    configuration = metadata["config"]["vpm"]
    config = SimpleNamespace(**configuration["viscous"])
    h, span = config.gbd_grid_spacing, configuration["induction"]["planar_span"]
    with h5py.File(vpm_path) as saved:
        position = np.asarray(saved["particles/position"])
        strength = np.asarray(saved["particles/vortex_strength"])
    with np.load(directory / "wall_trace_fields.npz") as saved:
        fields = {key: saved[key].copy() for key in saved.files}

    class FrozenParticles:
        n_particles_total = len(position)

        def position_cpu(self):
            return position

        def vortex_strength_cpu(self):
            return strength

        def kinematic_viscosity_cpu(self):
            return np.full(len(position), config.kinematic_viscosity)

    paths = [
        metadata_path,
        state_path,
        vpm_path,
        directory / "coupled_mesh.npz",
        directory / "wall_trace_fields.npz",
    ]
    hashes = {str(path): helper.digest(path) for path in paths}
    original = vortex_invariants(position, strength)
    rows = {}
    raw = {}
    for name, dt in (
        ("remesh_prune_only", 0),
        ("zero_flux_molecular_diffusion", configuration["time_step_size"]),
    ):
        induction = SimpleNamespace(planar_span=span, plane_z=0)
        timings = []
        for _repeat in range(2):
            start = perf_counter()
            replacement, substeps = planar_gbd(
                FrozenParticles(),
                config,
                induction,
                dt=dt,
                anchor=metadata["config"]["transfer_lattice"]["anchor"],
                core_radius_ratio=config.core_radius_ratio,
                max_particles=configuration["max_n_particles"],
                solid_at=lambda points: boundary.contains(points, include_boundary=False),
                wall_crosses=boundary.blocks_segments,
                geometry_key=("frozen-verified-wall", wall.revision),
            )
            timings.append(perf_counter() - start)
        new_position, new_strength = replacement["position"], replacement["vortex_strength"]
        assert not boundary.contains(new_position, include_boundary=False).any()
        invariants = vortex_invariants(new_position, new_strength)
        row = {
            "diffusion_interval_seconds": dt,
            "substeps": substeps,
            "operator_seconds_cold_warm": timings,
            "output_particle_count": len(new_position),
            "circulation": float(invariants.total_vortex_strength[2]),
            "circulation_change": float(
                invariants.total_vortex_strength[2] - original.total_vortex_strength[2]
            ),
            "linear_impulse_change": (invariants.linear_impulse - original.linear_impulse).tolist(),
            "strength_l1": float(np.abs(new_strength[:, 2]).sum()),
            "native_moment_recovery": induction.last_gbd_moment_recovery,
            "native_wall_transfer": induction.last_gbd_wall_transfer,
            "native_wall_geometry": induction.last_planar_wall_geometry,
        }
        for patch in ("cylinder", "numericalBoundary"):
            points, normals, area = [
                fields[f"{patch}_{key}"] for key in ("centre", "normal", "area")
            ]
            induced, gradient = helper.gaussian_velocity_and_gradient(
                points, new_position, new_strength[:, 2] / span, h * config.core_radius_ratio
            )
            old_induced, old_gradient = helper.gaussian_velocity_and_gradient(
                points, position, strength[:, 2] / span, h * config.core_radius_ratio
            )
            derivative = np.einsum("fij,fj->fi", gradient - old_gradient, normals)
            tangent = derivative - np.einsum("fi,fi->f", derivative, normals)[:, None] * normals
            row[patch] = {
                "total_velocity": helper.velocity_statistics(induced + [1, 0, 0], normals, area),
                "velocity_change": helper.velocity_statistics(induced - old_induced, normals, area),
                "tangential_normal_gradient_change_rms": float(
                    np.sqrt(np.average(np.sum(tangent**2, axis=1), weights=area))
                ),
            }
            raw[f"{name}_{patch}_velocity"] = induced + [1, 0, 0]
            raw[f"{name}_{patch}_jacobian"] = gradient
        rows[name] = row
    assert all(helper.digest(path) == hashes[str(path)] for path in paths)
    source_paths = [
        repository / "source/solvers/vpm/physics/diffusion/planar.py",
        repository / "source/solvers/vpm/core/evolution.py",
    ]
    report = {
        "physical_time": metadata["time"],
        "physical_time_advanced": 0,
        "source_sha256": {
            str(path.relative_to(repository)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        },
        "input_sha256": hashes,
        "measurements": rows,
        "interpretation": "Detached wall-visible M4/no-flux GBD replay on unchanged accepted 46 s particles. Molecular wall reflection can change first/second moments; pruning conserves those post-diffusion moments per connected fluid component. This demonstrates operator conservation, not evolved force-amplitude recovery. Timings exclude solver/device evolution, FVM and MPI.",
    }
    destination = directory / "candidate_planar_wall_gbd.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(directory / "candidate_planar_wall_gbd_fields.npz", **raw)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    arguments = parser.parse_args()
    print(audit(arguments.repository, arguments.directory))
