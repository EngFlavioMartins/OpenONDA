"""One native frozen GBD pass: remeshing/pruning versus diffusion/wall absorption.

The saved particles do not move. This diagnoses an operator and its induced
boundary-field change; it is not an accepted coupled time step or force result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from source.coupler.stable_renewal import vortex_invariants
from source.solvers.vpm.physics.diffusion.planar import planar_gbd
from tests.support.cylinder.audit_saved_wall_circulation import (
    digest,
    gaussian_velocity_and_gradient,
    load_donor_geometry,
    velocity_statistics,
)


class FrozenParticles:
    """Read-only host fields required by the detached native GBD operator."""

    def __init__(self, position, strength, viscosity):
        self.position = position
        self.strength = strength
        self.viscosity = viscosity
        self.n_particles_total = len(position)

    def position_cpu(self):
        return self.position

    def vortex_strength_cpu(self):
        return self.strength

    def kinematic_viscosity_cpu(self):
        return np.full(self.n_particles_total, self.viscosity)


def audit(directory):
    metadata_path = directory / "checkpoint/checkpoint_info.json"
    metadata = json.loads(metadata_path.read_text())
    state_path = metadata_path.parent / metadata["checkpoint_files"]["fvm"]
    vpm_path = metadata_path.parent / metadata["checkpoint_files"]["vpm"]
    paths = (
        metadata_path,
        state_path,
        vpm_path,
        directory / "coupled_mesh.npz",
        directory / "wall_trace_fields.npz",
    )
    hashes = {str(path): digest(path) for path in paths}
    _, _, _, _, boundary, wall, _ = load_donor_geometry(directory, state_path)
    if wall.revision not in metadata["config"]["solid_geometry"]["wall_revisions"]:
        raise ValueError("Reconstructed wall does not match checkpoint")
    configuration = metadata["config"]["vpm"]
    config = SimpleNamespace(**configuration["viscous"])
    h = config.particle_spacing
    span = configuration["induction"]["planar_span"]
    anchor = metadata["config"]["transfer_lattice"]["anchor"]
    with h5py.File(vpm_path) as saved:
        position = np.asarray(saved["particles/position"])
        strength = np.asarray(saved["particles/vortex_strength"])
    particles = FrozenParticles(position, strength, config.kinematic_viscosity)
    original = vortex_invariants(position, strength)
    with np.load(directory / "wall_trace_fields.npz") as saved:
        fields = {key: saved[key].copy() for key in saved.files}
    rows = {}
    arrays = {}
    for name, dt, use_solid in (
        ("remesh_prune_only", 0, True),
        ("remesh_prune_without_solid_mask", 0, False),
        ("molecular_diffusion_with_wall_absorption", configuration["time_step_size"], True),
        ("molecular_diffusion_without_solid_mask", configuration["time_step_size"], False),
    ):
        induction = SimpleNamespace(planar_span=span, plane_z=0)
        replacement, substeps = planar_gbd(
            particles,
            config,
            induction,
            dt=dt,
            anchor=anchor,
            core_radius_ratio=config.core_radius_ratio,
            max_particles=configuration["max_n_particles"],
            solid_at=(lambda points: boundary.contains(points, include_boundary=False))
            if use_solid
            else None,
        )
        new_position = replacement["position"]
        new_strength = replacement["vortex_strength"]
        invariants = vortex_invariants(new_position, new_strength)
        row = {
            "diffusion_interval_seconds": dt,
            "substeps": substeps,
            "output_particle_count": len(new_position),
            "circulation": float(invariants.total_vortex_strength[2]),
            "circulation_change": float(
                invariants.total_vortex_strength[2] - original.total_vortex_strength[2]
            ),
            "linear_impulse_change": (invariants.linear_impulse - original.linear_impulse).tolist(),
            "strength_l1": float(np.abs(new_strength[:, 2]).sum()),
            "native_moment_recovery": induction.last_gbd_moment_recovery,
        }
        for patch in ("cylinder", "numericalBoundary"):
            points = fields[f"{patch}_centre"]
            normals = fields[f"{patch}_normal"]
            area = fields[f"{patch}_area"]
            induced, gradient = gaussian_velocity_and_gradient(
                points, new_position, new_strength[:, 2] / span, h * config.core_radius_ratio
            )
            original_induced, original_gradient = gaussian_velocity_and_gradient(
                points, position, strength[:, 2] / span, h * config.core_radius_ratio
            )
            total = induced + np.array([1, 0, 0])
            derivative = np.einsum("fij,fj->fi", gradient - original_gradient, normals)
            tangential = derivative - np.einsum("fi,fi->f", derivative, normals)[:, None] * normals
            row[patch] = {
                "total_velocity": velocity_statistics(total, normals, area),
                "velocity_change": velocity_statistics(induced - original_induced, normals, area),
                "tangential_normal_gradient_change_rms": float(
                    np.sqrt(np.average(np.sum(tangential**2, axis=1), weights=area))
                ),
            }
            arrays[f"{name}_{patch}_velocity"] = total
            arrays[f"{name}_{patch}_jacobian"] = gradient
        rows[name] = row
    if any(digest(path) != hashes[str(path)] for path in paths):
        raise RuntimeError("Frozen inputs changed during GBD audit")
    report = {
        "physical_time": metadata["time"],
        "physical_time_advanced": 0,
        "original_particle_count": len(position),
        "original_circulation": float(original.total_vortex_strength[2]),
        "original_strength_l1": float(np.abs(strength[:, 2]).sum()),
        "input_sha256": hashes,
        "interpretation": (
            "Detached single GBD operator pass with no advection or FVM refresh. "
            "The no-solid-mask variant is diagnostic only; normal run guards and "
            "geometry are unchanged. Native moment recovery conserves the moments "
            "of the already diffused/masked grid, not pre-diffusion wall circulation. "
            "The bounded-domain removal phase is not part of this isolated operator."
        ),
        "measurements": rows,
    }
    np.savez_compressed(directory / "frozen_planar_gbd_fields.npz", **arrays)
    destination = directory / "frozen_planar_gbd.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    arguments = parser.parse_args()
    print(audit(arguments.directory))
