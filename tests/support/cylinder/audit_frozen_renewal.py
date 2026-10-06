"""Separate frozen native renewal from physical advection and GBD diffusion.

Repeated renewal of one saved FVM velocity target is an operator experiment.
It does not represent additional accepted flow time or establish force recovery.
Native geometry, Gaussian radius, amplification and particle capacity are kept.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

from source.coupler.stable_renewal import (
    build_stable_renewal_lattice,
    gaussian_represented_vortex_strength,
    renew_stable_overlap,
    scatter_m4_prime_to_lattice,
    vortex_invariants,
)
from source.coupler.vorticity_transfer import required_renewal_buffer_length
from tests.support.cylinder.audit_saved_transfer_fields import comparison, regions
from tests.support.cylinder.audit_saved_wall_circulation import (
    digest,
    gaussian_velocity_and_gradient,
    load_donor_geometry,
    velocity_statistics,
)


def smoothstep(distance, lower, upper):
    phase = np.clip((distance - lower) / (upper - lower), 0, 1)
    return phase**2 * (3 - 2 * phase)


def invariant_values(position, strength):
    value = vortex_invariants(position, strength)
    return {
        "circulation": float(value.total_vortex_strength[2]),
        "linear_impulse": value.linear_impulse.tolist(),
    }


def audit(directory, repetitions, particle_spacing=None, zero_inner_coefficients=False):
    started = perf_counter()
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
    mesh, geometry, state, gradient, boundary, wall, trace = load_donor_geometry(
        directory, state_path
    )
    if wall.revision not in metadata["config"]["solid_geometry"]["wall_revisions"]:
        raise ValueError("Reconstructed wall does not match checkpoint")
    configuration = metadata["config"]
    coupling = configuration["coupler"]
    vpm_configuration = configuration["vpm"]
    viscous = vpm_configuration["viscous"]
    original_spacing = viscous["particle_spacing"]
    h = original_spacing if particle_spacing is None else float(particle_spacing)
    span = vpm_configuration["induction"]["planar_span"]
    anchor = configuration["transfer_lattice"]["anchor"]
    box = coupling["transfer_region_bounds"]
    box = np.array([box[f"{axis}{side}"] for axis in "xyz" for side in ("min", "max")])
    outer_faces = np.concatenate(
        [
            np.arange(patch["start_face"], patch["start_face"] + patch["n_faces"])
            for patch in mesh["boundary"]
            if patch["type"] != "wall"
        ]
    )
    outer = geometry["face_centre"][outer_faces]
    fvm_box = np.column_stack((outer.min(axis=0), outer.max(axis=0))).ravel()

    def mesh_weight(points):
        roundoff = 64 * np.finfo(float).eps * np.maximum(1, np.abs(fvm_box))
        return np.all(
            (points >= fvm_box[::2] - roundoff[::2]) & (points <= fvm_box[1::2] + roundoff[1::2]),
            axis=1,
        ).astype(float)

    def fluid_weight(points):
        return smoothstep(boundary.signed_distance(points), -h, 0)

    def interior(points):
        return boundary.contains(points, include_boundary=False)

    lattice = build_stable_renewal_lattice(
        box,
        h,
        buffer_length=required_renewal_buffer_length(
            coupling["freestream_velocity"], vpm_configuration["time_step_size"], original_spacing
        ),
        blend_ramp_width=coupling["eta_blend_width"],
        vpm_dead_zone=coupling["vpm_only_width"],
        lattice_anchor=anchor,
        mesh_weight_at_node=mesh_weight,
        fluid_weight_at_node=fluid_weight,
        interior_at_node=interior,
        planar_span=span,
        plane_z=0,
        solid_boundary=boundary,
    )
    needed = (lattice.mesh_weight > 0) | (lattice.fluid_weight < 1)
    targets = np.zeros_like(lattice.positions)
    velocity = state["velocity"][: mesh["n_cells"]]
    for axis, component, sign in ((0, 1, 1), (1, 0, -1)):
        for side in (-1, 1):
            points = lattice.positions[needed].copy()
            points[:, axis] += side * h / 2
            weight = smoothstep(boundary.signed_distance(points), 0, h)
            valid = weight > 0
            values = np.zeros_like(points)
            values[valid] = trace.prepare(points[valid]).sample(velocity, gradient)
            targets[needed, 2] += sign * side * h * span * values[:, component] * weight
    with h5py.File(vpm_path) as saved:
        original_position = np.asarray(saved["particles/position"])
        original_strength = np.asarray(saved["particles/vortex_strength"])
    with np.load(directory / "wall_trace_fields.npz") as saved:
        fields = {key: saved[key].copy() for key in saved.files}
    fluid = (~lattice.solid_interior) & (lattice.mesh_weight > 0)
    regional_masks = regions(lattice.positions, fluid)
    report_rows = {}
    arrays = {"lattice_position": lattice.positions, "fixed_fvm_target": targets}
    pruning_cases = (
        ("native_pruning", "no_pruning") if particle_spacing is None else ("native_pruning",)
    )
    original_derivatives = {}
    for patch in ("cylinder", "numericalBoundary"):
        _, jacobian = gaussian_velocity_and_gradient(
            fields[f"{patch}_centre"],
            original_position,
            original_strength[:, 2] / span,
            original_spacing * viscous["core_radius_ratio"],
        )
        normals = fields[f"{patch}_normal"]
        derivative = np.einsum("fij,fj->fi", jacobian, normals)
        original_derivatives[patch] = (
            derivative - np.einsum("fi,fi->f", derivative, normals)[:, None] * normals
        )
    for pruning in pruning_cases:
        position, strength = original_position.copy(), original_strength.copy()
        if zero_inner_coefficients:
            in_belt = np.all(
                (position >= lattice.renewal_bounds[::2])
                & (position <= lattice.renewal_bounds[1::2]),
                axis=1,
            )
            position, strength = position[~in_belt], strength[~in_belt]
        measurements = []
        for repetition in range(repetitions + 1):
            result = None
            if repetition:
                result = renew_stable_overlap(
                    position,
                    strength,
                    lattice,
                    fvm_vortex_strength_at_node=lambda _points: targets,
                    particle_fluid_weight=fluid_weight,
                    particle_in_solid=interior,
                    prune_threshold=(coupling["transfer_vorticity_cutoff"] * h**2 * span)
                    if pruning == "native_pruning"
                    else 0,
                    release_prune_threshold=viscous["gbd_threshold"] * (h / original_spacing) ** 2
                    if pruning == "native_pruning"
                    else 0,
                    core_radius_ratio=viscous["core_radius_ratio"],
                    amplification_cap=coupling["transfer_amplification_cap"],
                    maximum_particle_count=vpm_configuration["max_n_particles"],
                    freestream_speed=float(np.linalg.norm(coupling["freestream_velocity"])),
                    time_step_size=vpm_configuration["time_step_size"],
                    compute_diagnostics=True,
                )
                position, strength = result.position, result.vortex_strength
            if repetition not in (0, 1, 5, repetitions):
                continue
            if repetition == 0 and particle_spacing is not None:
                # A refined convolution of coarse coefficients is not the
                # saved represented field. Compare mapped results to the
                # actual saved trace instead of reporting that false baseline.
                continue
            valid = ~interior(position)
            belt = valid & np.all(
                (position >= lattice.renewal_bounds[::2])
                & (position <= lattice.renewal_bounds[1::2]),
                axis=1,
            )
            coefficients = scatter_m4_prime_to_lattice(position[belt], strength[belt], lattice)
            represented = gaussian_represented_vortex_strength(
                coefficients,
                lattice.shape,
                h,
                core_radius=viscous["core_radius_ratio"] * h,
                dimensions=2,
            )
            row = {
                "renewal_repetitions": repetition,
                "particle_count": len(position),
                "renewed_coefficient_invariants": invariant_values(lattice.positions, coefficients),
                "represented_all_node_invariants": invariant_values(lattice.positions, represented),
                "represented_fluid_node_invariants": invariant_values(
                    lattice.positions[fluid], represented[fluid]
                ),
                "maximum_renewed_coefficient_strength": float(
                    np.max(np.abs(coefficients[:, 2]), initial=0)
                ),
                "regional_curl": {
                    name: comparison(
                        targets[mask, 2] / lattice.particle_volume,
                        represented[mask, 2] / lattice.particle_volume,
                        np.ones(mask.sum()),
                    )
                    for name, mask in regional_masks.items()
                },
            }
            for patch in ("cylinder", "numericalBoundary"):
                points = fields[f"{patch}_centre"]
                normals = fields[f"{patch}_normal"]
                area = fields[f"{patch}_area"]
                renewed_count = len(position) if result is None else result.renewed_output_count
                induced, jacobian = gaussian_velocity_and_gradient(
                    points,
                    position[:renewed_count],
                    strength[:renewed_count, 2] / span,
                    h * viscous["core_radius_ratio"],
                )
                if renewed_count < len(position):
                    outer_velocity, outer_gradient = gaussian_velocity_and_gradient(
                        points,
                        position[renewed_count:],
                        strength[renewed_count:, 2] / span,
                        original_spacing * viscous["core_radius_ratio"],
                    )
                    induced += outer_velocity
                    jacobian += outer_gradient
                total = induced + np.array([1.0, 0, 0])
                derivative = np.einsum("fij,fj->fi", jacobian, normals)
                tangent = derivative - np.einsum("fi,fi->f", derivative, normals)[:, None] * normals
                row[patch] = {
                    "total_particle_velocity": velocity_statistics(total, normals, area),
                    "change_from_saved_particle_velocity": velocity_statistics(
                        total - fields[f"{patch}_vpm_velocity"], normals, area
                    ),
                    "tangential_normal_gradient_rms": float(
                        np.sqrt(np.average(np.sum(tangent**2, axis=1), weights=area))
                    ),
                    "tangential_normal_gradient_change_rms": float(
                        np.sqrt(
                            np.average(
                                np.sum((tangent - original_derivatives[patch]) ** 2, axis=1),
                                weights=area,
                            )
                        )
                    ),
                }
                arrays[f"{pruning}_{repetition}_{patch}_velocity"] = total
                arrays[f"{pruning}_{repetition}_{patch}_jacobian"] = jacobian
            invariants = vortex_invariants(position, strength)
            row["circulation"] = float(invariants.total_vortex_strength[2])
            row["linear_impulse"] = invariants.linear_impulse.tolist()
            if result is not None:
                row.update(
                    representation_residual_before_prune=result.representation_residual_before_prune,
                    representation_residual_after_prune=result.representation_residual_after_prune,
                    maximum_transfer_amplification=result.maximum_transfer_amplification,
                    prune_correction_fraction=result.conservation_applied_particle_strength_fraction,
                )
            measurements.append(row)
            if repetition == repetitions:
                arrays[f"{pruning}_final_position"] = position
                arrays[f"{pruning}_final_strength"] = strength
        report_rows[pruning] = measurements
    if any(digest(path) != hashes[str(path)] for path in paths):
        raise RuntimeError("Frozen inputs changed during operator audit")
    report = {
        "physical_time": metadata["time"],
        "physical_time_advanced": 0,
        "zero_inner_coefficients": zero_inner_coefficients,
        "original_particle_spacing": original_spacing,
        "renewed_particle_spacing": h,
        "physical_blend_width": coupling["eta_blend_width"],
        "physical_release_width": coupling["vpm_only_width"],
        "physical_renewal_buffer": float(lattice.buffer_length),
        "fixed_fluid_fvm_target_invariants": invariant_values(
            lattice.positions[fluid], targets[fluid]
        ),
        "first_blended_target_invariants": invariant_values(
            lattice.positions[fluid],
            targets[fluid] * lattice.fvm_blend_weight[fluid, None],
        ),
        "fixed_fluid_target_maximum_strength": float(np.max(np.abs(targets[fluid, 2]), initial=0)),
        "lattice_shape": lattice.shape,
        "elapsed_seconds": perf_counter() - started,
        "input_sha256": hashes,
        "interpretation": (
            "Frozen target repeated renewal only, no physical advection or diffusion. "
            "Default and zero pruning share geometry, radius, capacity and amplification "
            "guards. Pure renewal convergence is not a force-amplitude result. Direct "
            "Gaussian outer-face changes are measured against saved accepted state. "
            "If spacing is refined, coarse saved coefficients seed the refined inner "
            "basis; the exterior untouched sources retain their actual old core radius "
            "for induced-field comparisons. This is an offline mixed-resolution mapping, "
            "not a supported native continuation. The physical blending, release and "
            "renewal-buffer widths stay fixed. Pruning scales with particle volume. "
            "A zero-inner-coefficient experiment removes the same physical renewal "
            "belt from the saved inputs for both spacings, preserving the exact "
            "saved exterior wake. This avoids the coarse-coefficient amplification "
            "exemption on the first refined correction, but deliberately starts from "
            "an empty VPM-owned release belt. It is not a compatible restart."
        ),
        "measurements": report_rows,
    }
    suffix = "" if particle_spacing is None else f"_h{h:.3f}"
    if zero_inner_coefficients:
        suffix += "_zero_inner"
    np.savez_compressed(directory / f"frozen_renewal{suffix}_fields.npz", **arrays)
    destination = directory / f"frozen_renewal{suffix}.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--repetitions", type=int, default=25)
    parser.add_argument("--particle-spacing", type=float)
    parser.add_argument("--zero-inner-coefficients", action="store_true")
    arguments = parser.parse_args()
    print(
        audit(
            arguments.directory,
            arguments.repetitions,
            arguments.particle_spacing,
            arguments.zero_inner_coefficients,
        )
    )
