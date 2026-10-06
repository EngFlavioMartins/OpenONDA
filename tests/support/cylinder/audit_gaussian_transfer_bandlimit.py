"""Read-only Gaussian transfer resolution and guard audit on a frozen cylinder.

The Fourier symbols describe the native sampled Gaussian in a homogeneous
infinite lattice. Target spectra are diagnostic: walls and blending mix these
modes, so this is neither an evolved amplitude prediction nor a guard ablation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from source.coupler.stable_renewal import (
    blend_represented_state,
    build_stable_renewal_lattice,
    gaussian_represented_vortex_strength,
    scatter_m4_prime_to_lattice,
)
from source.coupler.vorticity_transfer import required_renewal_buffer_length
from tests.support.cylinder.audit_frozen_renewal import smoothstep
from tests.support.cylinder.audit_saved_wall_circulation import digest, load_donor_geometry


def audit(directory):
    metadata_path = directory / "checkpoint/checkpoint_info.json"
    metadata = json.loads(metadata_path.read_text())
    state_path = metadata_path.parent / metadata["checkpoint_files"]["fvm"]
    vpm_path = metadata_path.parent / metadata["checkpoint_files"]["vpm"]
    field_path = directory / "frozen_renewal_fields.npz"
    input_paths = (metadata_path, state_path, vpm_path, field_path, directory / "coupled_mesh.npz")
    hashes = {str(path): digest(path) for path in input_paths}
    mesh, geometry, _, _, boundary, wall, _ = load_donor_geometry(directory, state_path)
    if wall.revision not in metadata["config"]["solid_geometry"]["wall_revisions"]:
        raise ValueError("Frozen wall differs from native checkpoint geometry")
    config = metadata["config"]
    coupling, vpm = config["coupler"], config["vpm"]
    viscous = vpm["viscous"]
    h, span = viscous["particle_spacing"], vpm["induction"]["planar_span"]
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
    fvm_box = np.column_stack((outer.min(0), outer.max(0))).ravel()

    def mesh_weight(points):
        tolerance = 64 * np.finfo(float).eps * np.maximum(1, np.abs(fvm_box))
        return np.all(
            (points >= fvm_box[::2] - tolerance[::2]) & (points <= fvm_box[1::2] + tolerance[1::2]),
            axis=1,
        ).astype(float)

    def fluid_weight(points):
        return smoothstep(boundary.signed_distance(points), -h, 0)

    lattice = build_stable_renewal_lattice(
        box,
        h,
        buffer_length=required_renewal_buffer_length(
            coupling["freestream_velocity"], vpm["time_step_size"], h
        ),
        blend_ramp_width=coupling["eta_blend_width"],
        vpm_dead_zone=coupling["vpm_only_width"],
        lattice_anchor=config["transfer_lattice"]["anchor"],
        mesh_weight_at_node=mesh_weight,
        fluid_weight_at_node=fluid_weight,
        interior_at_node=lambda p: boundary.contains(p, include_boundary=False),
        planar_span=span,
        plane_z=0,
        solid_boundary=boundary,
    )
    with np.load(field_path) as saved:
        np.testing.assert_array_equal(lattice.positions, saved["lattice_position"])
        target = saved["fixed_fvm_target"].copy()
    with h5py.File(vpm_path) as stored:
        position = np.asarray(stored["particles/position"])
        strength = np.asarray(stored["particles/vortex_strength"])
    in_belt = (~boundary.contains(position, include_boundary=False)) & np.all(
        (position >= lattice.renewal_bounds[::2]) & (position <= lattice.renewal_bounds[1::2]),
        axis=1,
    )
    coefficients = scatter_m4_prime_to_lattice(
        position[in_belt], strength[in_belt], lattice, position_dtype=position.dtype
    )
    coefficients *= lattice.fluid_weight[:, None]
    target *= (lattice.fluid_weight * lattice.mesh_weight)[:, None]
    support = lattice.fluid_weight * ~lattice.solid_interior
    cap = coupling["transfer_amplification_cap"]
    sigma = h * viscous["core_radius_ratio"]
    blend = blend_represented_state(
        coefficients,
        target,
        lattice.fvm_blend_weight,
        lattice.shape,
        h,
        core_radius=sigma,
        amplification_cap=cap,
        output_weight=support,
        dimensions=2,
    )
    represented = gaussian_represented_vortex_strength(
        coefficients, lattice.shape, h, core_radius=sigma, dimensions=2
    )
    mismatch = lattice.fvm_blend_weight[:, None] * (target - represented)
    mismatch[support <= 0] = 0
    mapped_mismatch = gaussian_represented_vortex_strength(
        mismatch, lattice.shape, h, core_radius=sigma, dimensions=2
    )
    proposal = coefficients * support[:, None] + support[:, None] * (
        mismatch + min(cap - 1, 1) * (mismatch - mapped_mismatch)
    )
    target_maximum = np.linalg.norm(blend.physical_target, axis=1).max()
    bounds = np.maximum(
        np.linalg.norm(coefficients * support[:, None], axis=1), cap * target_maximum
    )
    limited = np.linalg.norm(proposal, axis=1) > bounds
    wall_distance = boundary.signed_distance(lattice.positions)
    wall_fluid = (support > 0) & (lattice.mesh_weight > 0) & (wall_distance < 0.10)
    full_fluid = (support > 0) & (lattice.mesh_weight > 0)
    # This count observes the native guard; no unlimited proposal is committed.
    guard = {
        "physical_target_maximum_strength": float(target_maximum),
        "coefficient_bound_from_target": float(cap * target_maximum),
        "maximum_native_amplification": blend.maximum_amplification,
        "limited_node_count": int(np.count_nonzero(limited & full_fluid)),
        "limited_wall_node_count": int(np.count_nonzero(limited & wall_fluid)),
        "wall_node_count": int(np.count_nonzero(wall_fluid)),
        "maximum_uncommitted_proposal_over_bound": float(
            np.max(np.linalg.norm(proposal[full_fluid], axis=1) / bounds[full_fluid])
        ),
    }
    n = np.arange(-int(np.ceil(6 * sigma / h)), int(np.ceil(6 * sigma / h)) + 1)
    weights = h / (np.sqrt(np.pi) * sigma) * np.exp(-((n * h / sigma) ** 2))

    def symbol(theta):
        return np.cos(np.asarray(theta)[..., None] * n) @ weights

    modes = []
    for wavelength in (16, 8, 4, 2):
        for direction in ("axis", "diagonal"):
            value = float(symbol(2 * np.pi / wavelength))
            if direction == "diagonal":
                value *= value
            modes.append(
                {
                    "wavelength_per_axis_in_particle_spacings": wavelength,
                    "direction": direction,
                    "represented_amplitude_ratio": value,
                    "unconstrained_exact_inverse_gain": 1 / value,
                    "first_native_correction_amplitude_ratio": value
                    * (1 + min(cap - 1, 1) * (1 - value)),
                }
            )
    nx, ny, _ = lattice.shape
    kernel_symbol = (
        symbol(2 * np.pi * np.fft.fftfreq(nx))[:, None]
        * symbol(2 * np.pi * np.fft.fftfreq(ny))[None, :]
    )
    fft_target = np.fft.fft2((target[:, 2] * (support > 0)).reshape(nx, ny))
    power = np.abs(fft_target) ** 2
    bands = {}
    for lower, upper in ((0, 0.2), (0.2, 0.5), (0.5, 0.9), (0.9, 1.01)):
        selected = (kernel_symbol >= lower) & (kernel_symbol < upper)
        bands[f"gaussian_symbol_{lower:g}_to_{upper:g}"] = float(
            power[selected].sum() / power.sum()
        )
    deep_solid = wall_distance < -2 * h
    solid_leakage = {
        "deep_solid_node_count": int(np.count_nonzero(deep_solid)),
        "deep_solid_represented_curl_rms": float(
            np.sqrt(
                np.mean(
                    (blend.represented_vortex_strength[deep_solid, 2] / lattice.particle_volume)
                    ** 2
                )
            )
        ),
        "deep_solid_represented_strength_l1": float(
            np.abs(blend.represented_vortex_strength[deep_solid, 2]).sum()
        ),
    }
    wall_faces = np.concatenate(
        [
            np.arange(patch["start_face"], patch["start_face"] + patch["n_faces"])
            for patch in mesh["boundary"]
            if patch["type"] == "wall"
        ]
    )
    normals = geometry["face_area_vector"][wall_faces] / geometry["face_area"][wall_faces, None]
    distance = np.abs(
        np.einsum(
            "fi,fi->f",
            normals,
            geometry["face_centre"][wall_faces]
            - geometry["cell_centre"][mesh["owners"][wall_faces]],
        )
    )
    report = {
        "physical_time": metadata["time"],
        "physical_time_advanced": 0,
        "input_sha256": hashes,
        "gaussian_source_sha256": digest(
            Path(__file__).resolve().parents[3] / "source/coupler/stable_renewal.py"
        ),
        "sigma_over_particle_spacing": sigma / h,
        "native_guard": guard,
        "homogeneous_mode_response": modes,
        "target_fourier_power_fraction": bands,
        "solid_kernel_leakage": solid_leakage,
        "wall_owner_normal_distance_min_median_max": [
            float(distance.min()),
            float(np.median(distance)),
            float(distance.max()),
        ],
        "sigma_over_median_wall_owner_distance": sigma / float(np.median(distance)),
        "interpretation": "Homogeneous mode symbols are exact for the native sampled Gaussian; the finite target spectrum and saturated-node counts are diagnostic. Walls/blending couple modes. Deep-solid represented curl describes Gaussian tails, not particle centres inside the body. Complete no-slip zero extension is solenoidal and needs no extra potential velocity term. This does not prove evolved force causality; unchanged-guard mapped refinement and actual outer-face field measurements provide the separate geometric evidence.",
    }
    if any(digest(path) != hashes[str(path)] for path in input_paths):
        raise RuntimeError("Frozen inputs changed during Gaussian transfer audit")
    destination = directory / "gaussian_transfer_bandlimit.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(audit(parser.parse_args().directory))
