"""Assess a compact potential-flux source on the saved Cartesian t6 cloud.

This is a frozen-lattice control, not an RK implementation. The potential uses
free-space Gaussian induction with its gauge fixed at infinity. A fixed 0.18 m
taper vanishes at the occupied-support boundary, including holes and the body.
Internal face fluxes are applied once with opposite signs to their two cells.
The current moving RK particles do not supply this Cartesian cell geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from math import erf
from pathlib import Path

from cube_wake_nullspace_probe import direct_velocity_rate
from cube_wake_operator_audit import curl, reflection_asymmetry
from cube_wake_particle_probe import direct_gaussian, rms
from numba import njit, prange, set_num_threads
import numpy as np
from scipy import fft
from scipy.ndimage import distance_transform_edt
from scipy.signal import fftconvolve
from scipy.special import erf as array_erf
from threadpoolctl import threadpool_limits


@njit(cache=True, parallel=True)
def direct_potential(points, position, strength, sigma):
    """Independently sum psi=Sigma f_sigma(r) Gamma dot r in free space."""
    result = np.zeros(len(points))
    for i in prange(len(points)):
        value = 0.0
        for j in range(len(position)):
            dx = points[i, 0] - position[j, 0]
            dy = points[i, 1] - position[j, 1]
            dz = points[i, 2] - position[j, 2]
            squared = dx * dx + dy * dy + dz * dz
            if squared > 0:
                radius = np.sqrt(squared)
                scaled = radius / sigma[j]
                cutoff = erf(scaled) - 2 / np.sqrt(np.pi) * scaled * np.exp(-(scaled**2))
                factor = cutoff / (4 * np.pi * radius * squared)
                value += factor * (strength[j, 0] * dx + strength[j, 1] * dy + strength[j, 2] * dz)
        result[i] = value
    return result


def potential_on_lattice(coefficient, spacing, core_radius):
    """Use linear convolution with the sampled free-space Gaussian potential kernel."""
    shape = coefficient.shape[:3]
    axes = [(np.arange(2 * n - 1) - (n - 1)) * spacing for n in shape]
    dx, dy, dz = axes[0][:, None, None], axes[1][None, :, None], axes[2][None, None, :]
    radius = np.sqrt(dx * dx + dy * dy + dz * dz)
    scaled = radius / core_radius
    numerator = array_erf(scaled) - 2 / np.sqrt(np.pi) * scaled * np.exp(-(scaled**2))
    factor = np.zeros_like(radius)
    np.divide(numerator, 4 * np.pi * radius**3, out=factor, where=radius > 0)
    potential = np.zeros(shape)
    with fft.set_workers(2):
        for component, coordinate in enumerate((dx, dy, dz)):
            potential += fftconvolve(coefficient[..., component], factor * coordinate, mode="same")
    return potential


def potential_flux_source(phi, jacobian, index, identifiers, spacing):
    """Apply opposite integral rates on shared faces of frozen Cartesian cells.

    ``phi`` and ``jacobian`` are active-cell samples. An identifier of -1
    marks an absent cell; the supplied compact potential must taper to zero
    at those support boundaries. Ghost padding is required around the map.
    Returned gradients are diagnostic FV differences, not exact derivatives.
    """
    source = np.zeros((len(phi), 3))
    gradient_phi = np.zeros_like(source)
    divergence_transpose_j = np.zeros_like(source)
    links = 0
    for axis in range(3):
        other = index.copy()
        other[:, axis] += 1
        neighbour = identifiers[tuple(other.T)]
        left = np.flatnonzero(neighbour >= 0)
        right = neighbour[left]
        face_phi = 0.5 * (phi[left] + phi[right])
        face_row = 0.5 * (jacobian[left, axis, :] + jacobian[right, axis, :])
        flux = -2 * spacing**2 * face_phi[:, None] * face_row
        np.add.at(source, left, flux)
        np.add.at(source, right, -flux)
        np.add.at(gradient_phi[:, axis], left, face_phi / spacing)
        np.add.at(gradient_phi[:, axis], right, -face_phi / spacing)
        np.add.at(divergence_transpose_j, left, face_row / spacing)
        np.add.at(divergence_transpose_j, right, -face_row / spacing)
        links += len(left)
    return source, gradient_phi, divergence_transpose_j, links


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    field_path = args.audit / "fields_t6.npz"
    with np.load(field_path) as data:
        fields = {key: data[key].astype(float) for key in data.files}
    with np.load(args.audit / "particles_accepted_t6.npz") as data:
        sigma = data["core_radius"].astype(float)
    np.testing.assert_array_equal(sigma, np.full(len(sigma), sigma[0]))
    position, strength = fields["particle_position"], fields["particle_strength"]
    jacobian = fields["stage_gradient"]
    spacing, width = 0.06, 0.18
    origin = 0.03 + spacing * (np.floor((position.min(axis=0) - 0.03) / spacing) - 1)
    index = np.rint((position - origin) / spacing).astype(int)
    snap = origin + index * spacing
    if np.max(np.abs(position - snap)) > 3e-7:
        raise ValueError("This control requires the saved Cartesian GBD state")
    shape = tuple((index.max(axis=0) + 2).tolist())
    coefficient = np.zeros((*shape, 3))
    coefficient[tuple(index.T)] = strength
    identifiers = np.full(shape, -1, dtype=int)
    identifiers[tuple(index.T)] = np.arange(len(position))
    active = identifiers >= 0
    potential = potential_on_lattice(coefficient, spacing, float(sigma[0]))
    psi = potential[tuple(index.T)]
    selected = np.linspace(0, len(position) - 1, 48, dtype=int)
    independent = direct_potential(position[selected], position, strength, sigma)
    potential_error = rms((psi[selected] - independent)[:, None])
    if potential_error > 2e-6:
        raise ValueError(f"Free-space potential convolution disagrees by {potential_error}")

    distance = np.maximum(distance_transform_edt(active, sampling=spacing) - spacing / 2, 0)
    taper = 0.5 * (1 - np.cos(np.pi * np.clip(distance / width, 0, 1)))
    phi = taper[tuple(index.T)] * psi
    source, gradient_phi, divergence_transpose_j, links = potential_flux_source(
        phi, jacobian, index, identifiers, spacing
    )
    nodal_localized = -2 * spacing**3 * np.einsum("nji,nj->ni", jacobian, gradient_phi)
    points = fields["points"]
    native = direct_velocity_rate(
        points, position, strength, sigma, fields["stage_velocity"], fields["stage_strength_rate"]
    )
    correction_velocity, _ = direct_gaussian(points, position, source, sigma)
    product_difference_velocity, _ = direct_gaussian(
        points, position, source - nodal_localized, sigma
    )
    asymmetry = reflection_asymmetry(points, fields["accepted_velocity"])
    regions = {}
    for name, mask in {
        "renewal_seam": (points[:, 0] >= 1.25) & (points[:, 0] <= 1.62),
        "outer_wake": points[:, 0] > 1.62,
    }.items():
        regions[name] = {}
        for label, rate in {
            "native": native,
            "compact_flux": native + correction_velocity,
            "flux_minus_nodal_product": product_difference_velocity,
        }.items():
            odd = reflection_asymmetry(points, rate)
            regions[name][label] = {
                "velocity_rate_rms": rms(rate[mask]),
                "reflection_energy_derivative": float(
                    2 * np.mean(np.sum(asymmetry[mask] * odd[mask], axis=1))
                ),
            }
    report = {
        "scope": __doc__,
        "input_sha256": hashlib.sha256(field_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "particle_count": len(position),
        "internal_faces": links,
        "spacing": spacing,
        "taper_width": width,
        "potential_gauge": "Free-space potential tends to zero at infinity; no periodic substitution",
        "potential_direct_error_rms": potential_error,
        "source_net_strength_rate": source.sum(axis=0).tolist(),
        "source_strength_rate_l1": float(np.linalg.norm(source, axis=1).sum()),
        "source_impulse_rate_per_density": (0.5 * np.cross(position, source).sum(axis=0)).tolist(),
        "midpoint_localized_impulse_target": (
            -(spacing**3) * np.sum(phi[:, None] * curl(jacobian), axis=0)
        ).tolist(),
        "regions": regions,
        "stage_geometry_qualification": "Frozen lattice only; no moving-stage or refinement qualification",
    }
    np.savez_compressed(
        args.output / "fields.npz",
        position=position,
        potential=psi,
        localized_potential=phi,
        source=source,
        nodal_localized_source=nodal_localized,
        divergence_transpose_j=divergence_transpose_j,
        points=points,
        correction_velocity_rate=correction_velocity,
    )
    (args.output / "control.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    set_num_threads(2)
    with threadpool_limits(limits=2):
        run(args)
