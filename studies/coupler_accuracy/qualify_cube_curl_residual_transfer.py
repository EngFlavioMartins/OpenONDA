"""Qualify the study curl-residual increment on a compact, fully 3D field.

Run from the repository with:
python -m studies.coupler_accuracy.qualify_cube_curl_residual_transfer --output PATH

No simulation is advanced. The test uses analytical velocity/curl expressions,
the production Gaussian field operators, and complete compact support. Output
is a new JSON file; failed numerical gates retain the measured evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from source.coupler.renewal_projection import (
    gaussian_vorticity_basis,
    gaussian_vorticity_divergence_operator,
)
from source.coupler.stable_renewal import gaussian_represented_vortex_strength
from studies.coupler_accuracy.cube_curl_residual_transfer import curl_residual_correction

FloatArray = NDArray[np.float64]


def _velocity(position: FloatArray) -> FloatArray:
    """Return a 3D incompressible donor mismatch in m/s, with coordinates in m."""
    x, y, z = position.T
    return np.column_stack((np.sin(y) + 0.2 * z, np.cos(z) + 0.3 * x, np.sin(x) + 0.4 * y))


def _authority(position: FloatArray) -> FloatArray:
    """Return a C3 polynomial authority with support in the 1.6-m-side cube."""
    polynomial = np.maximum(1 - (position / 0.8) ** 2, 0.0)
    return np.prod(polynomial**4, axis=1)


def _fluid_weight(position: FloatArray) -> FloatArray:
    """Return a C2 spherical mask, zero through radius 0.3 m and one after 0.65 m."""
    phase = np.clip((np.linalg.norm(position, axis=1) - 0.3) / 0.35, 0.0, 1.0)
    # Evaluate the same quintic from its nearer endpoint so cancellation
    # cannot produce a value slightly greater than one near the outer edge.
    reduced = np.minimum(phase, 1 - phase)
    value = reduced**3 * (10 - 15 * reduced + 6 * reduced**2)
    return np.where(phase <= 0.5, value, 1 - value)


def _exact_curl(position: FloatArray) -> FloatArray:
    """Differentiate authority*fluid_weight*velocity analytically, in 1/s."""
    x, y, z = position.T
    curl_velocity = np.column_stack((0.4 + np.sin(z), 0.2 - np.cos(x), 0.3 - np.cos(y)))
    polynomial = np.maximum(1 - (position / 0.8) ** 2, 0.0)
    factors = polynomial**4
    factor_derivative = -8 * position / 0.8**2 * polynomial**3
    authority_gradient = np.column_stack(
        [
            factor_derivative[:, j] * np.prod(factors[:, [k for k in range(3) if k != j]], axis=1)
            for j in range(3)
        ]
    )
    radius = np.linalg.norm(position, axis=1)
    phase = np.clip((radius - 0.3) / 0.35, 0.0, 1.0)
    radial_derivative = 30 * phase**2 * (1 - phase) ** 2 / 0.35
    gradient_scale = np.zeros_like(radius)
    np.divide(radial_derivative, radius, out=gradient_scale, where=radius > 0)
    fluid_gradient = gradient_scale[:, None] * position
    weight_gradient = (
        _fluid_weight(position)[:, None] * authority_gradient
        + _authority(position)[:, None] * fluid_gradient
    )
    return (_authority(position) * _fluid_weight(position))[:, None] * curl_velocity + np.cross(
        weight_gradient, _velocity(position)
    )


def _centered_divergence(field: FloatArray, spacing: float) -> FloatArray:
    """Take the matching nodal centered divergence; the complete halo is zero."""
    result = np.zeros(field.shape[:3], dtype=np.float64)
    for axis in range(3):
        result += (np.roll(field[..., axis], -1, axis) - np.roll(field[..., axis], 1, axis)) / (
            2 * spacing
        )
    return result


def _vector_rms(field: FloatArray) -> float:
    """Return RMS Euclidean magnitude over equally weighted samples."""
    return float(np.sqrt(np.mean(np.sum(field**2, axis=-1))))


def _compact_lattice(spacing: float) -> tuple[FloatArray, tuple[int, int, int]]:
    """Return complete C-order support for the compact field and one scalar inverse."""
    # Keep complete curl and six-core inverse-filter support, without refining
    # physically empty space that neither operator can reach.
    half_count = int(np.ceil(0.8 / spacing)) + 10
    count = 2 * half_count + 1
    axis = np.asarray(spacing * np.arange(-half_count, half_count + 1), dtype=np.float64)
    position = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    shape = (count, count, count)
    return position, shape


def _case(spacing: float, probe: FloatArray) -> dict:
    """Measure identity, discrete curl, Gaussian reconstruction and mask effects."""
    position, shape = _compact_lattice(spacing)
    original_position = position.copy()
    radius = 1.1 * spacing
    parameters = {
        "authority_at": _authority,
        "fluid_weight_at": _fluid_weight,
        "core_radius": radius,
    }
    identity = curl_residual_correction(
        position,
        shape,
        spacing,
        fvm_velocity_at=_velocity,
        vpm_velocity_at=_velocity,
        **parameters,
    )
    correction = curl_residual_correction(
        position,
        shape,
        spacing,
        fvm_velocity_at=_velocity,
        vpm_velocity_at=np.zeros_like,
        **parameters,
    )
    inverse = curl_residual_correction(
        position,
        shape,
        spacing,
        fvm_velocity_at=_velocity,
        vpm_velocity_at=np.zeros_like,
        deconvolution_steps=1,
        **parameters,
    )
    exact = _exact_curl(position)
    interior = np.all(np.abs(position) <= 1.0 + 1e-12, axis=1)
    vorticity = correction / spacing**3
    divergence = _centered_divergence(vorticity.reshape(*shape, 3), spacing)
    inverse_divergence = _centered_divergence((inverse / spacing**3).reshape(*shape, 3), spacing)
    masked = vorticity * _fluid_weight(position)[:, None]
    masked_divergence = _centered_divergence(masked.reshape(*shape, 3), spacing)
    represented = gaussian_represented_vortex_strength(
        correction, shape, spacing, core_radius=radius
    )
    represented_inverse = gaussian_represented_vortex_strength(
        inverse, shape, spacing, core_radius=radius
    )

    # Exact-zero coefficients are omitted; no nonzero strength is pruned.
    active = np.any(correction != 0.0, axis=1)
    active_position, active_strength = position[active], correction[active]
    physical = np.empty_like(probe)
    continuous_divergence = np.empty(len(probe))
    for start in range(0, len(probe), 8):
        selection = slice(start, start + 8)
        physical[selection] = (
            gaussian_vorticity_basis(probe[selection], active_position, radius) @ active_strength
        )
        continuous_divergence[selection] = (
            gaussian_vorticity_divergence_operator(probe[selection], active_position, radius)
            @ active_strength.ravel()
        )
    body = np.linalg.norm(position, axis=1) <= 0.1 + 1e-14
    return {
        "particle_spacing_m": spacing,
        "core_radius_m": radius,
        "lattice_shape": shape,
        "active_correction_count": int(np.count_nonzero(active)),
        "identity_max_strength_m3_per_s": float(np.max(np.abs(identity))),
        "input_coordinates_unchanged": np.array_equal(position, original_position),
        "curl_vorticity_rms_error_per_s": _vector_rms((vorticity - exact)[interior]),
        "matching_divergence_max_per_m_s": float(np.max(np.abs(divergence))),
        "inverse_matching_divergence_max_per_m_s": float(np.max(np.abs(inverse_divergence))),
        "continuous_gaussian_vorticity_rms_error_per_s": _vector_rms(physical - _exact_curl(probe)),
        "continuous_gaussian_divergence_rms_per_m_s": float(
            np.sqrt(np.mean(continuous_divergence**2))
        ),
        "post_curl_mask_divergence_rms_per_m_s": float(np.sqrt(np.mean(masked_divergence**2))),
        "raw_reconstruction_residual_l2_m3_per_s": float(np.linalg.norm(represented - correction)),
        "inverse_reconstruction_residual_l2_m3_per_s": float(
            np.linalg.norm(represented_inverse - correction)
        ),
        "total_strength_m3_per_s": correction.sum(axis=0).tolist(),
        "forbidden_body_max_raw_strength_m3_per_s": float(np.max(np.abs(correction[body]))),
        "forbidden_body_max_inverse_strength_m3_per_s": float(np.max(np.abs(inverse[body]))),
    }


def _fixed_core_quadrature(probe: FloatArray) -> dict:
    """Refine h at fixed sigma=0.11 m against exact zero continuous divergence.

    Unlike fixed sigma/h, this path makes h/sigma tend to zero and suppresses
    derivative aliases. The continuum target is curl of a Gaussian-filtered
    velocity residual, whose divergence is exactly zero at every fixed core.
    No claim of vanishing filter bias relative to unfiltered vorticity follows.
    """
    radius = 0.11
    records = []
    for spacing in (0.1, 0.05, 0.025):
        position, shape = _compact_lattice(spacing)
        correction = curl_residual_correction(
            position,
            shape,
            spacing,
            fvm_velocity_at=_velocity,
            vpm_velocity_at=np.zeros_like,
            authority_at=_authority,
            fluid_weight_at=_fluid_weight,
        )
        active = np.any(correction != 0.0, axis=1)
        source_position, source_strength = position[active], correction[active]
        divergence = np.empty(len(probe))
        for start in range(0, len(probe), 8):
            selection = slice(start, start + 8)
            divergence[selection] = (
                gaussian_vorticity_divergence_operator(probe[selection], source_position, radius)
                @ source_strength.ravel()
            )
        records.append(
            {
                "particle_spacing_m": spacing,
                "core_radius_m": radius,
                "spacing_over_core": spacing / radius,
                "continuous_gaussian_divergence_rms_per_m_s": float(
                    np.sqrt(np.mean(divergence**2))
                ),
            }
        )
    orders = [
        float(
            np.log2(
                a["continuous_gaussian_divergence_rms_per_m_s"]
                / b["continuous_gaussian_divergence_rms_per_m_s"]
            )
        )
        for a, b in zip(records[:-1], records[1:], strict=True)
    ]
    return {
        "description": _fixed_core_quadrature.__doc__,
        "records": records,
        "observed_orders": orders,
        "passed": orders[-1] > 1.7,
    }


def main() -> None:
    """Run five spatial resolutions, retain measured evidence, then enforce qualification gates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixed-core-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    probe = np.random.default_rng(20260915).uniform(-0.7, 0.7, size=(64, 3))
    if args.fixed_core_only:
        with threadpool_limits(limits=2):
            quadrature = _fixed_core_quadrature(probe)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(quadrature, stream, indent=2)
            stream.write("\n")
        if not quadrature["passed"]:
            raise AssertionError("Fixed-core Gaussian divergence has not reached its order gate")
        return
    with threadpool_limits(limits=2):
        records = [_case(spacing, probe) for spacing in (0.2, 0.1, 0.05, 0.025, 0.0125)]
    orders = {
        name: [
            float(np.log2(a[name] / b[name]))
            for a, b in zip(records[:-1], records[1:], strict=True)
        ]
        for name in (
            "curl_vorticity_rms_error_per_s",
            "continuous_gaussian_vorticity_rms_error_per_s",
            "continuous_gaussian_divergence_rms_per_m_s",
        )
    }
    checks = {
        "exact_matched_velocity_identity": all(
            row["identity_max_strength_m3_per_s"] == 0 for row in records
        ),
        "read_only_input": all(row["input_coordinates_unchanged"] for row in records),
        "matching_discrete_divergence": all(
            row["matching_divergence_max_per_m_s"] < 2e-11 for row in records
        ),
        "scalar_inverse_preserves_discrete_divergence": all(
            row["inverse_matching_divergence_max_per_m_s"] < 2e-11 for row in records
        ),
        "second_order_curl": orders["curl_vorticity_rms_error_per_s"][-1] > 1.7,
        "gaussian_vorticity_converges": orders["continuous_gaussian_vorticity_rms_error_per_s"][-1]
        > 1.5,
        "gaussian_divergence_converges": orders["continuous_gaussian_divergence_rms_per_m_s"][-1]
        > 1.5,
        "inverse_reduces_reconstruction_error": all(
            row["inverse_reconstruction_residual_l2_m3_per_s"]
            < row["raw_reconstruction_residual_l2_m3_per_s"]
            for row in records
        ),
        "solid_guard_preserves_raw_support": all(
            row["forbidden_body_max_raw_strength_m3_per_s"] == 0 for row in records
        ),
        "post_curl_mask_negative_control": all(
            row["post_curl_mask_divergence_rms_per_m_s"] > 1e-5 for row in records
        ),
    }
    report = {"scope": __doc__, "records": records, "observed_orders": orders, "checks": checks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"Failed curl-residual qualifications: {failed}")


if __name__ == "__main__":
    main()
