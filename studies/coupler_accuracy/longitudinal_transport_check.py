"""Check longitudinal transport and Gaussian filtering on a periodic 3D field.

The box has side 2*pi m. Its finite Fourier series are resolved without product
aliasing on either grid. This checks continuum identities and a uniform-core
particle-density limit; it does not advance particles or measure the cube.

Usage: python studies/coupler_accuracy/longitudinal_transport_check.py --output PATH
The JSON output must not already exist. No simulation state is read or modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
SPATIAL_AXES = (0, 1, 2)


def _multiply_fourier(field: FloatArray, factor: np.ndarray) -> FloatArray:
    """Apply a scalar Fourier multiplier on the first three axes, returning a copy."""
    extension = (None,) * (field.ndim - 3)
    transformed = np.fft.fftn(field, axes=SPATIAL_AXES)
    return np.fft.ifftn(transformed * factor[(..., *extension)], axes=SPATIAL_AXES).real


def _gradient(field: FloatArray, wavevector: FloatArray) -> FloatArray:
    """Return partial_j field_i with derivative index last; lengths are in m."""
    return np.stack([_multiply_fourier(field, 1j * wavevector[..., j]) for j in range(3)], axis=-1)


def _curl(field: FloatArray, wavevector: FloatArray) -> FloatArray:
    """Return the spectral curl of a vector field with component index last."""
    jacobian = _gradient(field, wavevector)
    return np.stack(
        (
            jacobian[..., 2, 1] - jacobian[..., 1, 2],
            jacobian[..., 0, 2] - jacobian[..., 2, 0],
            jacobian[..., 1, 0] - jacobian[..., 0, 1],
        ),
        axis=-1,
    )


def _induce(field: FloatArray, wavevector: FloatArray) -> FloatArray:
    """Return zero-mean periodic Biot-Savart velocity, removing gradients."""
    squared_wavenumber = np.sum(wavevector**2, axis=-1)
    inverse = np.zeros_like(squared_wavenumber)
    np.divide(1.0, squared_wavenumber, out=inverse, where=squared_wavenumber > 0)
    return _multiply_fourier(_curl(field, wavevector), inverse)


def _rms(field: FloatArray) -> float:
    """Return vector RMS over the three-dimensional box, without component averaging."""
    return float(np.sqrt(np.mean(np.sum(field**2, axis=-1))))


def _check_grid(count: int, core_radius: float) -> dict[str, float | int]:
    """Check exact identities and measure finite-core residuals on one periodic grid.

    Parameters
    ----------
    count : int
        Number of equally spaced nodes on each axis. Both prescribed values,
        24 and 32, resolve all input and product modes.
    core_radius : float
        Radius in m of exp(-r^2/sigma^2)/(pi^(3/2)*sigma^3). Uniform Gaussian
        convolution has multiplier exp(-sigma^2*|k|^2/4).

    Returns
    -------
    dict
        Independent scalar evidence. Velocity-rate RMS entries are in m/s^2;
        vorticity-rate identity errors are in 1/s^2. No particle quadrature,
        boundaries, time integration, viscosity, or production kernels enter.
    """
    coordinate = np.arange(count) * (2 * np.pi / count)
    x, y, z = np.meshgrid(coordinate, coordinate, coordinate, indexing="ij")
    frequencies = np.fft.fftfreq(count, d=1 / count)
    wavevector = np.stack(np.meshgrid(frequencies, frequencies, frequencies, indexing="ij"), -1)
    squared_wavenumber = np.sum(wavevector**2, axis=-1)
    vector_potential = np.stack(
        (0.7 * np.cos(y + 2 * z), 0.4 * np.sin(2 * x - z), np.sin(x + 2 * y)), -1
    )
    velocity = _curl(vector_potential, wavevector)
    jacobian = _gradient(velocity, wavevector)
    physical_vorticity = _curl(velocity, wavevector)
    scalar_potential = 0.3 * np.sin(x + y + z) + 0.2 * np.cos(2 * x - y + z)
    longitudinal = _gradient(scalar_potential, wavevector)
    represented = physical_vorticity + longitudinal

    def positive_transport(field: FloatArray) -> FloatArray:
        """Evaluate -u_j partial_j field_i + partial_i u_j field_j, in 1/s^2."""
        return -np.einsum("...j,...ij->...i", velocity, _gradient(field, wavevector)) + transpose(
            field
        )

    def transpose(field: FloatArray) -> FloatArray:
        """Contract J_ji field_j using the gradient of this same velocity."""
        return np.einsum("...ji,...j->...i", jacobian, field)

    euler_rate = _curl(np.cross(velocity, physical_vorticity), wavevector)
    corrected_resolved_rate = positive_transport(represented) - 2 * transpose(longitudinal)
    potential_rate = -_gradient(np.sum(velocity * longitudinal, axis=-1), wavevector)
    identity_error = _rms(corrected_resolved_rate - euler_rate - potential_rate)
    resolved_velocity_error = _rms(_induce(corrected_resolved_rate - euler_rate, wavevector))
    np.testing.assert_allclose(
        corrected_resolved_rate, euler_rate + potential_rate, rtol=0.0, atol=1e-10
    )
    np.testing.assert_allclose(_induce(represented, wavevector), velocity, rtol=0.0, atol=1e-11)

    gaussian = np.exp(-(core_radius**2) * squared_wavenumber / 4)
    # The prescribed input series has |k|^2 <= 6. Invert only those known modes;
    # amplifying absent high-frequency roundoff is not a deconvolution test.
    inverse_known_modes = np.where(squared_wavenumber <= 6, 1 / gaussian, 0)
    transverse_density = _multiply_fourier(physical_vorticity, inverse_known_modes)
    longitudinal_density = _multiply_fourier(longitudinal, inverse_known_modes)
    density = transverse_density + longitudinal_density
    np.testing.assert_allclose(
        _multiply_fourier(density, gaussian), represented, rtol=0.0, atol=1e-11
    )

    transverse_rate = positive_transport(transverse_density)
    native_rate = positive_transport(density)
    corrected_density_rate = native_rate - 2 * transpose(longitudinal_density)
    naive_density_rate = native_rate - 2 * transpose(longitudinal)

    def visible_rate(field: FloatArray) -> FloatArray:
        """Map a coefficient-density rate to velocity rate with the fixed Gaussian core."""
        return _induce(_multiply_fourier(field, gaussian), wavevector)

    native_leak = visible_rate(native_rate - transverse_rate)
    corrected_leak = visible_rate(corrected_density_rate - transverse_rate)
    naive_leak = visible_rate(naive_density_rate - transverse_rate)
    np.testing.assert_allclose(corrected_leak, 0.0, rtol=0.0, atol=1e-11)
    finite_core_error = visible_rate(transverse_rate) - _induce(euler_rate, wavevector)
    commutator = _multiply_fourier(native_rate, gaussian) - positive_transport(represented)
    lifted_resolved_rate = _multiply_fourier(native_rate, gaussian) - 2 * transpose(longitudinal)
    np.testing.assert_allclose(
        lifted_resolved_rate - corrected_resolved_rate, commutator, rtol=0.0, atol=1e-11
    )

    # The exact curl update is compared with its deliberately incomplete
    # eta*curl(delta_u) counterpart, using the same smooth donor mismatch.
    mismatch = np.stack((0.2 * np.sin(y), 0.1 * np.sin(z), 0.15 * np.sin(x)), -1)
    authority = 0.5 + 0.25 * np.cos(x + y - z)
    curl_update = _curl(authority[..., None] * mismatch, wavevector)
    expanded_update = authority[..., None] * _curl(mismatch, wavevector) + np.cross(
        _gradient(authority, wavevector), mismatch
    )
    np.testing.assert_allclose(curl_update, expanded_update, rtol=0.0, atol=1e-11)
    curl_divergence = np.trace(_gradient(curl_update, wavevector), axis1=-2, axis2=-1)
    incomplete = authority[..., None] * _curl(mismatch, wavevector)
    incomplete_divergence = np.trace(_gradient(incomplete, wavevector), axis1=-2, axis2=-1)
    np.testing.assert_allclose(curl_divergence, 0.0, rtol=0.0, atol=1e-11)

    return {
        "grid_count_per_axis": count,
        "core_radius_m": core_radius,
        "resolved_vorticity_rate_identity_rms": identity_error,
        "resolved_corrected_velocity_rate_error_rms": resolved_velocity_error,
        "native_longitudinal_velocity_rate_rms": _rms(native_leak),
        "coefficient_corrected_longitudinal_velocity_rate_rms": _rms(corrected_leak),
        "naive_resolved_density_substitution_velocity_rate_rms": _rms(naive_leak),
        "transverse_only_finite_core_velocity_rate_error_rms": _rms(finite_core_error),
        "visible_transport_filter_commutator_rms": _rms(_induce(commutator, wavevector)),
        "curl_transfer_divergence_maximum": float(np.max(np.abs(curl_divergence))),
        "incomplete_vorticity_blend_divergence_rms": float(
            np.sqrt(np.mean(incomplete_divergence**2))
        ),
    }


def main() -> None:
    """Run bounded manufactured checks and write one new machine-readable JSON file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = [_check_grid(count, sigma) for count in (24, 32) for sigma in (0.4, 0.2, 0.1)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"scope": __doc__, "records": records}, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
