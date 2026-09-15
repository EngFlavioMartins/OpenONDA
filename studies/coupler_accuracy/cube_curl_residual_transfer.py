"""Study-only curl residual on a complete uniform renewal lattice.

This operator returns an incremental particle strength. Apply it after a
coefficient-preserving remap and its cleanup, evaluating the VPM velocity on
that resulting baseline. Add the increment on its complete support; subsequent
masking, pruning, or invariant repair would change the tested operator.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from source.coupler.stable_renewal import (
    gaussian_represented_vortex_strength,
    vortex_strength_from_velocity_trace,
)

FloatArray = NDArray[np.float64]
VectorTrace = Callable[[FloatArray], FloatArray]
ScalarTrace = Callable[[FloatArray], FloatArray]

__all__ = ["curl_residual_correction"]


def _weight_at(callback: ScalarTrace, position: FloatArray, name: str) -> FloatArray:
    """Validate one dimensionless, shape-(N,) authority or fluid-weight sample."""
    weight = np.asarray(callback(position), dtype=np.float64)
    if weight.shape != (len(position),):
        raise ValueError(f"{name} must return shape ({len(position)},)")
    if not np.isfinite(weight).all() or np.any((weight < 0.0) | (weight > 1.0)):
        raise ValueError(f"{name} must return finite weights between zero and one")
    return weight


def _velocity_at(callback: VectorTrace, position: FloatArray, name: str) -> FloatArray:
    """Validate one shape-(N,3) velocity sample in the common world frame, in m/s."""
    velocity = np.asarray(callback(position), dtype=np.float64)
    if velocity.shape != position.shape or not np.isfinite(velocity).all():
        raise ValueError(f"{name} must return finite velocities with shape {position.shape}")
    return velocity


def curl_residual_correction(
    positions: FloatArray,
    shape: tuple[int, int, int],
    particle_spacing: float,
    *,
    fvm_velocity_at: VectorTrace,
    vpm_velocity_at: VectorTrace,
    authority_at: ScalarTrace,
    fluid_weight_at: ScalarTrace | None = None,
    core_radius: float | None = None,
    deconvolution_steps: int = 0,
) -> FloatArray:
    """Form delta-Gamma = h^3 curl_h[eta m (u_F-u_V)] on a complete lattice.

    Parameters
    ----------
    positions : numpy.ndarray, shape (N, 3)
        Finite float64 node coordinates in m, in C order for ``shape`` with
        z varying fastest. The full uniform Cartesian support is required,
        including nodes with zero baseline strength. Inputs are read only.
    shape : tuple of three int
        Lattice counts (nx, ny, nz), each positive, with N = nx*ny*nz.
    particle_spacing : float
        Positive spacing h in m, common to all axes; node volume is h^3.
    fvm_velocity_at, vpm_velocity_at : callable
        Functions from shape-(N,3) query coordinates to shape-(N,3) finite
        velocities in m/s. Use the same world frame, harmonic/background
        convention and physical time. VPM must describe the post-remap
        baseline, including preserved outer particles and body contribution.
        Each callback receives six arrays, in order x+h, x-h, y+h, y-h, z+h,
        z-h relative to ``positions``. Sampling must be deterministic.
    authority_at : callable
        Dimensionless shape-(N,) weights eta in [0,1], evaluated at those
        same query points. Include donor-support authority in this function.
    fluid_weight_at : callable or None, default=None
        Optional dimensionless shape-(N,) weights m in [0,1], sampled inside
        the curl. None means m=1. A forbidden solid centre has zero correction
        only if all its stencil samples have zero weighted residual; with
        this stencil a one-h zero-residual guard outside the solid suffices
        before deconvolution. Do not mask the returned curl afterwards.
    core_radius : float or None, default=None
        Positive common Gaussian radius sigma in m, required for the optional
        inverse-filter step. The default operator does no inverse filtering.
    deconvolution_steps : {0, 1}, default=0
        Zero returns d=h^3 curl_h[eta m (u_F-u_V)]. One returns (2I-G_h)d,
        using the existing physical Gaussian lattice convolution G_h. This
        one-step scalar approximate inverse acts only on the increment. It
        requires extra zero-support padding through ceil(6*sigma/h) cells.
        It is not an exact inverse and can spread support towards a body.

    Returns
    -------
    numpy.ndarray, shape (N, 3)
        Independent float64 vector-strength increments in m^3/s. Exactly
        matched velocity samples give exactly zero increments, even for
        spatially varying authority or fluid weight.

    Raises
    ------
    ValueError
        If coordinates, dimensions, spacing, callback values, or options are
        invalid, or the computed curl lacks the required zero outer halo.

    Notes
    -----
    ``curl_h`` uses the centered nodal derivative (f(x+h)-f(x-h))/(2h).
    It reuses the velocity-trace surface integral at spacing 2h and divides
    the integrated strength by 8, retaining particle volume h^3. The matching
    centered nodal divergence cancels curl_h to roundoff on complete support.
    Continuous divergence of the Gaussian particle field is not identically
    zero; the lattice derivative, quadrature and finite-core errors require
    refinement. The default represented increment is G_h d, not d.

    There is no particle-state mutation, device initialization, body solve,
    remapping, pruning, clipping, or moment correction. A final state may drop
    exact-zero increments; dropping nonzero values changes the operator.

    Examples
    --------
    A matched velocity remains unchanged under spatially varying authority:

    >>> axis = 0.1 * np.arange(-2, 3)
    >>> nodes = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    >>> delta = curl_residual_correction(
    ...     nodes, (5, 5, 5), 0.1,
    ...     fvm_velocity_at=np.zeros_like, vpm_velocity_at=np.zeros_like,
    ...     authority_at=lambda x: np.exp(-np.sum(x**2, axis=1)),
    ... )
    >>> bool(np.all(delta == 0.0))
    True
    """
    position = np.asarray(positions, dtype=np.float64)
    if position.ndim != 2 or position.shape[1] != 3 or not np.isfinite(position).all():
        raise ValueError("positions must be finite with shape (N, 3)")
    if len(shape) != 3 or any(not isinstance(n, int | np.integer) or n < 1 for n in shape):
        raise ValueError("shape must contain three positive integers")
    if len(position) != int(np.prod(shape)):
        raise ValueError("positions must contain the complete lattice shape")
    spacing = particle_spacing
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("particle_spacing must be finite and positive")
    if not isinstance(deconvolution_steps, int | np.integer) or deconvolution_steps not in (0, 1):
        raise ValueError("deconvolution_steps must be zero or one")
    if core_radius is not None and (not np.isfinite(core_radius) or core_radius <= 0.0):
        raise ValueError("core_radius must be finite and positive")
    if deconvolution_steps and core_radius is None:
        raise ValueError("one inverse-filter step requires core_radius")

    grid = position.reshape(*shape, 3)
    tolerance = 64 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(position))))
    for axis in range(3):
        dimensions = [1, 1, 1]
        dimensions[axis] = shape[axis]
        coordinate = (position[0, axis] + spacing * np.arange(shape[axis])).reshape(dimensions)
        if not np.allclose(grid[..., axis], coordinate, rtol=0.0, atol=tolerance):
            raise ValueError("positions must form the declared uniform C-order lattice")

    def residual_at(query: FloatArray) -> FloatArray:
        """Sample all spatial weights inside the velocity residual, in m/s."""
        mismatch = _velocity_at(fvm_velocity_at, query, "fvm_velocity_at") - _velocity_at(
            vpm_velocity_at, query, "vpm_velocity_at"
        )
        weight = _weight_at(authority_at, query, "authority_at")
        if fluid_weight_at is not None:
            weight = weight * _weight_at(fluid_weight_at, query, "fluid_weight_at")
        return weight[:, None] * mismatch

    correction = vortex_strength_from_velocity_trace(position, 2 * spacing, residual_at) / 8
    padding = 1
    if deconvolution_steps and core_radius is not None:
        padding += int(np.ceil(6 * core_radius / spacing))
    field = correction.reshape(*shape, 3)
    for axis in range(3):
        for boundary in (slice(0, padding), slice(-padding, None)):
            selection = [slice(None)] * 4
            selection[axis] = boundary
            if np.any(field[tuple(selection)] != 0.0):
                raise ValueError(
                    f"curl correction needs {padding} exact-zero outer layers; "
                    "enlarge the lattice or compact the velocity residual before taking curl"
                )
    if deconvolution_steps and core_radius is not None:
        represented = gaussian_represented_vortex_strength(
            correction, shape, spacing, core_radius=core_radius
        )
        correction = 2 * correction - represented
    return correction
