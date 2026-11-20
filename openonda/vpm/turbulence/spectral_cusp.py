"""Spectral eddy viscosity with the Kraichnan cusp, applied to the GBD vorticity lattice.

Each GBD step holds the vorticity on a uniform lattice between the explicit Laplacian and
the regeneration of the particles. The closure transforms that lattice field and multiplies
every Fourier mode by

    exp(-nu_t(k | k_c) k^2 dt),
    nu_t(k | k_c) = C_K^(-3/2) [0.441 + 15.2 exp(-3.03 k_c / k)] sqrt(E(k_c) / k_c),

the eddy viscosity of the eddy-damped quasi-normal Markovian theory for the energy transfer
across a sharp cut-off (Kraichnan 1976; Chollet & Lesieur 1981). E(k_c) is the shell-averaged
kinetic-energy spectrum of the lattice field over 0.9 k_c <= k <= k_c, so the closure acts
only as far as energy has reached the cut-off: a field with no content there is unchanged.

A real multiplier of |k| leaves k . w_hat proportional to w_hat, so it creates no vorticity
divergence; it leaves the total vorticity (k = 0) and the linear impulse (its gradient vanishes
at k = 0) unchanged. Its energy sink is known mode by mode: the multiplier removes
(1 - f(k)^2) of the energy of each mode.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import fft as scipy_fft

__all__ = ["SpectralCuspClosure"]

PLATEAU = 0.441
CUSP_AMPLITUDE = 15.2
CUSP_RATE = 3.03
CUTOFF_BAND = 0.9  # E(k_c) averages the shells 0.9 k_c <= k <= k_c
MARGIN_CELLS = 8  # zero cells around the vorticity support, on the lattice and in the transform


@dataclass
class SpectralCuspStatistics:
    """Diagnostics of the last application, in SI units (energy per unit density)."""

    cutoff_wavenumber: float = 0.0
    cutoff_energy_spectrum: float = 0.0
    plateau_eddy_viscosity: float = 0.0
    cutoff_eddy_viscosity: float = 0.0
    subgrid_kinetic_energy_rate: float = 0.0
    lattice_kinetic_energy: float = 0.0

    def as_row(self) -> dict[str, float]:
        return {
            "spectral_cutoff_wavenumber": self.cutoff_wavenumber,
            "spectral_cutoff_energy_spectrum": self.cutoff_energy_spectrum,
            "spectral_plateau_eddy_viscosity": self.plateau_eddy_viscosity,
            "spectral_cutoff_eddy_viscosity": self.cutoff_eddy_viscosity,
            "subgrid_kinetic_energy_rate": -self.subgrid_kinetic_energy_rate,
        }


class SpectralCuspClosure:
    """Apply the cusp eddy viscosity to a GBD vorticity lattice of spacing ``h``.

    Parameters
    ----------
    kolmogorov_constant : float
        Kolmogorov constant ``C_K`` of the assumed inertial range.
    cutoff_wavenumber : float or None
        Cut-off ``k_c`` in rad/m; None uses the lattice Nyquist wavenumber ``pi/h``.
    use_cuda : bool
        Transform on the GPU with CuPy (cuFFT); otherwise with SciPy on the host.
    """

    def __init__(
        self,
        kolmogorov_constant: float = 1.4,
        cutoff_wavenumber: float | None = None,
        *,
        use_cuda: bool = False,
    ) -> None:
        self.kolmogorov_constant = float(kolmogorov_constant)
        self.cutoff_wavenumber = cutoff_wavenumber
        self.use_cuda = bool(use_cuda)
        if self.use_cuda:
            import cupy
            import cupyx.scipy.fft as cufft

            self._xp, self._fft = cupy, cufft
        else:
            self._xp, self._fft = np, scipy_fft
        self.statistics = SpectralCuspStatistics()

    def report_rows(self) -> list[tuple[str, str]]:
        cutoff = "pi / h" if self.cutoff_wavenumber is None else f"{self.cutoff_wavenumber:g} rad/m"
        return [
            ("model", "spectral eddy viscosity with Kraichnan cusp (Chollet-Lesieur)"),
            ("Kolmogorov constant", f"{self.kolmogorov_constant:g}"),
            ("cut-off wavenumber", cutoff),
            ("transform", "cuFFT" if self.use_cuda else "SciPy FFT"),
        ]

    def _wavenumbers(self, shape: tuple[int, int, int], spacing: float):
        xp = self._xp
        k = [2.0 * math.pi * scipy_fft.fftfreq(n, d=spacing) for n in shape[:2]]
        k.append(2.0 * math.pi * scipy_fft.rfftfreq(shape[2], d=spacing))
        kx = xp.asarray(k[0], dtype=xp.float32)[:, None, None]
        ky = xp.asarray(k[1], dtype=xp.float32)[None, :, None]
        kz = xp.asarray(k[2], dtype=xp.float32)[None, None, :]
        return kx, ky, kz

    def apply(
        self, grid: np.ndarray, spacing: float, time_step_size: float
    ) -> SpectralCuspStatistics:
        """Damp ``grid``, the vortex strength per node in m³/s, shape (nx, ny, nz, 3), in place."""
        xp, fft = self._xp, self._fft
        h = float(spacing)
        cutoff = math.pi / h if self.cutoff_wavenumber is None else float(self.cutoff_wavenumber)
        statistics = SpectralCuspStatistics(cutoff_wavenumber=cutoff)
        occupied = np.any(grid != 0.0, axis=-1)
        if not occupied.any():
            self.statistics = statistics
            return statistics
        # The support with a margin of zeros, written back whole: the damping spreads a
        # little vorticity into the margin, and a further margin of transform padding
        # keeps the periodic images apart.
        lower, upper = [], []
        for axis in range(3):
            filled = np.flatnonzero(occupied.any(axis=tuple(a for a in range(3) if a != axis)))
            lower.append(max(int(filled[0]) - MARGIN_CELLS, 0))
            upper.append(min(int(filled[-1]) + 1 + MARGIN_CELLS, grid.shape[axis]))
        box = tuple(slice(a, b) for a, b in zip(lower, upper, strict=True))
        shape = tuple(
            scipy_fft.next_fast_len(b - a + MARGIN_CELLS, real=True)
            for a, b in zip(lower, upper, strict=True)
        )

        field = xp.asarray(grid[box], dtype=xp.float32)
        spectrum = fft.rfftn(field, s=shape, axes=(0, 1, 2))
        kx, ky, kz = self._wavenumbers(shape, h)
        k2 = kx * kx + ky * ky + kz * kz
        k2[0, 0, 0] = 1.0
        # |u_hat|^2 = (|w_hat|^2 - |k . w_hat|^2 / k^2) / k^2 for the induced velocity
        longitudinal = kx * spectrum[..., 0] + ky * spectrum[..., 1] + kz * spectrum[..., 2]
        modal_energy = (
            xp.sum(xp.abs(spectrum) ** 2, axis=-1) - xp.abs(longitudinal) ** 2 / k2
        ) / k2
        modal_energy[0, 0, 0] = 0.0
        # The half spectrum stores the modes 0 < k_z < Nyquist once for two
        weight = xp.full(shape[2] // 2 + 1, 2.0, dtype=xp.float32)
        weight[0] = 1.0
        if shape[2] % 2 == 0:
            weight[-1] = 1.0
        modal_energy *= weight[None, None, :]
        # Parseval for the vorticity w = Gamma / h^3 of nodes on cells of volume h^3
        energy_scale = 0.5 / (h**3 * float(np.prod(shape)))
        magnitude = xp.sqrt(k2)
        band = (magnitude >= CUTOFF_BAND * cutoff) & (magnitude <= cutoff)
        band_energy = float(xp.sum(modal_energy[band])) * energy_scale
        cutoff_spectrum = band_energy / ((1.0 - CUTOFF_BAND) * cutoff)
        amplitude = self.kolmogorov_constant**-1.5 * math.sqrt(max(cutoff_spectrum, 0.0) / cutoff)
        safe = xp.maximum(magnitude, xp.float32(1e-30))
        eddy_viscosity = amplitude * (PLATEAU + CUSP_AMPLITUDE * xp.exp(-CUSP_RATE * cutoff / safe))
        exponent = eddy_viscosity * k2 * xp.float32(time_step_size)
        exponent[0, 0, 0] = 0.0
        multiplier = xp.exp(-exponent)
        # 1 - f^2 without cancellation: the damping per step is often below 1e-6
        removed = float(xp.sum(modal_energy * -xp.expm1(-2.0 * exponent))) * energy_scale
        total = float(xp.sum(modal_energy)) * energy_scale
        spectrum *= multiplier[..., None]
        damped = fft.irfftn(spectrum, s=shape, axes=(0, 1, 2))
        window = tuple(slice(0, b - a) for a, b in zip(lower, upper, strict=True))
        result = damped[window]
        grid[box] = result.get() if self.use_cuda else result

        statistics.cutoff_energy_spectrum = cutoff_spectrum
        statistics.plateau_eddy_viscosity = amplitude * PLATEAU
        statistics.cutoff_eddy_viscosity = amplitude * (
            PLATEAU + CUSP_AMPLITUDE * math.exp(-CUSP_RATE)
        )
        statistics.subgrid_kinetic_energy_rate = removed / float(time_step_size)
        statistics.lattice_kinetic_energy = total
        self.statistics = statistics
        return statistics
