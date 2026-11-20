"""Spectral energy budget of the particle field: where in wavenumber each term moves energy.

For the field of Gaussian particles, sampled on a lattice of the particle spacing,

    dE(k)/dt = T(k) - D_mol(k) - D_sgs(k) - D_sel(k) - D_num(k),

with every term a shell density in k (its integral over k is the total rate):

- ``T(k) = Re(conj(u_hat) . (u x w)_hat)``, the resolved non-linear transfer; it integrates to
  zero and moves energy between wavenumbers;
- ``D_mol(k)``, the energy removed by the GBD operator div(nu grad w) with the molecular
  viscosity, on the solver's 7-point stencil with arithmetic face viscosities, applied to
  the M4' deposit of the particles and seen through their Gaussian cores:
  ``-Re(conj(k x w_hat) . (k x F_hat)) / k^4`` for the vorticity tendency ``F``;
- ``D_sgs(k)``, the closure's sink: the particle eddy viscosity in the same operator for
  Smagorinsky or sigma, or the multiplier of the spectral cusp model;
- ``D_sel(k)``, the selective eddy viscosity C h^2 max(w.S.w / |w|^2, 0) reconstructed
  from the resolved field when the run uses it (it is not stored in the backups);
- ``dE(k)/dt`` from the backups half an interval before and after;
- ``D_num(k)``, the remainder: what the discretisation removes, mostly the regeneration of
  the particles on the lattice every step. ``nu_num(k) = D_num(k) / (2 k^2 E(k))`` is its
  effective viscosity.

``python -m assets.energy_transfer collision_les --times 2 4`` writes
``samples/<case>/energy_transfer.csv``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py

import numpy as np
import pandas as pd
from scipy import fft

from openonda.vpm.numerics.fourier_integrals import (
    CartesianGrid,
    _grid_for_particles,
    _scatter_vortex_strength_m4,
)
from openonda.vpm.physics.diffusion.regeneration import (
    nearest_node_mapping,
    scatter_scalar_weighted,
)
from openonda.vpm.turbulence.spectral_cusp import (
    CUSP_AMPLITUDE,
    CUSP_RATE,
    PLATEAU,
    SpectralCuspClosure,
)

from assets.turbulence_statistics import backups, read_particles

CASE_DIR = Path(__file__).resolve().parents[1]
PADDING = 1.5  # FFT box over the particle support
COLUMNS = (
    "time",
    "wavenumber",
    "energy",
    "energy_rate",
    "transfer",
    "molecular_dissipation",
    "subgrid_dissipation",
    "selective_dissipation",
    "numerical_dissipation",
    "numerical_viscosity",
)


def common_grid(states, spacing):
    position = np.vstack([state["position"] for state in states])
    return _grid_for_particles(position, spacing, padding=4)


def lattice_vorticity(state, grid):
    """M4' deposit of the strengths divided by the cell volume: the field GBD diffuses."""
    return (
        _scatter_vortex_strength_m4(state["position"], state["vortex_strength"], grid)
        / grid.spacing**3
    )


def variable_laplacian(field, viscosity, spacing):
    """div(nu grad w) on the 7-point stencil with arithmetic face viscosities, zero outside."""
    out = np.zeros_like(field)
    padded = np.pad(field, ((1, 1), (1, 1), (1, 1), (0, 0)))
    nu = np.pad(viscosity, 1)
    centre = (slice(1, -1),) * 3
    for axis in range(3):
        for shift in (-1, 1):
            neighbour = list(centre)
            neighbour[axis] = slice(1 + shift, padded.shape[axis] - 1 + shift)
            neighbour = tuple(neighbour)
            face = 0.5 * (nu[centre] + nu[neighbour])
            out += face[..., None] * (padded[neighbour] - padded[centre])
    return out / spacing**2


class Spectral:
    """Zero-padded transforms on one grid, with the Gaussian particle-core multiplier."""

    def __init__(self, grid, sigma):
        self.grid = grid
        self.shape = tuple(fft.next_fast_len(int(np.ceil(PADDING * n))) for n in grid.shape)
        h = grid.spacing
        frequencies = [2.0 * np.pi * fft.fftfreq(n, d=h) for n in self.shape[:2]]
        frequencies.append(2.0 * np.pi * fft.rfftfreq(self.shape[2], d=h))
        self.k = np.meshgrid(*frequencies, indexing="ij", sparse=True)
        self.k2 = self.k[0] ** 2 + self.k[1] ** 2 + self.k[2] ** 2
        self.inverse = np.divide(1.0, self.k2, out=np.zeros_like(self.k2), where=self.k2 > 0)
        self.core = np.exp(-0.25 * sigma**2 * self.k2)
        weight = np.full(self.shape[2] // 2 + 1, 2.0)
        weight[0] = 1.0
        if self.shape[2] % 2 == 0:
            weight[-1] = 1.0
        self.weight = np.broadcast_to(weight[None, None, :], self.k2.shape)
        self.width = 2.0 * np.pi / (min(self.shape) * h)
        self.bins = np.minimum(
            (np.sqrt(self.k2) / self.width + 0.5).astype(np.int64), int(np.pi / h / self.width) + 1
        )
        self.norm = h**3 / np.prod(self.shape) / self.width

    def forward(self, field):
        return [fft.rfftn(field[..., a], s=self.shape, workers=-1) * self.core for a in range(3)]

    def inverse_field(self, spectrum):
        window = tuple(slice(0, n) for n in self.grid.shape)
        return np.stack([fft.irfftn(c, s=self.shape, workers=-1)[window] for c in spectrum], -1)

    def cross(self, a, b):
        return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]

    def velocity(self, w):
        return [1j * c * self.inverse for c in self.cross(self.k, w)]

    def shell(self, values):
        count = int(self.bins.max()) + 1
        return np.bincount(self.bins.ravel(), (self.weight * values).ravel(), count) * self.norm

    def energy(self, w):
        u = self.velocity(w)
        return self.shell(0.5 * sum(np.abs(c) ** 2 for c in u))

    def dissipation(self, w, tendency):
        """Energy removed per k by a vorticity tendency, from its transverse part."""
        kw, kf = self.cross(self.k, w), self.cross(self.k, tendency)
        rate = sum(np.real(np.conj(a) * b) for a, b in zip(kw, kf)) * self.inverse**2
        return -self.shell(rate)


def configuration(path):
    """The solver configuration stored in a backup."""
    with h5py.File(path, "r") as handle:
        return json.loads(handle["solver"].attrs["configuration"])


def velocity_gradient(spectral, u):
    """du_i/dx_j on the lattice nodes."""
    window = tuple(slice(0, n) for n in spectral.grid.shape)
    gradient = np.empty((*spectral.grid.shape, 3, 3))
    for i in range(3):
        for j in range(3):
            gradient[..., i, j] = fft.irfftn(
                1j * spectral.k[j] * u[i], s=spectral.shape, workers=-1
            )[window]
    return gradient


def closure_sinks(state, config, spectral, field, w, u, mapping):
    """Energy removed per k by the configured closure: model and selective viscosity.

    Smagorinsky or sigma: the particle eddy viscosity on the lattice nodes in the GBD
    operator. Spectral cusp: the multiplier of the closure, with E(k_c) of this lattice.
    Selective eddy viscosity: C h^2 max(w.S.w / |w|^2, 0) from the resolved field.
    """
    grid, h = spectral.grid, spectral.grid.spacing
    zero = np.zeros(int(spectral.bins.max()) + 1)
    turbulence = config["turbulence"]
    if turbulence["model"] in ("smagorinsky", "sigma"):
        eddy = scatter_scalar_weighted(
            state["position"],
            state["vortex_strength"],
            state["eddy_viscosity"].astype(np.float32),
            grid.origin,
            h,
            *grid.shape,
            mapping=mapping,
            fill_empty=True,
        ).astype(np.float64)
        model = spectral.dissipation(w, spectral.forward(variable_laplacian(field, eddy, h)))
    elif turbulence["model"] == "spectral_cusp":
        closure = SpectralCuspClosure(
            turbulence["kolmogorov_constant"], turbulence["cutoff_wavenumber"]
        )
        statistics = closure.apply((field * h**3).astype(np.float32), h, 1.0)
        cutoff = statistics.cutoff_wavenumber
        amplitude = statistics.plateau_eddy_viscosity / PLATEAU
        magnitude = np.sqrt(spectral.k2)
        eddy = amplitude * (
            PLATEAU + CUSP_AMPLITUDE * np.exp(-CUSP_RATE * cutoff / np.maximum(magnitude, 1e-30))
        )
        dt = float(config["time_step_size"])
        loss = -np.expm1(-2.0 * eddy * spectral.k2 * dt) / dt
        modal = 0.5 * sum(np.abs(c) ** 2 for c in u)
        model = spectral.shell(loss * modal)
    else:
        model = zero
    coefficient = config["stabilization"].get("selective_eddy_viscosity_coefficient") or 0.0
    if coefficient > 0.0:
        strain = velocity_gradient(spectral, u)
        strain = 0.5 * (strain + np.swapaxes(strain, -1, -2))
        vorticity = spectral.inverse_field(w)
        magnitude2 = np.einsum("...i,...i->...", vorticity, vorticity)
        stretching = np.einsum("...i,...ij,...j->...", vorticity, strain, vorticity)
        selective_viscosity = (
            coefficient * h**2 * np.maximum(stretching / np.maximum(magnitude2, 1e-30), 0.0)
        )
        selective_viscosity[magnitude2 < 1e-6 * magnitude2.max()] = 0.0
        selective = spectral.dissipation(
            w, spectral.forward(variable_laplacian(field, selective_viscosity, h))
        )
    else:
        selective = zero
    return model, selective


def budget(case, time, interval=0.5):
    timed = {round(t, 6): path for t, path in backups(case)}
    before, now, after = (timed[round(time + d, 6)] for d in (-interval, 0.0, interval))
    states = [read_particles(path) for path in (before, now, after)]
    spacing = float(np.median(np.cbrt(states[1]["particle_volume"])))
    sigma = float(np.median(states[1]["core_radius"]))
    nu = float(np.median(states[1]["kinematic_viscosity"]))
    grid = common_grid(states, spacing)
    spectral = Spectral(grid, sigma)
    energies = [
        spectral.energy(spectral.forward(lattice_vorticity(s, grid)))
        for s in (states[0], states[2])
    ]
    state = states[1]
    field = lattice_vorticity(state, grid)
    w = spectral.forward(field)
    u = spectral.velocity(w)
    nonlinear = np.moveaxis(np.cross(spectral.inverse_field(u), spectral.inverse_field(w)), -1, 0)
    transfer = spectral.shell(
        sum(
            np.real(np.conj(a) * fft.rfftn(b, s=spectral.shape, workers=-1))
            for a, b in zip(u, nonlinear, strict=True)
        )
    )
    mapping = nearest_node_mapping(
        state["position"], state["vortex_strength"], grid.origin, spacing, *grid.shape
    )
    molecular = spectral.dissipation(
        w, spectral.forward(variable_laplacian(field, np.full(grid.shape, nu), spacing))
    )
    subgrid, selective = closure_sinks(state, configuration(now), spectral, field, w, u, mapping)
    energy = spectral.energy(w)
    rate = (energies[1] - energies[0]) / (2.0 * interval)
    count = min(len(energy), len(energies[0]), len(energies[1]))
    k = np.arange(count) * spectral.width
    numerical = (
        transfer[:count] - molecular[:count] - subgrid[:count] - selective[:count] - rate[:count]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        viscosity = np.where(k > 0, numerical / (2.0 * k**2 * energy[:count]), np.nan)
    return pd.DataFrame(
        {
            "time": time,
            "wavenumber": k,
            "energy": energy[:count],
            "energy_rate": rate[:count],
            "transfer": transfer[:count],
            "molecular_dissipation": molecular[:count],
            "subgrid_dissipation": subgrid[:count],
            "selective_dissipation": selective[:count],
            "numerical_dissipation": numerical,
            "numerical_viscosity": viscosity,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case")
    parser.add_argument("--times", type=float, nargs="+", required=True)
    arguments = parser.parse_args()
    available = {round(t, 6) for t, _ in backups(arguments.case)}
    times = [
        t for t in arguments.times if all(round(t + d, 6) in available for d in (-0.5, 0.0, 0.5))
    ]
    # A run that stopped early leaves the later times out; the table may hold none
    table = (
        pd.concat([budget(arguments.case, t) for t in times])
        if times
        else pd.DataFrame(columns=COLUMNS)
    )
    table.to_csv(CASE_DIR / "samples" / arguments.case / "energy_transfer.csv", index=False)
    for time, rows in table.groupby("time"):
        dk = rows["wavenumber"].iloc[1]
        totals = {
            c: rows[c].sum() * dk
            for c in rows.columns
            if c not in ("time", "wavenumber", "numerical_viscosity")
        }
        print(f"t = {time:g}: " + ", ".join(f"{c} {v:.4g}" for c, v in totals.items()))
