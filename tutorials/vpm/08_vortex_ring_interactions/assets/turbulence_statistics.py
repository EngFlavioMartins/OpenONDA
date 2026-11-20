"""Spectra, sub-grid dissipation and velocity-gradient statistics of the saved particle fields.

Each backup in ``solution/<case>/vpm/`` is deposited on a Cartesian grid of the particle
spacing with the M4' kernel of the solver's flow integrals, zero-padded, and transformed;
the Gaussian particle cores multiply the transform by ``exp(-sigma^2 k^2 / 4)``. From the
vorticity transform

- ``E(k)``, the kinetic-energy spectrum, ``0.5 |u_hat|^2`` with ``u_hat = i k x w_hat / k^2``;
- ``Omega(k)``, the enstrophy spectrum ``|w_hat|^2`` (no one-half, as ``total_enstrophy``);
- ``eps_mol(k) = nu Omega(k)`` and ``eps_sgs(k) = Re(conj(w_hat) . (nu_t w)_hat)``, the
  molecular and eddy-viscous energy dissipation at each wavenumber. Their integrals are
  the two parts of ``-viscous_kinetic_energy_rate`` in ``flow_integrals.csv``.

Spectra are shell densities: their integrals over ``k`` are the totals. The joint PDF
of the invariants ``Q = -tr(A^2)/2`` and ``R = -tr(A^3)/3`` of the velocity gradient
``A`` is taken over the nodes where ``|w|`` exceeds a tenth of its maximum and is
normalised by ``Q_w = <|w|^2>/4`` there.

Run from the case directory: ``python -m assets.turbulence_statistics`` writes
``samples/<case>/spectra.csv`` and ``samples/<case>/gradient_invariants.npz``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
from scipy import fft

from openonda.vpm.numerics.fourier_integrals import _grid_for_particles, _scatter_vortex_strength_m4

CASE_DIR = Path(__file__).resolve().parents[1]
CASES = ("collision_dns", "collision_les", "leapfrog_dns", "leapfrog_les")
SPECTRUM_INTERVAL = 2.0  # [s] between analysed backups
INVARIANT_BINS = 101
INVARIANT_RANGE = 3.0  # Q / Q_w and R / Q_w^(3/2) span [-3, 3]
VORTICAL_FRACTION = 0.1  # nodes with |w| above this fraction of the maximum


def backups(case):
    """Backup files of ``case`` with their times, in time order."""
    files = sorted((CASE_DIR / "solution" / case / "vpm").glob("vpm_*.h5"))
    timed = []
    for path in files:
        with h5py.File(path, "r") as handle:
            timed.append((float(handle["solver"].attrs["time"]), path))
    return timed


def read_particles(path):
    with h5py.File(path, "r") as handle:
        particles = handle["particles"]
        return {name: particles[name][()].astype(np.float64) for name in particles}


def spectral_fields(state):
    """Wavenumbers, vorticity and nu_t * vorticity transforms of one particle field."""
    position = state["position"]
    strength = state["vortex_strength"]
    spacing = float(np.median(np.cbrt(state["particle_volume"])))
    sigma = float(np.median(state["core_radius"]))
    grid = _grid_for_particles(position, spacing)
    padded = tuple(fft.next_fast_len(int(np.ceil(1.5 * n))) for n in grid.shape)
    frequencies = [2.0 * np.pi * fft.fftfreq(n, d=spacing) for n in padded[:2]]
    frequencies.append(2.0 * np.pi * fft.rfftfreq(padded[2], d=spacing))
    k = np.meshgrid(*frequencies, indexing="ij", sparse=True)
    k_squared = k[0] ** 2 + k[1] ** 2 + k[2] ** 2
    core = np.exp(-0.25 * sigma**2 * k_squared) / spacing**3

    def transform(values):
        deposit = _scatter_vortex_strength_m4(position, values, grid)
        return [fft.rfftn(deposit[..., axis], s=padded, workers=-1) * core for axis in range(3)]

    vorticity = transform(strength)
    eddy_viscosity = state["eddy_viscosity"]
    eddy = (
        transform(strength * eddy_viscosity[:, None])
        if np.any(eddy_viscosity)
        else [np.zeros_like(component) for component in vorticity]
    )
    return {
        "k": k,
        "k_squared": k_squared,
        "vorticity": vorticity,
        "eddy": eddy,
        "padded": padded,
        "grid_shape": grid.shape,
        "spacing": spacing,
        "nu": float(np.median(state["kinematic_viscosity"])),
    }


def shell_spectra(fields):
    """Shell-averaged E, Omega, eps_mol and eps_sgs, each a density in k."""
    k, k_squared = fields["k"], fields["k_squared"]
    padded, spacing = fields["padded"], fields["spacing"]
    magnitude = np.sqrt(k_squared)
    width = 2.0 * np.pi / (min(padded) * spacing)
    bins = np.minimum((magnitude / width + 0.5).astype(np.int64), int(np.pi / spacing / width) + 1)
    # The real transform stores half of k_z: every other plane counts twice
    weight = np.full(padded[2] // 2 + 1, 2.0)
    weight[0] = 1.0
    if padded[2] % 2 == 0:
        weight[-1] = 1.0
    weight = np.broadcast_to(weight[None, None, :], k_squared.shape)
    inverse = np.divide(1.0, k_squared, out=np.zeros_like(k_squared), where=k_squared > 0.0)
    w = fields["vorticity"]
    enstrophy = sum(np.abs(component) ** 2 for component in w)
    sgs = sum(np.real(np.conj(a) * b) for a, b in zip(w, fields["eddy"]))
    # |u_hat|^2 = |k x w_hat|^2 / k^4 = (|w_hat|^2 - |k . w_hat|^2 / k^2) / k^2
    longitudinal = np.abs(k[0] * w[0] + k[1] * w[1] + k[2] * w[2]) ** 2 * inverse
    energy = 0.5 * (enstrophy - longitudinal) * inverse
    norm = spacing**3 / np.prod(padded) / width
    count = int(bins.max()) + 1

    def reduce(values):
        return np.bincount(bins.ravel(), weights=(weight * values).ravel(), minlength=count) * norm

    shells = np.arange(count) * width
    omega = reduce(enstrophy)
    return {
        "k": shells,
        "energy": reduce(energy),
        "enstrophy": omega,
        "molecular_dissipation": fields["nu"] * omega,
        "subgrid_dissipation": reduce(sgs),
    }


def gradient_invariants(fields):
    """Joint histogram of Q / Q_w and R / Q_w^(3/2) over the vortical nodes."""
    k, k_squared = fields["k"], fields["k_squared"]
    padded, shape = fields["padded"], fields["grid_shape"]
    inverse = np.divide(1.0, k_squared, out=np.zeros_like(k_squared), where=k_squared > 0.0)
    w = fields["vorticity"]
    # u_hat = i k x w_hat / k^2, A_ij = du_i/dx_j
    velocity = [
        1j * (k[(i + 1) % 3] * w[(i + 2) % 3] - k[(i + 2) % 3] * w[(i + 1) % 3]) * inverse
        for i in range(3)
    ]
    window = tuple(slice(0, n) for n in shape)
    gradient = np.empty((*shape, 3, 3))
    for i in range(3):
        for j in range(3):
            gradient[..., i, j] = fft.irfftn(1j * k[j] * velocity[i], s=padded, workers=-1)[window]
    vorticity = np.stack(
        [
            gradient[..., 2, 1] - gradient[..., 1, 2],
            gradient[..., 0, 2] - gradient[..., 2, 0],
            gradient[..., 1, 0] - gradient[..., 0, 1],
        ],
        axis=-1,
    )
    magnitude = np.linalg.norm(vorticity, axis=-1)
    vortical = magnitude > VORTICAL_FRACTION * magnitude.max()
    a = gradient[vortical]
    a_squared = a @ a
    q = -0.5 * np.trace(a_squared, axis1=1, axis2=2)
    r = -np.trace(a_squared @ a, axis1=1, axis2=2) / 3.0
    q_w = 0.25 * np.mean(magnitude[vortical] ** 2)
    edges = np.linspace(-INVARIANT_RANGE, INVARIANT_RANGE, INVARIANT_BINS + 1)
    histogram, _, _ = np.histogram2d(r / q_w**1.5, q / q_w, bins=(edges, edges), density=True)
    return histogram, edges, q_w, int(vortical.sum())


def analyse(case, *, invariant_times=()):
    """Write the spectra of every analysed backup and the requested invariant PDFs."""
    samples = CASE_DIR / "samples" / case
    samples.mkdir(parents=True, exist_ok=True)
    rows, invariants = [], {}
    next_time = 0.0
    for time, path in backups(case):
        wanted = any(abs(time - target) < 1e-6 for target in invariant_times)
        if time < next_time - 1e-6 and not wanted:
            continue
        next_time = time + SPECTRUM_INTERVAL
        fields = spectral_fields(read_particles(path))
        spectra = shell_spectra(fields)
        for index, k in enumerate(spectra["k"]):
            rows.append(
                {
                    "time": time,
                    "wavenumber": k,
                    **{name: spectra[name][index] for name in spectra if name != "k"},
                }
            )
        if wanted:
            histogram, edges, q_w, nodes = gradient_invariants(fields)
            invariants[f"{time:.2f}"] = (histogram, q_w, nodes)
            invariants["edges"] = edges
        print(f"{case}: t = {time:.2f} s", flush=True)
    with open(samples / "spectra.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if invariants:
        edges = invariants.pop("edges")
        np.savez_compressed(
            samples / "gradient_invariants.npz",
            edges=edges,
            times=np.array([float(t) for t in invariants]),
            histograms=np.array([value[0] for value in invariants.values()]),
            q_w=np.array([value[1] for value in invariants.values()]),
            nodes=np.array([value[2] for value in invariants.values()]),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cases", nargs="*", default=list(CASES))
    parser.add_argument(
        "--invariant-times",
        type=float,
        nargs="*",
        default=(),
        help="backup times [s] that also get a Q-R joint PDF",
    )
    arguments = parser.parse_args()
    for name in arguments.cases:
        analyse(name, invariant_times=arguments.invariant_times)
