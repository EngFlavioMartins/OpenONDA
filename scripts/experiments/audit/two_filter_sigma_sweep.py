#!/usr/bin/env python3
"""Two follow-ups to the OpenONDA VPM-LES audit.

(1) Is the DIAD rejection specific to core_radius/particle_spacing = 2.5?  Sweep the
    particle-vs-added-filter enstrophy-transfer split over core_radius/particle_spacing.
(2) What coefficient would a dissipation-matched Mansfield model need,
    versus the appendix value and the dynamic value?
"""

import math
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PHASES = (0.0, 0.21, 0.37)
LES_N, DELTA_OVER_H = 32, 2.0


class Grid:
    def __init__(self, n):
        self.n = n
        k = np.fft.fftfreq(n, d=1.0 / n)
        self.kx, self.ky, self.kz = np.meshgrid(k, k, k, indexing="ij")
        self.kvec = np.stack([self.kx, self.ky, self.kz])
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2

    @staticmethod
    def fft(f):
        return np.fft.fftn(f, axes=(-3, -2, -1))

    @staticmethod
    def ifft(f):
        return np.fft.ifftn(f, axes=(-3, -2, -1)).real

    def derivative(self, f, i):
        return self.ifft(1j * self.kvec[i] * self.fft(f))

    def gradient(self, v):
        return np.stack([np.stack([self.derivative(v[i], j) for j in range(3)]) for i in range(3)])

    def curl(self, v):
        g = self.gradient(v)
        return np.array((g[2, 1] - g[1, 2], g[0, 2] - g[2, 0], g[1, 0] - g[0, 1]))


def m4_prime_1d(r):
    q = np.abs(r)
    w = np.zeros_like(q)
    m1, m2 = q <= 1.0, (q > 1.0) & (q <= 2.0)
    w[m1] = 1.0 - 2.5 * q[m1] ** 2 + 1.5 * q[m1] ** 3
    w[m2] = 0.5 * (2.0 - q[m2]) ** 2 * (1.0 - q[m2])
    return w


def m4_symbol(theta, phase):
    q = np.arange(-3, 4, dtype=float)
    d = q + phase
    wt = m4_prime_1d(np.abs(d))
    return np.sum(
        wt[:, None] * np.exp(1j * d[:, None] * np.asarray(theta, float).reshape(1, -1)), axis=0
    )


def particle_symbol(grid, particle_spacing, core_radius):
    wave = np.fft.fftfreq(grid.n, d=1.0 / grid.n)
    o = [np.abs(m4_symbol(wave * particle_spacing, p)) ** 2 for p in PHASES]
    return np.exp(-(core_radius**2) * grid.k2 / 4.0) * (
        o[0][:, None, None] * o[1][None, :, None] * o[2][None, None, :]
    )


def nl(grid, u, w):
    return (
        np.einsum("j...,ij...->i...", u, grid.gradient(w)),
        np.einsum("j...,ij...->i...", w, grid.gradient(u)),
    )


def sgs(grid, u, filter_function):
    v = grid.curl(u)
    c, s = nl(grid, u, v)
    ub, wb = filter_function(u), filter_function(v)
    cb, sb = nl(grid, ub, wb)
    return ub, wb, (-filter_function(c) + cb) + (filter_function(s) - sb)


def strain_mag(grid, u):
    g = grid.gradient(u)
    s = 0.5 * (g + np.swapaxes(g, 0, 1))
    return np.sqrt(2.0 * np.sum(s * s, axis=(0, 1)))


def nrm(a):
    return float(np.sqrt(np.mean(a * a)))


def tr(a, b):
    return float(np.mean(np.sum(a * b, axis=0)))


path = ROOT / "docs/dns/agard_hom02/CB128_9.bin"
hlen = int(re.search(rb"HEADERLENGTH=(\d+)", path.read_bytes()[:4096]).group(1))
vel = (
    np.memmap(path, dtype=">f4", mode="r", offset=hlen, shape=(128, 128, 128, 3))
    .transpose(3, 2, 1, 0)
    .astype(np.float64)
)
grid = Grid(128)
particle_spacing = 2.0 * np.pi / LES_N
paper_delta = DELTA_OVER_H * particle_spacing
gdelta = paper_delta / math.sqrt(6.0)
gsym = np.exp(-(gdelta**2) * grid.k2 / 4.0)


def G(f):
    return grid.ifft(gsym * grid.fft(f))


WR = float((6.0 * 2.0**1.5 * math.sqrt(math.pi)) ** (1.0 / 3.0))

print(
    "(1) Two-filter transfer split vs core_radius/particle_spacing  (added filter Delta/particle_spacing = 2)"
)
print(
    f"{'core_radius/particle_spacing':>8} {'K_sig(pi/particle_spacing)':>12} {'particle share':>15} {'added share':>12} {'D_eff/particle_spacing':>8}"
)
for s_h in (0.75, 1.0, 1.5, 2.0, 2.5):
    core_radius = s_h * particle_spacing
    symbol = particle_symbol(grid, particle_spacing, core_radius)

    def particle_filter(field, filter_symbol=symbol):
        return grid.ifft(filter_symbol * grid.fft(field))

    def combined_filter(field, particle_filter_operator=particle_filter):
        return G(particle_filter_operator(field))

    _, wb_t, g_tot = sgs(grid, vel, combined_filter)
    ut, wt, g_par = sgs(grid, vel, particle_filter)
    _, _, g_add = sgs(grid, ut, G)
    t_tot = tr(wb_t, g_tot)
    t_par = tr(wb_t, G(g_par))
    t_add = tr(wb_t, g_add)
    knyq = math.exp(-((s_h * math.pi) ** 2) / 4.0)
    deff = math.sqrt(DELTA_OVER_H**2 + 6.0 * s_h**2)
    print(f"{s_h:>8.2f} {knyq:>12.2e} {t_par / t_tot:>15.5f} {t_add / t_tot:>12.5f} {deff:>8.3f}")

print("\n(2) Mansfield coefficient calibration at core_radius/particle_spacing = 2.5")
core_radius = 2.5 * particle_spacing
symbol = particle_symbol(grid, particle_spacing, core_radius)


def P(f, sy=symbol):
    return grid.ifft(sy * grid.fft(f))


u, w, g_ex = sgs(grid, vel, P)
width = WR * core_radius
base = -(width**2) * grid.curl(strain_mag(grid, u)[None, ...] * grid.curl(w))
t_ex, t_base = tr(w, g_ex), tr(w, base)
c2_match = t_ex / t_base
c2_ls = float(np.sum(g_ex * base) / np.sum(base * base))
print(f"exact transfer                     {t_ex:+.6f}")
for name, c2 in (
    ("dynamic (bug fixed)", 0.00239603),
    ("appendix-A 0.136700", 0.136700**2),
    ("Mansfield paper 0.12", 0.12**2),
    ("transfer-matched", c2_match),
    ("least-squares optimal", c2_ls),
):
    if c2 <= 0:
        continue
    g_m = c2 * base
    print(
        f"  {name:22s} C_r={math.sqrt(c2):.6f}  "
        f"transfer ratio={tr(w, g_m) / t_ex:>7.4f}  "
        f"rel L2={nrm(g_m - g_ex) / nrm(g_ex):.4f}"
    )
print(
    "\ncorrelation (C_r-independent)       "
    f"{float(np.sum(g_ex * base) / np.sqrt(np.sum(g_ex**2) * np.sum(base**2))):.6f}"
)
