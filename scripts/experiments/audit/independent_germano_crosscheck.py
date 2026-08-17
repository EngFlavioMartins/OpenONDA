#!/usr/bin/env python3
"""Independent re-derivation of the Mansfield dynamic coefficient in stage_8a.

Compares the shipped `ell` (Germano/Leonard) construction against the
Germano-consistent one, under stage_8a's own sign convention.
Self-contained: reimplements only the small operators needed, no repo imports.
"""

import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

LES_N, SIGMA_OVER_H, TEST_RATIO = 32, 2.5, 2.0
PHASES = (0.0, 0.21, 0.37)


# ---------- operators (mirrors stage_4a_formulation.SpectralGrid) ----------
class Grid:
    def __init__(self, n):
        self.n = n
        k = np.fft.fftfreq(n, d=1.0 / n)
        self.kx, self.ky, self.kz = np.meshgrid(k, k, k, indexing="ij")
        self.kvec = np.stack([self.kx, self.ky, self.kz])
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2

    def fft(self, f):
        return np.fft.fftn(f, axes=(-3, -2, -1))

    def ifft(self, f):
        return np.real(np.fft.ifftn(f, axes=(-3, -2, -1)))

    def derivative(self, f, i):
        return self.ifft(1j * self.kvec[i] * self.fft(f))

    def gradient(self, v):
        # g[i][j] = d v_i / d x_j  (matches stage_4a SpectralGrid.gradient)
        return np.stack([np.stack([self.derivative(v[i], j) for j in range(3)]) for i in range(3)])

    def curl(self, v):
        g = self.gradient(v)
        return np.array((g[2, 1] - g[1, 2], g[0, 2] - g[2, 0], g[1, 0] - g[0, 1]))

    def gaussian(self, f, delta):
        # stage_4a convention: exp(-delta^2 k^2 / 4)
        return self.ifft(np.exp(-(delta**2) * self.k2 / 4.0) * self.fft(f))


def m4_prime_1d(r):
    q = np.abs(r)
    w = np.zeros_like(q)
    m1, m2 = q <= 1.0, (q > 1.0) & (q <= 2.0)
    w[m1] = 1.0 - 2.5 * q[m1] ** 2 + 1.5 * q[m1] ** 3
    w[m2] = 0.5 * (2.0 - q[m2]) ** 2 * (1.0 - q[m2])
    return w


def m4_symbol(theta, phase):
    theta = np.asarray(theta, dtype=float)
    q = np.arange(-3, 4, dtype=float)
    d = q + phase
    w = m4_prime_1d(np.abs(d))
    return np.sum(w[:, None] * np.exp(1j * d[:, None] * theta.reshape(1, -1)), axis=0)


def particle_symbol(grid, h, sigma):
    wave = np.fft.fftfreq(grid.n, d=1.0 / grid.n)
    o = [np.abs(m4_symbol(wave * h, p)) ** 2 for p in PHASES]
    m4 = o[0][:, None, None] * o[1][None, :, None] * o[2][None, None, :]
    return np.exp(-(sigma**2) * grid.k2 / 4.0) * m4


def nonlinear_parts(grid, u, w):
    conv = np.einsum("j...,ij...->i...", u, grid.gradient(w))
    stre = np.einsum("j...,ij...->i...", w, grid.gradient(u))
    return conv, stre


def strain_magnitude(grid, u):
    g = grid.gradient(u)
    s = 0.5 * (g + np.swapaxes(g, 0, 1))
    return np.sqrt(2.0 * np.sum(s * s, axis=(0, 1)))


def norm(a):
    return float(np.sqrt(np.mean(a * a)))


# ---------- load AGARD field (matches stage_4a load_agard) ----------
import re

path = ROOT / "docs/dns/agard_hom02/CB128_9.bin"
hlen = int(re.search(rb"HEADERLENGTH=(\d+)", path.read_bytes()[:4096]).group(1))
vel = (
    np.memmap(path, dtype=">f4", mode="r", offset=hlen, shape=(128, 128, 128, 3))
    .transpose(3, 2, 1, 0)
    .astype(np.float64)
)
n = 128
grid = Grid(n)
print(f"field {n}^3, u_rms={norm(vel[0]):.4f}")

# ---------- particle-filtered resolved state ----------
h = 2.0 * np.pi / LES_N
sigma = SIGMA_OVER_H * h
psym = particle_symbol(grid, h, sigma)


def P(f):
    return grid.ifft(psym * grid.fft(f))


vort = grid.curl(vel)
u, w = P(vel), P(vort)
conv_dns, stre_dns = nonlinear_parts(grid, vel, vort)
conv_bar, stre_bar = nonlinear_parts(grid, u, w)
g_exact = (-P(conv_dns) + conv_bar) + (P(stre_dns) - stre_bar)

width = float((6.0 * 2.0**1.5 * math.sqrt(math.pi)) ** (1.0 / 3.0)) * sigma
print(f"Delta_p/h = {width / h:.6f}   (doc: 7.774940)")

tr_exact = float(np.mean(np.sum(w * g_exact, axis=0)))
print(f"exact enstrophy transfer = {tr_exact:+.6f}   (doc: -0.443896)  <-- calibration check")

# ---------- dynamic procedure ----------
test_sigma = TEST_RATIO * sigma


def T(f):
    return grid.gaussian(f, test_sigma)


u_t, w_t = T(u), T(w)

conv, stre = nonlinear_parts(grid, u, w)
conv_t, stre_t = nonlinear_parts(grid, u_t, w_t)

base = -(width**2) * grid.curl(strain_magnitude(grid, u)[None, ...] * grid.curl(w))
tw = TEST_RATIO * width
test_basis = -(tw**2) * grid.curl(strain_magnitude(grid, u_t)[None, ...] * grid.curl(w_t))
M = T(base) - test_basis
den = float(np.mean(np.sum(M * M, axis=0)))

variants = {
    "(a) shipped in stage_8a": conv - conv_t - stre + stre_t,
    "(b) Germano-consistent (correct)": -T(conv) + conv_t + T(stre) - stre_t,
    "(c) test filter added, shipped sign": T(conv) - conv_t - T(stre) + stre_t,
    "(d) shipped sign flipped, no filter": -conv + conv_t + stre - stre_t,
}
print(f"\ndenominator <M.M> = {den:.6e}\n")
for name, ell in variants.items():
    num = float(np.mean(np.sum(ell * M, axis=0)))
    c2 = num / den
    cr = f"{math.sqrt(c2):.6f}" if c2 > 0 else "clipped to 0"
    print(f"{name:38s} <L.M>={num:+.6e}  C_r^2={c2:+.8f}  C_r={cr}")

print("\nreference C_r: Mansfield paper 0.120000, stage_8a appendix-A 0.136700")

# ---------- does the corrected coefficient pass stage_8a's own gates? ----------
ell_ok = variants["(b) Germano-consistent (correct)"]
c2_ok = float(np.mean(np.sum(ell_ok * M, axis=0))) / den
if c2_ok > 0:
    cr_ok = math.sqrt(c2_ok)
    nu_t = (cr_ok * width) ** 2 * strain_magnitude(grid, u)
    torque = -grid.curl(nu_t[None, ...] * grid.curl(w))
    tr_model = float(np.mean(np.sum(w * torque, axis=0)))

    corr = float(np.sum(g_exact * torque) / np.sqrt(np.sum(g_exact**2) * np.sum(torque**2)))
    div = sum(grid.derivative(torque[i], i) for i in range(3))
    print(f"\n--- corrected dynamic model, C_r = {cr_ok:.6f} ---")
    print(f"exact enstrophy transfer     {tr_exact:+.6f}  (doc: -0.443896)")
    print(f"model enstrophy transfer     {tr_model:+.6f}")
    print(f"transfer ratio               {tr_model / tr_exact:.6f}   gate [0.5, 1.5]")
    print(f"correlation                  {corr:.6f}")
    print(f"relative divergence          {norm(div) / norm(grid.gradient(torque)):.3e}")
    print(f"mean-dissipative             {tr_model < 0.0}")
