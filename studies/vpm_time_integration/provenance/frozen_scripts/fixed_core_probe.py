#!/usr/bin/env python3
"""Bounded CPU probes for the fixed-core Gaussian VPM stability study.

The implementation mirrors OpenONDA's direct Gaussian convention:
sigma_ij=(sigma_i+sigma_j)/2, q includes 1/(4*pi), the small-rho series
switches at rho=0.2, and stretching uses the historical/default transposed
velocity-gradient form.  It intentionally excludes diffusion, remeshing,
relaxation, VLM feedback, and every topology-changing operation.
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PI = math.pi
ZETA0 = PI ** -1.5
TWO_OVER_SQRT_PI = 2.0 / math.sqrt(PI)
ONE_OVER_FOUR_PI = 1.0 / (4.0 * PI)
FOUR_OVER_THREE_SQRT_PI = 4.0 / (3.0 * math.sqrt(PI))
Q_SERIES_CROSSOVER = 0.2


def erf_as(x: np.ndarray) -> np.ndarray:
    """Vectorized Abramowitz-Stegun erf approximation used in production."""
    a1, a2, a3 = 0.254829592, -0.284496736, 1.421413741
    a4, a5, p = -1.453152027, 1.061405429, 0.327591100
    ax = np.abs(x)
    t = 1.0 / (1.0 + p * ax)
    poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t
    return np.sign(x) * (1.0 - poly * np.exp(-(ax * ax)))


def gaussian_q(rho: np.ndarray) -> np.ndarray:
    rho = np.asarray(rho, dtype=float)
    d2 = rho * rho
    series = (
        FOUR_OVER_THREE_SQRT_PI
        * rho
        * d2
        * (1.0 - 0.6 * d2 + (3.0 / 14.0) * d2 * d2)
        * ONE_OVER_FOUR_PI
    )
    closed = (erf_as(rho) - TWO_OVER_SQRT_PI * rho * np.exp(-d2)) * ONE_OVER_FOUR_PI
    return np.where(rho < Q_SERIES_CROSSOVER, series, closed)


def gaussian_zeta(rho: np.ndarray) -> np.ndarray:
    return ZETA0 * np.exp(-(np.asarray(rho) ** 2))


def skew(v: np.ndarray) -> np.ndarray:
    out = np.zeros((len(v), 3, 3), dtype=float)
    out[:, 0, 1], out[:, 0, 2] = -v[:, 2], v[:, 1]
    out[:, 1, 0], out[:, 1, 2] = v[:, 2], -v[:, 0]
    out[:, 2, 0], out[:, 2, 1] = -v[:, 1], v[:, 0]
    return out


def velocity_gradient(x: np.ndarray, gamma: np.ndarray, sigma: np.ndarray):
    """Return velocity and grad(u), matching the fused direct pair algebra."""
    r = x[:, None, :] - x[None, :, :]
    r2 = np.einsum("ijk,ijk->ij", r, r)
    rmag = np.sqrt(r2)
    sig = 0.5 * (sigma[:, None] + sigma[None, :])
    rho = rmag / sig
    q = gaussian_q(rho)
    zeta = gaussian_zeta(rho) / (sig**3)
    nz = rmag > 1.0e-14

    term1 = np.zeros_like(rmag)
    term2 = np.zeros_like(rmag)
    term1[nz] = q[nz] / (r2[nz] * rmag[nz])
    term2[nz] = 3.0 * q[nz] / (r2[nz] ** 2 * rmag[nz]) - zeta[nz] / r2[nz]

    # Production accumulates r x Gamma and negates it: u = Gamma x r.
    gx_r = np.cross(gamma[None, :, :], r)
    u = np.sum(term1[:, :, None] * gx_r, axis=1)

    r_x_g = np.cross(r, gamma[None, :, :])
    pair_grad = term1[:, :, None, None] * skew(gamma)[None, :, :, :]
    pair_grad += term2[:, :, None, None] * (
        r_x_g[:, :, :, None] * r[:, :, None, :]
    )
    # The implementation supplies the analytic finite centre gradient for
    # every zero-separation pair (normally just i=j).
    zero_i, zero_j = np.nonzero(~nz)
    pair_grad[zero_i, zero_j] = (
        ZETA0 / (3.0 * sig[zero_i, zero_j] ** 3)
    )[:, None, None] * skew(gamma)[zero_j]
    return u, np.sum(pair_grad, axis=1)


def pack(x: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    return np.concatenate((x.ravel(), gamma.ravel()))


def unpack(y: np.ndarray):
    n = len(y) // 6
    return y[: 3 * n].reshape(n, 3), y[3 * n :].reshape(n, 3)


def rhs(y: np.ndarray, sigma: np.ndarray, part: str = "coupled") -> np.ndarray:
    x, gamma = unpack(y)
    u, grad = velocity_gradient(x, gamma, sigma)
    # Historical OpenONDA default: (grad u)^T Gamma.
    stretch = np.einsum("nij,ni->nj", grad, gamma)
    if part == "A":
        stretch[:] = 0.0
    elif part == "B":
        u[:] = 0.0
    return pack(u, stretch)


def ssprk3(y: np.ndarray, h: float, sigma: np.ndarray, part="coupled") -> np.ndarray:
    k1 = rhs(y, sigma, part)
    k2 = rhs(y + h * k1, sigma, part)
    k3 = rhs(y + 0.25 * h * (k1 + k2), sigma, part)
    return y + h * (k1 / 6.0 + k2 / 6.0 + 2.0 * k3 / 3.0)


def rk4(y: np.ndarray, h: float, sigma: np.ndarray) -> np.ndarray:
    k1 = rhs(y, sigma)
    k2 = rhs(y + 0.5 * h * k1, sigma)
    k3 = rhs(y + 0.5 * h * k2, sigma)
    k4 = rhs(y + h * k3, sigma)
    return y + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def step(y: np.ndarray, h: float, sigma: np.ndarray, method: str) -> np.ndarray:
    if method == "coupled_ssprk3":
        return ssprk3(y, h, sigma)
    if method == "historical_lie":
        return ssprk3(ssprk3(y, h, sigma, "A"), h, sigma, "B")
    if method == "symmetric_strang":
        y = ssprk3(y, 0.5 * h, sigma, "A")
        y = ssprk3(y, h, sigma, "B")
        return ssprk3(y, 0.5 * h, sigma, "A")
    raise ValueError(method)


def integrate(y0, sigma, horizon, nsteps, method):
    h = horizon / nsteps
    y = y0.copy()
    for _ in range(nsteps):
        y = step(y, h, sigma, method)
    return y


def reference_integrate(y0, sigma, horizon, nsteps=2048):
    h = horizon / nsteps
    y = y0.copy()
    for _ in range(nsteps):
        y = rk4(y, h, sigma)
    return y


def smooth_vorticity(x: np.ndarray) -> np.ndarray:
    w = np.column_stack(
        (
            np.sin(0.7 * x[:, 1]) + 0.5 * np.cos(0.5 * x[:, 2]),
            np.sin(0.6 * x[:, 2]) + 0.4 * np.cos(0.4 * x[:, 0]),
            np.sin(0.5 * x[:, 0]) + 0.3 * np.cos(0.6 * x[:, 1]),
        )
    )
    # Each component is independent of its own coordinate, hence the analytic
    # field is solenoidal.  Remove the finite-cloud mean circulation.
    w -= np.mean(w, axis=0)
    return 0.6 * w


def cloud(name: str, side=3):
    axis = np.linspace(-1.0, 1.0, side)
    x = np.array(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1).T
    if name.startswith("jitter"):
        seed = 314159 if name == "jitter_a" else 271828
        x += np.random.default_rng(seed).uniform(-0.18, 0.18, size=x.shape)
    elif name == "cluster_void":
        x[:, 2] *= 0.45
        x[:, 0] = np.where(x[:, 0] > 0.0, 0.55 * x[:, 0], x[:, 0])
        x += np.random.default_rng(161803).uniform(-0.04, 0.04, size=x.shape)
    volume = np.ones(len(x))
    gamma = smooth_vorticity(x) * volume[:, None]
    return x, gamma, volume


def geometry_metrics(name, x, volume, sigma):
    r = x[:, None, :] - x[None, :, :]
    d = np.linalg.norm(r, axis=2)
    d_no_self = d + np.eye(len(x)) * 1.0e9
    nn = np.min(d_no_self, axis=1)
    ell_v = volume ** (1.0 / 3.0)

    grid_axis = np.linspace(-1.0, 1.0, 9)
    sample = np.array(np.meshgrid(grid_axis, grid_axis, grid_axis, indexing="ij")).reshape(3, -1).T
    fill = float(np.max(np.min(np.linalg.norm(sample[:, None, :] - x[None, :, :], axis=2), axis=1)))
    separation = 0.5 * float(np.min(d_no_self))

    weights = volume[None, :] * gaussian_zeta(d / sigma) / sigma**3
    coverage = np.sum(weights, axis=1)
    m1 = np.sum(weights[:, :, None] * (x[None, :, :] - x[:, None, :]) / sigma, axis=1)

    anis = []
    moment_cond = []
    for i in range(len(x)):
        offsets = (x - x[i]) / sigma
        cov = np.einsum("n,ni,nj->ij", weights[i], offsets, offsets)
        eig = np.linalg.eigvalsh(cov)
        anis.append(math.sqrt(max(eig[-1], 1e-30) / max(eig[0], 1e-30)))
        moment_cond.append(max(eig[-1], 1e-30) / max(eig[0], 1e-30))

    return {
        "cloud": name,
        "n": len(x),
        "sigma_over_volume_spacing": float(sigma / np.median(ell_v)),
        "nearest_neighbor_min": float(np.min(nn)),
        "nearest_neighbor_median": float(np.median(nn)),
        "sigma_over_nn_median": float(np.median(sigma / nn)),
        "sampled_fill_distance_box": fill,
        "separation_radius": separation,
        "sampled_mesh_ratio_box": fill / separation,
        "coverage_mean": float(np.mean(coverage)),
        "coverage_min": float(np.min(coverage)),
        "coverage_max": float(np.max(coverage)),
        "zeroth_moment_abs_rms": float(np.sqrt(np.mean((coverage - 1.0) ** 2))),
        "first_moment_norm_rms": float(np.sqrt(np.mean(np.sum(m1 * m1, axis=1)))),
        "local_anisotropy_median": float(np.median(anis)),
        "local_anisotropy_max": float(np.max(anis)),
        "moment_matrix_condition_median": float(np.median(moment_cond)),
        "moment_matrix_condition_max": float(np.max(moment_cond)),
    }


def scaled_error(y, ref, ell, gamma_scale):
    x, g = unpack(y)
    xr, gr = unpack(ref)
    z = np.concatenate(((x - xr).ravel() / ell, (g - gr).ravel() / gamma_scale))
    return float(np.linalg.norm(z) / math.sqrt(len(z)))


def derivative_checks():
    gamma = np.array([[0.7, -0.2, 0.4]])
    direction = np.array([1.0, 2.0, -0.7])
    direction /= np.linalg.norm(direction)
    rows = []
    for rho in [0.0, 1e-4, 0.05, 0.199, 0.201, 0.5, 1.0, 3.0]:
        # A separate source at the origin avoids conflating the source-centre
        # derivative with movement of both target and source coordinates.
        xt = rho * direction
        x = np.vstack((xt, np.zeros(3)))
        g = np.vstack((np.zeros(3), gamma[0]))
        sig = np.ones(2)
        _, grad = velocity_gradient(x, g, sig)
        analytic = grad[0]
        eps = 2e-6
        fd = np.empty((3, 3))
        for k in range(3):
            xp, xm = x.copy(), x.copy()
            xp[0, k] += eps
            xm[0, k] -= eps
            up, _ = velocity_gradient(xp, g, sig)
            um, _ = velocity_gradient(xm, g, sig)
            fd[:, k] = (up[0] - um[0]) / (2 * eps)
        abs_err = np.linalg.norm(fd - analytic, ord=2)
        denom = max(np.linalg.norm(fd, ord=2), np.linalg.norm(analytic, ord=2), 1e-15)
        rows.append(
            {
                "rho": rho,
                "absolute_operator_error": float(abs_err),
                "relative_operator_error": float(abs_err / denom),
                "analytic_operator_norm": float(np.linalg.norm(analytic, ord=2)),
            }
        )
    return rows


def kernel_crossover_check():
    rho = Q_SERIES_CROSSOVER
    d2 = rho * rho
    series = (
        FOUR_OVER_THREE_SQRT_PI
        * rho
        * d2
        * (1.0 - 0.6 * d2 + (3.0 / 14.0) * d2 * d2)
        * ONE_OVER_FOUR_PI
    )
    device_series64 = (
        0.7522527780636751
        * rho
        * d2
        * (1.0 - 0.6 * d2 + (3.0 / 14.0) * d2 * d2)
        * 0.0795774715
    )
    device_closed64 = float(
        (erf_as(np.array([rho]))[0] - 1.1283791671 * rho * math.exp(-d2))
        * 0.0795774715
    )
    exact_closed = (
        math.erf(rho) - TWO_OVER_SQRT_PI * rho * math.exp(-d2)
    ) * ONE_OVER_FOUR_PI
    f32 = np.float32
    rho32 = f32(rho)
    d232 = f32(rho32 * rho32)
    series32 = f32(
        f32(FOUR_OVER_THREE_SQRT_PI)
        * rho32
        * d232
        * f32(f32(1.0) - f32(0.6) * d232 + f32(3.0 / 14.0) * d232 * d232)
        * f32(0.0795774715)
    )
    a1, a2, a3, a4, a5, p = map(
        f32, [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429, 0.327591100]
    )
    t32 = f32(1.0) / f32(f32(1.0) + p * rho32)
    poly32 = f32(((((a5 * t32 + a4) * t32 + a3) * t32 + a2) * t32 + a1) * t32)
    exp32 = f32(np.exp(f32(-rho32 * rho32)))
    erf32 = f32(f32(1.0) - poly32 * exp32)
    approximate_closed32 = f32(
        (erf32 - f32(1.1283791671) * rho32 * exp32) * f32(0.0795774715)
    )
    return {
        "rho": rho,
        "host_exact_constant_series_branch_limit": series,
        "exact_gaussian_value": exact_closed,
        "device_formula_f64_series_value": device_series64,
        "device_formula_f64_approximate_erf_value": device_closed64,
        "device_formula_f64_branch_jump": device_closed64 - device_series64,
        "device_formula_f64_branch_jump_relative_to_exact": abs(device_closed64 - device_series64)
        / abs(exact_closed),
        "device_formula_f32_series_value": float(series32),
        "device_formula_f32_approximate_erf_value": float(approximate_closed32),
        "device_formula_f32_branch_jump": float(approximate_closed32 - series32),
        "device_formula_f32_branch_jump_relative_to_exact": abs(
            float(approximate_closed32 - series32)
        )
        / abs(exact_closed),
        "host_exact_erf_branch_jump": exact_closed - series,
        "host_exact_erf_branch_jump_relative_to_exact": abs(exact_closed - series)
        / abs(exact_closed),
        "device_formula_f64_approximate_branch_relative_error": abs(device_closed64 - exact_closed)
        / abs(exact_closed),
        "series_branch_relative_error": abs(series - exact_closed) / abs(exact_closed),
    }


def scales(y0, ell=1.0):
    _, g = unpack(y0)
    gscale = float(np.sqrt(np.mean(np.sum(g * g, axis=1))))
    return ell, max(gscale, 1e-12)


def to_z(y, ell, gscale):
    x, g = unpack(y)
    return pack(x / ell, g / gscale)


def from_z(z, ell, gscale):
    x, g = unpack(z)
    return pack(x * ell, g * gscale)


def numerical_jacobian_map(map_z, z, eps=2e-6):
    m = len(z)
    jac = np.empty((m, m))
    for k in range(m):
        dz = np.zeros(m)
        dz[k] = eps
        jac[:, k] = (map_z(z + dz) - map_z(z - dz)) / (2.0 * eps)
    return jac


def rhs_jacobian_scaled(y, sigma, ell, gscale):
    z = to_z(y, ell, gscale)

    def fz(zz):
        fy = rhs(from_z(zz, ell, gscale), sigma)
        fx, fg = unpack(fy)
        return pack(fx / ell, fg / gscale)

    return numerical_jacobian_map(fz, z)


def tangent_product(y0, sigma, horizon, nsteps, method, ell, gscale):
    h = horizon / nsteps
    y = y0.copy()
    product = np.eye(len(y))
    step_jacs = []
    for _ in range(nsteps):
        z = to_z(y, ell, gscale)

        def one_step_z(zz):
            yy = from_z(zz, ell, gscale)
            return to_z(step(yy, h, sigma, method), ell, gscale)

        jac = numerical_jacobian_map(one_step_z, z)
        step_jacs.append(float(np.linalg.svd(jac, compute_uv=False)[0]))
        product = jac @ product
        y = step(y, h, sigma, method)
    return y, product, step_jacs


def tangent_study():
    rows = []
    horizon, nsteps = 0.30, 6
    for cname in ["regular", "cluster_void"]:
        x, gamma, _ = cloud(cname, side=2)
        y0 = pack(x, gamma)
        ell, gscale = scales(y0)
        for ratio in [0.6, 1.5]:
            sigma = np.full(len(x), ratio)
            # Time-dependent exact-flow diagnostics along a fine RK4 path.
            y = y0.copy()
            fine_steps = 64
            fine_h = horizon / fine_steps
            mus, alphas, times = [], [], []
            for k in range(fine_steps + 1):
                if k % 16 == 0:
                    jac = rhs_jacobian_scaled(y, sigma, ell, gscale)
                    mus.append(float(np.linalg.eigvalsh(0.5 * (jac + jac.T))[-1]))
                    alphas.append(float(np.max(np.linalg.eigvals(jac).real)))
                    times.append(k * fine_h)
                if k < fine_steps:
                    y = rk4(y, fine_h, sigma)
            sampled_log_norm_exponential = math.exp(float(np.trapezoid(mus, times)))

            z0 = to_z(y0, ell, gscale)

            def reference_map_z(zz):
                yy = from_z(zz, ell, gscale)
                return to_z(reference_integrate(yy, sigma, horizon, 96), ell, gscale)

            ref_tangent = numerical_jacobian_map(reference_map_z, z0)
            ref_smax_estimate = float(np.linalg.svd(ref_tangent, compute_uv=False)[0])
            for method in ["coupled_ssprk3", "historical_lie", "symmetric_strang"]:
                final, product, step_smax = tangent_product(
                    y0, sigma, horizon, nsteps, method, ell, gscale
                )

                def full_map_z(zz):
                    yy = from_z(zz, ell, gscale)
                    out = integrate(yy, sigma, horizon, nsteps, method)
                    return to_z(out, ell, gscale)

                direct = numerical_jacobian_map(full_map_z, z0)
                smax = float(np.linalg.svd(product, compute_uv=False)[0])
                rows.append(
                    {
                        "cloud": cname,
                        "sigma_over_ell": ratio,
                        "method": method,
                        "dt": horizon / nsteps,
                        "sampled_rhs_log_norm_max": max(mus),
                        "sampled_rhs_spectral_abscissa_max": max(alphas),
                        "sampled_nonnormal_gap_max": max(np.array(mus) - np.array(alphas)),
                        "sampled_exp_trapezoid_log_norm": sampled_log_norm_exponential,
                        "refined_flow_tangent_smax_estimate": ref_smax_estimate,
                        "numerical_tangent_product_smax_estimate": smax,
                        "numerical_over_refined_tangent_estimate": smax / ref_smax_estimate,
                        "max_one_step_tangent_smax_estimate": max(step_smax),
                        "product_direct_relative_difference_estimate": float(
                            np.linalg.norm(product - direct, ord=2)
                            / max(np.linalg.norm(direct, ord=2), 1e-15)
                        ),
                        "final_state_norm": float(np.linalg.norm(final)),
                    }
                )
    return rows


def temporal_study():
    geometry, rows = [], []
    horizon = 0.40
    clouds = ["regular", "jitter_a", "jitter_b", "cluster_void"]
    ratios = [0.6, 1.0, 1.5]
    methods = ["coupled_ssprk3", "historical_lie", "symmetric_strang"]
    for cname in clouds:
        x, gamma, volume = cloud(cname)
        y0 = pack(x, gamma)
        ell, gscale = scales(y0)
        for ratio in ratios:
            sigma = np.full(len(x), ratio)
            geometry.append(geometry_metrics(cname, x, volume, ratio))
            reference = reference_integrate(y0, sigma, horizon, 512)
            for method in methods:
                previous_error = None
                for nsteps in [8, 16, 32, 64]:
                    result = integrate(y0, sigma, horizon, nsteps, method)
                    error = scaled_error(result, reference, ell, gscale)
                    order = (
                        math.log(previous_error / error, 2.0)
                        if previous_error is not None and error > 0.0
                        else None
                    )
                    rows.append(
                        {
                            "cloud": cname,
                            "sigma_over_ell": ratio,
                            "method": method,
                            "horizon": horizon,
                            "nsteps": nsteps,
                            "dt": horizon / nsteps,
                            "scaled_rms_error": error,
                            "observed_order_from_previous": order,
                            "rhs_evaluations_per_step": {
                                "coupled_ssprk3": 3,
                                "historical_lie": 6,
                                "symmetric_strang": 9,
                            }[method],
                        }
                    )
                    previous_error = error
    return geometry, rows


def write_csv(name, rows):
    if not rows:
        return
    with (ROOT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figures(geometry, temporal, tangent):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), constrained_layout=True)
    representative = [
        ("regular", 0.6),
        ("regular", 1.5),
        ("cluster_void", 0.6),
    ]
    labels = {
        "coupled_ssprk3": "coupled SSPRK3",
        "historical_lie": "historical A→B",
        "symmetric_strang": "symmetric A/2→B→A/2",
    }
    for ax, (cname, ratio) in zip(axes, representative):
        for method, label in labels.items():
            subset = [
                r
                for r in temporal
                if r["cloud"] == cname
                and r["sigma_over_ell"] == ratio
                and r["method"] == method
            ]
            subset.sort(key=lambda r: r["dt"])
            ax.loglog(
                [r["dt"] for r in subset],
                [r["scaled_rms_error"] for r in subset],
                marker="o",
                label=label,
            )
        display_name = cname.replace("cluster_void", "cluster/void")
        ax.set_title(f"{display_name}, σ/ℓ={ratio}")
        ax.set_xlabel("Δt")
        tested_dt = [0.00625, 0.0125, 0.025, 0.05]
        ax.set_xticks(tested_dt, ["0.00625", "0.0125", "0.025", "0.05"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("scaled RMS state error")
    axes[-1].legend(fontsize=8)
    fig.savefig(ROOT / "temporal_error.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    clouds = ["regular", "jitter_a", "jitter_b", "cluster_void"]
    markers = ["o", "s", "^", "D"]
    for cname, marker in zip(clouds, markers):
        subset = [r for r in geometry if r["cloud"] == cname]
        subset.sort(key=lambda r: r["sigma_over_volume_spacing"])
        ax.semilogy(
            [r["sigma_over_volume_spacing"] for r in subset],
            [r["zeroth_moment_abs_rms"] for r in subset],
            marker=marker,
            label=f"{cname}: zeroth-moment defect",
        )
        ax.semilogy(
            [r["sigma_over_volume_spacing"] for r in subset],
            [r["first_moment_norm_rms"] for r in subset],
            marker=marker,
            linestyle="--",
            label=f"{cname}: first-moment defect",
        )
    ax.set_xlabel("σ / median(Vᵢ¹ᐟ³)")
    ax.set_ylabel("finite-cloud kernel moment defect")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(ncol=2, fontsize=7)
    fig.savefig(ROOT / "geometry_conditioning.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.5), constrained_layout=True)
    labels_x, values, colors = [], [], []
    palette = {
        "coupled_ssprk3": "#1f77b4",
        "historical_lie": "#d62728",
        "symmetric_strang": "#2ca02c",
    }
    for r in tangent:
        labels_x.append(f"{r['cloud'][:3]}\n{r['sigma_over_ell']}\n{r['method'].split('_')[0]}")
        values.append(1.0e6 * (r["numerical_over_refined_tangent_estimate"] - 1.0))
        colors.append(palette[r["method"]])
    ax.bar(range(len(values)), values, color=colors)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(range(len(values)), labels_x, fontsize=7)
    ax.set_ylabel("excess tangent amplification over reference (ppm)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(ROOT / "tangent_growth.png", dpi=180)
    plt.close(fig)


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    started = time.perf_counter()
    deriv = derivative_checks()
    geometry, temporal = temporal_study()
    tangent = tangent_study()
    make_figures(geometry, temporal, tangent)
    elapsed = time.perf_counter() - started
    payload = {
        "metadata": {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "elapsed_seconds": elapsed,
            "max_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "single_thread_environment": {
                key: os.environ.get(key)
                for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]
            },
        },
        "kernel": {
            "zeta": "pi^(-3/2) exp(-rho^2)",
            "q": "[erf(rho)-2 rho exp(-rho^2)/sqrt(pi)]/(4 pi)",
            "q_evaluator": "production Abramowitz-Stegun approximation with rho<0.2 series",
            "pair_core": "(sigma_i+sigma_j)/2",
            "stretching": "(grad u)^T Gamma (historical/default TRANSPOSED mode)",
            "crossover_check": kernel_crossover_check(),
        },
        "norms": {
            "temporal_error": "RMS of full scaled Euclidean state difference; X/L and Gamma/G",
            "tangent": "induced 2-norm of centered-difference maps in full scaled Euclidean coordinates",
            "row_sum_theory": "separate scaled maximum over per-particle Euclidean X/Gamma blocks",
        },
        "derivative_checks": deriv,
        "geometry": geometry,
        "temporal": temporal,
        "tangent": tangent,
    }
    (ROOT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_csv("derivative_checks.csv", deriv)
    write_csv("kernel_crossover.csv", [payload["kernel"]["crossover_check"]])
    write_csv("geometry.csv", geometry)
    write_csv("temporal.csv", temporal)
    write_csv("tangent.csv", tangent)
    print(json.dumps(payload["metadata"], indent=2))


if __name__ == "__main__":
    main()
