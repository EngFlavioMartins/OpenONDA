#!/usr/bin/env python3
"""Gate E.0: manufactured SGS-torque coupling through VPM remeshing.

This test does not evaluate the DIAD closure.  It verifies the representation
contract that any explicit vorticity LES source must satisfy in OpenONDA:

    d Gamma_p / dt = V_p g_SGS(x_p).

A smooth, divergence-free manufactured torque has exact circulation and
linear-impulse source integrals.  Particle increments are deposited with the
production M4' remeshing kernel, then compared with the analytical grid field
under refinement, particle disorder, and f32/f64 accumulation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/openonda_stage5a_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/openonda_stage5a_cache")

import matplotlib.pyplot as plt
import numpy as np

from source.solvers.VPM.physics.diffusion.grid import _m4_prime_1d

TIME_STEP_SIZE = 0.03
EXACT_IMPULSE_RATE_Z = 27.0 / 512.0
INK = "#20252a"
BLUE = "#286f9b"
GOLD = "#d9973b"
GREY = "#8a99a8"
GRID = "#d8dde2"


def manufactured_torque(position: np.ndarray) -> np.ndarray:
    """Return curl(0, 0, psi), psi=prod_d sin(pi*x_d)^4 on [0,1]^3."""
    x, y, z = position.T
    sx, sy, sz = np.sin(np.pi * x), np.sin(np.pi * y), np.sin(np.pi * z)
    cx, cy = np.cos(np.pi * x), np.cos(np.pi * y)
    torque = np.zeros_like(position, dtype=np.float64)
    torque[:, 0] = 4.0 * np.pi * sx**4 * sy**3 * cy * sz**4
    torque[:, 1] = -4.0 * np.pi * sx**3 * cx * sy**4 * sz**4
    return torque


def particle_lattice(
    n: int,
    jitter_fraction: float,
    seed: int,
    disorder: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    h = 1.0 / (n - 1)
    axis = np.linspace(0.0, 1.0, n)
    i, j, k = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    position = np.column_stack((axis[i.ravel()], axis[j.ravel()], axis[k.ravel()]))
    boundary = (
        (i.ravel() == 0)
        | (i.ravel() == n - 1)
        | (j.ravel() == 0)
        | (j.ravel() == n - 1)
        | (k.ravel() == 0)
        | (k.ravel() == n - 1)
    )
    if jitter_fraction > 0.0:
        if disorder == "random":
            rng = np.random.default_rng(seed)
            displacement = rng.uniform(-jitter_fraction, jitter_fraction, size=position.shape) * h
            displacement[boundary] = 0.0
        elif disorder == "volume_preserving_shear":
            # Triangular flow map with det(F)=1 exactly. Its O(h) amplitude
            # represents the relative deformation accumulated over one
            # CFL-scaled particle/remeshing step; a uniform translation is
            # irrelevant because the GBD grid origin follows the cloud.
            amplitude = jitter_fraction * h
            displacement = np.zeros_like(position)
            displacement[:, 0] = amplitude * np.sin(2.0 * np.pi * position[:, 1])
            displacement[:, 1] = amplitude * np.sin(2.0 * np.pi * position[:, 2])
        else:
            raise ValueError(f"unknown disorder mode {disorder!r}")
        position += displacement
    volume = np.full(len(position), h**3)
    return position, volume, h


def m4_scatter(
    position: np.ndarray,
    increment: np.ndarray,
    grid_min: np.ndarray,
    h: float,
    shape: tuple[int, int, int],
    dtype: np.dtype,
) -> np.ndarray:
    """NumPy audit of the production 4^3 M4' deposit, including its weights."""
    grid = np.zeros((*shape, 3), dtype=dtype)
    flat = grid.reshape(-1, 3)
    fractional = (position - grid_min) / h
    base = np.floor(fractional).astype(np.int64)
    nx, ny, nz = shape
    values = np.asarray(increment, dtype=dtype)
    for di in range(-1, 3):
        ii = base[:, 0] + di
        wx = _m4_prime_1d(np.abs(fractional[:, 0] - ii))
        for dj in range(-1, 3):
            jj = base[:, 1] + dj
            wy = _m4_prime_1d(np.abs(fractional[:, 1] - jj))
            for dk in range(-1, 3):
                kk = base[:, 2] + dk
                wz = _m4_prime_1d(np.abs(fractional[:, 2] - kk))
                valid = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny) & (kk >= 0) & (kk < nz)
                linear = ii[valid] * (ny * nz) + jj[valid] * nz + kk[valid]
                weight = np.asarray(wx[valid] * wy[valid] * wz[valid], dtype=dtype)
                np.add.at(flat, linear, weight[:, None] * values[valid])
    return grid


def impulse_rate(position: np.ndarray, circulation_rate: np.ndarray) -> np.ndarray:
    return 0.5 * np.sum(np.cross(position, circulation_rate), axis=0)


def run_case(
    n: int,
    jitter: float,
    dtype_name: str,
    seed: int,
    disorder: str = "volume_preserving_shear",
) -> dict[str, float | int | str]:
    dtype = np.dtype(dtype_name)
    position, volume, h = particle_lattice(n, jitter, seed, disorder)
    torque = manufactured_torque(position)
    increment = np.asarray(TIME_STEP_SIZE * volume[:, None] * torque, dtype=dtype)
    padding = 2
    shape = (n + 2 * padding,) * 3
    grid_min = np.full(3, -padding * h)
    deposited = m4_scatter(position, increment, grid_min, h, shape, dtype)
    central = deposited[padding : padding + n, padding : padding + n, padding : padding + n]

    axis = np.linspace(0.0, 1.0, n)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    target = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    exact = manufactured_torque(target).reshape(n, n, n, 3)
    recovered = np.asarray(central, dtype=np.float64) / (TIME_STEP_SIZE * h**3)
    relative_l2 = float(np.linalg.norm(recovered - exact) / np.linalg.norm(exact))

    source_rate = np.asarray(increment, dtype=np.float64) / TIME_STEP_SIZE
    deposited_rate = np.asarray(deposited, dtype=np.float64) / TIME_STEP_SIZE
    source_sum = source_rate.sum(axis=0)
    deposited_sum = deposited_rate.sum(axis=(0, 1, 2))
    scale = float(np.sum(np.linalg.norm(source_rate, axis=1)))
    scatter_sum_error = float(np.linalg.norm(deposited_sum - source_sum) / scale)
    zero_circulation_error = float(np.linalg.norm(deposited_sum) / scale)

    source_impulse = impulse_rate(position, source_rate)
    gx, gy, gz = np.meshgrid(
        grid_min[0] + h * np.arange(shape[0]),
        grid_min[1] + h * np.arange(shape[1]),
        grid_min[2] + h * np.arange(shape[2]),
        indexing="ij",
    )
    grid_position = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))
    deposited_impulse = impulse_rate(grid_position, deposited_rate.reshape(-1, 3))
    exact_impulse = np.array((0.0, 0.0, EXACT_IMPULSE_RATE_Z))
    impulse_relative_error = float(
        np.linalg.norm(deposited_impulse - exact_impulse) / np.linalg.norm(exact_impulse)
    )
    scatter_impulse_error = float(
        np.linalg.norm(deposited_impulse - source_impulse) / np.linalg.norm(exact_impulse)
    )
    return {
        "n": n,
        "particle_spacing": h,
        "particles": len(position),
        "jitter_over_h": jitter,
        "disorder": disorder,
        "dtype": dtype_name,
        "field_relative_l2": relative_l2,
        "zero_circulation_relative": zero_circulation_error,
        "scatter_circulation_relative": scatter_sum_error,
        "impulse_rate_z": float(deposited_impulse[2]),
        "impulse_relative_error": impulse_relative_error,
        "scatter_impulse_relative": scatter_impulse_error,
    }


def convergence_order(cases: list[dict[str, float | int | str]]) -> float:
    h = np.asarray([case["particle_spacing"] for case in cases], dtype=float)
    error = np.asarray([case["field_relative_l2"] for case in cases], dtype=float)
    return float(np.polyfit(np.log(h), np.log(error), 1)[0])


def evaluate(resolutions: tuple[int, ...], seed: int) -> dict[str, object]:
    cases = []
    for dtype in ("float64", "float32"):
        for n in resolutions:
            cases.append(run_case(n, 0.0, dtype, seed + n))
            cases.append(run_case(n, 0.15, dtype, seed + n))
    fine_n = max(resolutions)
    random_stress = [
        run_case(fine_n, value, "float32", seed + fine_n, disorder="random")
        for value in (0.0, 0.05, 0.1, 0.15, 0.2)
    ]
    jittered = {
        dtype: [case for case in cases if case["dtype"] == dtype and case["jitter_over_h"] == 0.15]
        for dtype in ("float64", "float32")
    }
    aligned = {
        dtype: [case for case in cases if case["dtype"] == dtype and case["jitter_over_h"] == 0.0]
        for dtype in ("float64", "float32")
    }
    orders = {dtype: convergence_order(selected) for dtype, selected in jittered.items()}
    fine_jitter = {dtype: selected[-1] for dtype, selected in jittered.items()}
    fine_aligned = {dtype: selected[-1] for dtype, selected in aligned.items()}
    all_cases = [*cases, *random_stress]
    checks = {
        "aligned_f64_is_exact": fine_aligned["float64"]["field_relative_l2"] < 1.0e-12,
        "aligned_f32_below_5e-6": fine_aligned["float32"]["field_relative_l2"] < 5.0e-6,
        "jittered_f64_below_5_percent": fine_jitter["float64"]["field_relative_l2"] < 0.05,
        "jittered_f32_below_5_percent": fine_jitter["float32"]["field_relative_l2"] < 0.05,
        "jittered_convergence_order_above_1p5": min(orders.values()) > 1.5,
        "m4_preserves_source_circulation": max(
            case["scatter_circulation_relative"] for case in all_cases
        )
        < 2.0e-6,
        "m4_preserves_source_impulse": max(case["scatter_impulse_relative"] for case in all_cases)
        < 2.0e-5,
        "fine_impulse_within_2_percent": max(
            fine_jitter[dtype]["impulse_relative_error"] for dtype in fine_jitter
        )
        < 0.02,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "claim": "particle circulation-source mapping only; DIAD closure not evaluated",
        "manufactured_field": "g = curl(0,0,prod_d sin(pi*x_d)^4)",
        "exact_integrals": {
            "volume_integral_g": [0.0, 0.0, 0.0],
            "linear_impulse_rate": [0.0, 0.0, EXACT_IMPULSE_RATE_Z],
        },
        "checks": checks,
        "convergence_orders": orders,
        "cases": cases,
        "random_disorder_stress_cases": random_stress,
        "random_disorder_scope": (
            "diagnostic only: independent jitter with fixed particle volumes is not an "
            "incompressible flow map"
        ),
    }


def plot(result: dict[str, object], output: Path) -> None:
    cases = result["cases"]
    disorder = result["random_disorder_stress_cases"]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.1), constrained_layout=True)

    axis = axes[0]
    for dtype, color, marker in (("float64", BLUE, "o"), ("float32", GOLD, "s")):
        selected = [
            case for case in cases if case["dtype"] == dtype and case["jitter_over_h"] == 0.15
        ]
        h = np.asarray([case["particle_spacing"] for case in selected])
        error = np.asarray([case["field_relative_l2"] for case in selected])
        order = result["convergence_orders"][dtype]
        axis.loglog(h, error, color=color, marker=marker, label=f"{dtype}, slope {order:.2f}")
    guide_h = np.asarray([min(h), max(h)])
    guide = 0.7 * error[-1] * (guide_h / h[-1]) ** 2
    axis.loglog(guide_h, guide, color=GREY, linestyle="--", label=r"$O(h^2)$ guide")
    axis.set_xlabel(r"particle spacing $h$")
    axis.set_ylabel(r"relative $L_2$ torque error")
    axis.set_title(r"M4$'$ transfer, volume-preserving shear")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1]
    jitter = 100.0 * np.asarray([case["jitter_over_h"] for case in disorder])
    error = 100.0 * np.asarray([case["field_relative_l2"] for case in disorder])
    axis.plot(jitter, error, color=BLUE, marker="o")
    axis.axhline(8.0, color=GOLD, linestyle="--", label="8% gate")
    axis.set_xlabel(r"particle jitter $(\%h)$")
    axis.set_ylabel("torque error (%)")
    axis.set_title("Independent-jitter stress test, f32")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[2]
    fine = [
        case
        for case in cases
        if case["n"] == max(case["n"] for case in cases) and case["jitter_over_h"] == 0.15
    ]
    labels = [str(case["dtype"]).replace("float", "f") for case in fine]
    values = [case["impulse_rate_z"] for case in fine]
    axis.bar(labels, values, color=(BLUE, GOLD), edgecolor=INK, linewidth=0.7)
    axis.axhline(EXACT_IMPULSE_RATE_Z, color=INK, linestyle="--", label=r"theory $27/512$")
    axis.set_ylim(0.0, 1.18 * EXACT_IMPULSE_RATE_Z)
    axis.set_ylabel(r"$dI_z/dt$")
    axis.set_title("Linear-impulse source")
    axis.legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.grid(color=GRID, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("VPM manufactured SGS-torque coupling gate", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", type=int, nargs="+", default=(12, 16, 24, 32))
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(tuple(args.resolutions), args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    plot(result, args.figure)
    if result["status"] != "PASS":
        raise SystemExit("VPM TORQUE COUPLING GATE FAIL")


if __name__ == "__main__":
    main()
