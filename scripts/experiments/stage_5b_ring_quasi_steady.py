#!/usr/bin/env python3
"""Verify and apply the axisymmetric vortex-ring quasi-steady diagnostic.

For a steadily translating, inviscid, axisymmetric flow without swirl,

    q = omega_phi / r = F(psi),

where psi is the Stokes streamfunction in the translating frame.  The script
checks that numerical measurements recover an exact manufactured relation,
reject a deliberately broken relation, and then measures the raw OpenONDA
Gaussian-ring states on successively refined meridional grids.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/openonda_qpsi_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/openonda_qpsi_cache")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from source.solvers.VPM import Solver  # noqa: E402
from source.solvers.VPM.io import BackupSystem  # noqa: E402

RING_RADIUS = 1.0
RING_CIRCULATION = 1.0
CORE_RADIUS = 0.4131
MEASURED_SPEED = 0.19136286820937143
GAUSSIAN_SPEED = 0.1918387673489999
RELAXED_EMPIRICAL_SPEED = 0.16544471931110244
GRID_SIZES = (65, 97, 129)
CORE_FRACTIONS = (0.02, 0.05, 0.10)
PRIMARY_CORE_FRACTION = 0.05
QUANTILE_BINS = 32

RUN_DIRECTORY = ROOT / (
    "tutorials/VPM/vortexRing/solution/relaxed_reference_tail002_cs_h012_dt002_tstar02"
)
BACKUP_PREFIX = "vpm_relaxed_reference_tail002_cs_h012_dt002_tstar02"
FINAL_BACKUP = RUN_DIRECTORY / f"{BACKUP_PREFIX}_final"
INITIAL_H5 = RUN_DIRECTORY / f"{BACKUP_PREFIX}_000000.h5"

INK = "#20252a"
BLUE = "#286f9b"
GOLD = "#d9973b"
GREEN = "#35845d"
RED = "#b84a4a"
GREY = "#8a99a8"
GRID = "#d8dde2"


def cumulative_streamfunction(
    axial_velocity: np.ndarray,
    radial_coordinate: np.ndarray,
    translation_speed: float,
) -> np.ndarray:
    """Integrate d(psi)/dr = r (u_x - U), with psi=0 on the axis."""
    relative_axial = axial_velocity - translation_speed
    integrand = relative_axial * radial_coordinate[None, :]
    increments = 0.5 * (integrand[:, 1:] + integrand[:, :-1]) * np.diff(radial_coordinate)[None, :]
    psi = np.zeros_like(axial_velocity, dtype=np.float64)
    psi[:, 1:] = np.cumsum(increments, axis=1)
    return psi


def local_single_value_residual(
    psi: np.ndarray,
    q: np.ndarray,
    mask: np.ndarray,
    *,
    bins: int = QUANTILE_BINS,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return normalized local-linear scatter about a single q=F(psi) curve."""
    psi_values = np.asarray(psi[mask], dtype=np.float64)
    q_values = np.asarray(q[mask], dtype=np.float64)
    if len(psi_values) < 4 * bins:
        raise ValueError("too few core samples for the requested q(psi) bins")

    order = np.argsort(psi_values)
    groups = np.array_split(order, bins)
    prediction = np.empty_like(q_values)
    curve_psi: list[float] = []
    curve_q: list[float] = []
    for group in groups:
        p = psi_values[group]
        values = q_values[group]
        p_center = float(np.mean(p))
        q_center = float(np.mean(values))
        p_shift = p - p_center
        denominator = float(np.dot(p_shift, p_shift))
        slope = float(np.dot(p_shift, values - q_center) / denominator) if denominator else 0.0
        prediction[group] = q_center + slope * p_shift
        curve_psi.append(p_center)
        curve_q.append(q_center)

    scale = float(np.sqrt(np.mean(q_values**2)))
    residual = float(np.sqrt(np.mean((q_values - prediction) ** 2)) / scale)
    return residual, np.asarray(curve_psi), np.asarray(curve_q)


def field_metrics(
    x: np.ndarray,
    r: np.ndarray,
    velocity: np.ndarray,
    omega_phi: np.ndarray,
    *,
    translation_speed: float,
    core_fraction: float,
    bins: int = QUANTILE_BINS,
    ring_radius: float = RING_RADIUS,
    circulation: float = RING_CIRCULATION,
) -> dict[str, object]:
    """Measure q(psi) collapse and the translating-frame material residual."""
    ux = np.asarray(velocity[..., 0], dtype=np.float64)
    ur = np.asarray(velocity[..., 1], dtype=np.float64)
    omega_phi = np.asarray(omega_phi, dtype=np.float64)
    rr = r[None, :]
    q = np.divide(
        omega_phi,
        rr,
        out=np.zeros_like(omega_phi, dtype=np.float64),
        where=rr > 0.0,
    )
    peak = float(np.max(omega_phi))
    mask = (rr > 0.0) & (omega_phi >= core_fraction * peak)
    mask[[0, -1], :] = False
    mask[:, [0, -1]] = False
    psi = cumulative_streamfunction(ux, r, translation_speed)

    collapse, curve_psi, curve_q = local_single_value_residual(psi, q, mask, bins=bins)
    dq_dx, dq_dr = np.gradient(q, x, r, edge_order=2)
    material_rate = (ux - translation_speed) * dq_dx + ur * dq_dr
    q_rms = float(np.sqrt(np.mean(q[mask] ** 2)))
    material_rms = float(np.sqrt(np.mean(material_rate[mask] ** 2)))
    advective_residual = material_rms * ring_radius**2 / (circulation * q_rms)

    base_rate = ux * dq_dx + ur * dq_dr
    qx = dq_dx[mask]
    denominator = float(np.dot(qx, qx))
    fitted_speed = (
        float(np.dot(base_rate[mask], qx) / denominator)
        if denominator > np.finfo(float).tiny
        else translation_speed
    )
    fitted_rate = base_rate - fitted_speed * dq_dx
    fitted_advective_residual = (
        float(np.sqrt(np.mean(fitted_rate[mask] ** 2))) * ring_radius**2 / (circulation * q_rms)
    )

    dpsi_dx, dpsi_dr = np.gradient(psi, x, r, edge_order=2)
    recovered_ux = (
        np.divide(
            dpsi_dr,
            rr,
            out=np.zeros_like(dpsi_dr),
            where=rr > 0.0,
        )
        + translation_speed
    )
    recovered_ur = np.divide(
        -dpsi_dx,
        rr,
        out=np.zeros_like(dpsi_dx),
        where=rr > 0.0,
    )
    velocity_scale = float(np.sqrt(np.mean((ux[mask] - translation_speed) ** 2 + ur[mask] ** 2)))
    streamfunction_velocity_error = float(
        np.sqrt(
            np.mean((recovered_ux[mask] - ux[mask]) ** 2 + (recovered_ur[mask] - ur[mask]) ** 2)
        )
        / velocity_scale
    )

    return {
        "collapse_residual": collapse,
        "advective_residual": advective_residual,
        "fitted_advective_residual": fitted_advective_residual,
        "fitted_translation_speed": fitted_speed,
        "fitted_speed_relative_difference": abs(fitted_speed / translation_speed - 1.0)
        if translation_speed
        else None,
        "streamfunction_velocity_relative_error": streamfunction_velocity_error,
        "core_samples": int(np.count_nonzero(mask)),
        "peak_omega_phi": peak,
        "psi": psi,
        "q": q,
        "mask": mask,
        "curve_psi": curve_psi,
        "curve_q": curve_q,
    }


def manufactured_controls(grid_size: int) -> dict[str, dict[str, float]]:
    """Exact q=F(psi) control and the same field with a broken relation."""
    x = np.linspace(-0.9, 0.9, grid_size)
    r = np.linspace(0.0, 1.4, grid_size)
    xx, rr = np.meshgrid(x, r, indexing="ij")
    exponential = np.exp(-(xx**2 + rr**2))
    psi = rr**2 * exponential
    velocity = np.zeros((*xx.shape, 3), dtype=np.float64)
    velocity[..., 0] = 2.0 * exponential * (1.0 - rr**2)
    velocity[..., 1] = 2.0 * xx * rr * exponential
    q_exact = 1.0 + 0.8 * psi
    omega_exact = rr * q_exact
    q_broken = q_exact + 0.45 * xx
    omega_broken = rr * q_broken

    # Use an explicit interior mask encoded through a uniform positive omega;
    # field_metrics also removes the finite-difference boundary points.
    exact = field_metrics(
        x,
        r,
        velocity,
        omega_exact,
        translation_speed=0.0,
        core_fraction=0.05,
        ring_radius=1.0,
        circulation=1.0,
    )
    broken = field_metrics(
        x,
        r,
        velocity,
        omega_broken,
        translation_speed=0.0,
        core_fraction=0.05,
        ring_radius=1.0,
        circulation=1.0,
    )
    keep = (
        "collapse_residual",
        "advective_residual",
        "fitted_advective_residual",
        "streamfunction_velocity_relative_error",
    )
    return {
        "exact_q_equals_one_plus_0p8_psi": {key: float(exact[key]) for key in keep},
        "broken_q_adds_0p45_x": {key: float(broken[key]) for key in keep},
    }


def sample_solver(
    solver: Solver,
    grid_size: int,
    *,
    axial_center: float,
    translation_speed: float,
) -> dict[str, object]:
    x = axial_center + np.linspace(-0.85, 0.85, grid_size)
    r = np.linspace(0.0, 1.75, grid_size)
    xx, rr = np.meshgrid(x, r, indexing="ij")
    targets = np.column_stack((xx.ravel(), rr.ravel(), np.zeros(xx.size)))
    velocity = solver.compute_target_velocities(targets, include_freestream=False).reshape(
        grid_size, grid_size, 3
    )
    vorticity = solver.compute_target_vorticities(targets).reshape(grid_size, grid_size, 3)
    sensitivity: dict[str, dict[str, object]] = {}
    for fraction in CORE_FRACTIONS:
        metrics = field_metrics(
            x,
            r,
            velocity,
            vorticity[..., 2],
            translation_speed=translation_speed,
            core_fraction=fraction,
        )
        sensitivity[f"{fraction:.2f}"] = metrics
    return {"x": x, "r": r, "sensitivity": sensitivity}


def serializable_metrics(metrics: dict[str, object]) -> dict[str, object]:
    hidden = {"psi", "q", "mask", "curve_psi", "curve_q"}
    return {key: value for key, value in metrics.items() if key not in hidden}


def relative_change(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), np.finfo(float).tiny)


def evaluate_gate(
    controls: dict[str, dict[str, float]],
    final_samples: dict[int, dict[str, object]],
) -> dict[str, object]:
    exact = controls["exact_q_equals_one_plus_0p8_psi"]
    broken = controls["broken_q_adds_0p45_x"]
    coarse = final_samples[GRID_SIZES[0]]["sensitivity"][f"{PRIMARY_CORE_FRACTION:.2f}"]
    fine = final_samples[GRID_SIZES[-1]]["sensitivity"][f"{PRIMARY_CORE_FRACTION:.2f}"]
    convergence = {
        key: relative_change(float(coarse[key]), float(fine[key]))
        for key in ("collapse_residual", "advective_residual", "fitted_translation_speed")
    }
    checks = {
        "exact_functional_relation": exact["collapse_residual"] < 5.0e-3,
        "exact_material_invariance": exact["advective_residual"] < 2.0e-2,
        "broken_relation_detected": (
            broken["collapse_residual"] > max(0.05, 10.0 * exact["collapse_residual"])
            and broken["advective_residual"] > max(0.05, 10.0 * exact["advective_residual"])
        ),
        "streamfunction_reconstruction": float(fine["streamfunction_velocity_relative_error"])
        < 3.0e-2,
        "actual_grid_sensitivity": max(convergence.values()) < 0.10,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "coarse_to_fine_relative_changes": convergence,
        "meaning": (
            "This gate validates the measurement. It does not declare the t*=0.2 ring "
            "quasi-steady; that requires a later time plateau."
        ),
    }


def plot_results(
    controls: dict[str, dict[str, float]],
    initial: dict[str, object],
    final_samples: dict[int, dict[str, object]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.7), constrained_layout=True)
    fig.suptitle(
        r"Vortex-ring relaxation criterion: $q=\omega_\phi/r=F(\psi)$",
        color=INK,
        fontsize=13,
        fontweight="bold",
    )

    axis = axes[0, 0]
    initial_metrics = initial["sensitivity"][f"{PRIMARY_CORE_FRACTION:.2f}"]
    final_metrics = final_samples[GRID_SIZES[-1]]["sensitivity"][f"{PRIMARY_CORE_FRACTION:.2f}"]
    for metrics, color, label, alpha in (
        (initial_metrics, GREY, r"raw $t^*=0$", 0.25),
        (final_metrics, BLUE, r"raw $t^*=0.2$", 0.32),
    ):
        mask = metrics["mask"]
        psi_values = metrics["psi"][mask]
        q_values = metrics["q"][mask]
        axis.scatter(
            psi_values / np.max(np.abs(psi_values)),
            q_values / np.max(np.abs(q_values)),
            s=3,
            color=color,
            alpha=alpha,
            rasterized=True,
            label=label,
        )
        axis.plot(
            metrics["curve_psi"] / np.max(np.abs(psi_values)),
            metrics["curve_q"] / np.max(np.abs(q_values)),
            color=color,
            linewidth=2.0,
        )
    axis.set_xlabel(r"translating-frame streamfunction $\psi/\max|\psi|$")
    axis.set_ylabel(r"$q/\max|q|$")
    axis.set_title("A steady ring collapses onto one curve")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[0, 1]
    sizes = np.asarray(GRID_SIZES)
    collapse = []
    advection = []
    for size in GRID_SIZES:
        metrics = final_samples[size]["sensitivity"][f"{PRIMARY_CORE_FRACTION:.2f}"]
        collapse.append(float(metrics["collapse_residual"]))
        advection.append(float(metrics["advective_residual"]))
    axis.plot(sizes, collapse, "o-", color=BLUE, label=r"scatter about $F(\psi)$")
    axis.plot(sizes, advection, "s-", color=GOLD, label=r"$\mathbf{u}_{rel}\cdot\nabla q$")
    axis.axhline(0.0, color=INK, linewidth=1.0, linestyle="--", label="exact steady value")
    axis.set_xlabel("points in each meridional direction")
    axis.set_ylabel("dimensionless residual")
    axis.set_title("The measured residual must converge with the grid")
    axis.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1, 0]
    exact = controls["exact_q_equals_one_plus_0p8_psi"]
    broken = controls["broken_q_adds_0p45_x"]
    labels = [r"$q=1+0.8\psi$", r"same field $+0.45x$"]
    positions = np.arange(2)
    width = 0.34
    axis.bar(
        positions - width / 2,
        [exact["collapse_residual"], broken["collapse_residual"]],
        width,
        color=BLUE,
        label=r"$q(\psi)$ scatter",
    )
    axis.bar(
        positions + width / 2,
        [exact["advective_residual"], broken["advective_residual"]],
        width,
        color=GOLD,
        label="material residual",
    )
    axis.set_xticks(positions, labels)
    axis.set_yscale("log")
    axis.set_ylabel("dimensionless residual")
    axis.set_title("Known exact relation passes; broken relation fails")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1, 1]
    finest = final_samples[GRID_SIZES[-1]]["sensitivity"][f"{PRIMARY_CORE_FRACTION:.2f}"]
    values = [
        MEASURED_SPEED,
        float(finest["fitted_translation_speed"]),
        GAUSSIAN_SPEED,
        RELAXED_EMPIRICAL_SPEED,
    ]
    labels = ["measured\nVPM", "best steady\nframe", "Gaussian\ntheory", "relaxed-core\nformula"]
    axis.bar(np.arange(4), values, color=[BLUE, GREEN, INK, GOLD])
    axis.set_xticks(np.arange(4), labels)
    axis.set_ylabel(r"$UR/\Gamma$")
    axis.set_title("The steady-frame speed is independently identified")
    axis.set_ylim(0.0, max(values) * 1.22)

    for axis in axes.flat:
        axis.grid(True, color=GRID, linewidth=0.7, alpha=0.75)
        axis.spines[["top", "right"]].set_visible(False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main() -> None:
    controls = manufactured_controls(GRID_SIZES[-1])
    solver = Solver.continue_from_backup(str(FINAL_BACKUP))
    if solver is None:
        raise RuntimeError(f"could not restore {FINAL_BACKUP}")

    final_samples: dict[int, dict[str, object]] = {}
    for size in GRID_SIZES:
        final_samples[size] = sample_solver(
            solver,
            size,
            axial_center=0.03827257364187429,
            translation_speed=MEASURED_SPEED,
        )

    BackupSystem.load_numerical_state(solver, INITIAL_H5)
    initial = sample_solver(
        solver,
        GRID_SIZES[-1],
        axial_center=0.0,
        translation_speed=MEASURED_SPEED,
    )
    gate = evaluate_gate(controls, final_samples)

    payload = {
        "stage": "5B axisymmetric quasi-steady diagnostic verification",
        "status": gate["status"],
        "theory": (
            "A steadily translating axisymmetric no-swirl ring has "
            "q=omega_phi/r=F(psi) in the translating frame."
        ),
        "raw_states": {
            "initial": str(INITIAL_H5.relative_to(ROOT)),
            "final": f"{FINAL_BACKUP.relative_to(ROOT)}.h5",
        },
        "translation_speed": {
            "measured_over_tstar_0_to_0p2": MEASURED_SPEED,
            "gaussian_reference": GAUSSIAN_SPEED,
            "relaxed_empirical_reference": RELAXED_EMPIRICAL_SPEED,
        },
        "controls": controls,
        "gate": gate,
        "initial_tstar_0": {
            fraction: serializable_metrics(metrics)
            for fraction, metrics in initial["sensitivity"].items()
        },
        "final_tstar_0p2_grid_refinement": {
            str(size): {
                fraction: serializable_metrics(metrics)
                for fraction, metrics in sample["sensitivity"].items()
            }
            for size, sample in final_samples.items()
        },
        "interpretation": (
            "The current final state is an early relaxation sample, not an accepted "
            "quasi-steady base state. Acceptance additionally requires both residuals "
            "and the independently fitted speed to plateau over several saved times."
        ),
    }
    result_path = ROOT / "scripts/experiments/stage_5b_ring_quasi_steady_results.json"
    figure_path = ROOT / "docs/figures/vpm_les/stage_5b_ring_quasi_steady.png"
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    plot_results(controls, initial, final_samples, figure_path)
    finest = payload["final_tstar_0p2_grid_refinement"][str(GRID_SIZES[-1])]["0.05"]
    print(
        json.dumps(
            {
                "status": gate["status"],
                "gate": gate,
                "tstar_0p2_primary": finest,
                "result": str(result_path.relative_to(ROOT)),
                "figure": str(figure_path.relative_to(ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
