#!/usr/bin/env python3
"""Small, standalone assessment of coupled RK versus split VPM evolution.

This file is intentionally outside ``source/`` and ``openonda/``. It uses two
linear cross-coupled systems for exact error and stability measurements, plus a
tiny synthetic fixed-core smooth particle system. The historical method is the
repository's pre-11997d99 advection-then-stretching path: SSPRK3 for the
advection subproblem followed by SSPRK3 for the stretching subproblem. Work is
instrumented from the actual RHS calls made by each run; it is not a production
backend timing model.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ASSESSMENT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = ASSESSMENT_DIR / "figures"
os.environ.setdefault("MPLCONFIGDIR", str(ASSESSMENT_DIR / ".mplconfig"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker


METHODS = {
    "ssp3_coupled": {"label": "coupled SSPRK3"},
    "historical_lie": {"label": "historical Lie (SSPRK3 A then B)"},
    "strang_rk4": {"label": "Strang (RK4 subflows)"},
}
DT_VALUES = (0.4, 0.2, 0.1, 0.05, 0.025)


class _WorkCounter:
    """Count actual RHS/evaluator calls for one integration run."""

    def __init__(self) -> None:
        self.rhs_calls = 0
        self.full_rhs_calls = 0
        self.subproblem_rhs_calls = 0
        self.a_rhs_calls = 0
        self.b_rhs_calls = 0
        self.evaluator_calls = 0
        self.full_velocity_gradient_evaluator_calls = 0
        self.subproblem_evaluator_calls = 0

    def wrap(self, rhs, role: str):
        if role not in {"full", "A", "B"}:
            raise ValueError(f"unknown RHS role {role}")

        def counted(state):
            self.rhs_calls += 1
            if role == "full":
                self.full_rhs_calls += 1
            else:
                self.subproblem_rhs_calls += 1
                if role == "A":
                    self.a_rhs_calls += 1
                else:
                    self.b_rhs_calls += 1
            return rhs(state)

        return counted

    def record_evaluator(self, role: str) -> None:
        """Record the synthetic evaluator call made by one particle RHS."""
        self.evaluator_calls += 1
        if role == "full":
            self.full_velocity_gradient_evaluator_calls += 1
        else:
            self.subproblem_evaluator_calls += 1

    def snapshot(self) -> dict:
        return {
            "rhs_calls": self.rhs_calls,
            "full_rhs_calls": self.full_rhs_calls,
            "subproblem_rhs_calls": self.subproblem_rhs_calls,
            "a_rhs_calls": self.a_rhs_calls,
            "b_rhs_calls": self.b_rhs_calls,
            "evaluator_calls": self.evaluator_calls,
            "full_velocity_gradient_evaluator_calls": self.full_velocity_gradient_evaluator_calls,
            "subproblem_evaluator_calls": self.subproblem_evaluator_calls,
        }


def _rk3_step(state: np.ndarray, h: float, rhs) -> np.ndarray:
    """The repository's SSPRK3 tableau, applied to an arbitrary state."""
    k1 = rhs(state)
    k2 = rhs(state + h * k1)
    k3 = rhs(state + 0.25 * h * (k1 + k2))
    return state + h * (k1 + k2 + 4.0 * k3) / 6.0


def _rk4_step(state: np.ndarray, h: float, rhs) -> np.ndarray:
    k1 = rhs(state)
    k2 = rhs(state + 0.5 * h * k1)
    k3 = rhs(state + 0.5 * h * k2)
    k4 = rhs(state + h * k3)
    return state + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def _integrate_split(
    initial: np.ndarray,
    horizon: float,
    h: float,
    method: str,
    rhs_a,
    rhs_b,
    rhs_full,
    counter: _WorkCounter | None = None,
) -> np.ndarray:
    steps = int(round(horizon / h))
    if not math.isclose(steps * h, horizon, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"horizon={horizon} is not an integer number of steps h={h}")
    if counter is not None:
        rhs_a = counter.wrap(rhs_a, "A")
        rhs_b = counter.wrap(rhs_b, "B")
        rhs_full = counter.wrap(rhs_full, "full")
    state = np.asarray(initial, dtype=np.float64).copy()
    for _ in range(steps):
        if method == "ssp3_coupled":
            state = _rk3_step(state, h, rhs_full)
        elif method == "historical_lie":
            state = _rk3_step(state, h, rhs_a)
            state = _rk3_step(state, h, rhs_b)
        elif method == "strang_rk4":
            state = _rk4_step(state, 0.5 * h, rhs_a)
            state = _rk4_step(state, h, rhs_b)
            state = _rk4_step(state, 0.5 * h, rhs_a)
        else:
            raise ValueError(f"unknown method {method}")
    return state


def _linear_system(sign: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return A, B, M and an exact-solution initial state.

    A advances position from strength, B advances strength from position. With
    sign=-1 the combined system is an oscillator; sign=+1 has an exact growing
    eigenmode. Both are genuinely two-way coupled.
    """
    a = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    b = np.array([[0.0, 0.0], [sign, 0.0]], dtype=np.float64)
    return a, b, a + b, np.array([1.0, 0.0], dtype=np.float64)


def _exact_linear(state0: np.ndarray, time: float, sign: float) -> np.ndarray:
    x0, alpha0 = state0
    if sign < 0.0:
        return np.array(
            [math.cos(time) * x0 + math.sin(time) * alpha0,
             -math.sin(time) * x0 + math.cos(time) * alpha0],
            dtype=np.float64,
        )
    return np.array(
        [math.cosh(time) * x0 + math.sinh(time) * alpha0,
         math.sinh(time) * x0 + math.cosh(time) * alpha0],
        dtype=np.float64,
    )


def _linear_results() -> tuple[list[dict], dict]:
    rows: list[dict] = []
    # A common exact horizon for all halved decimal steps; the final state is
    # evaluated against the analytic oscillator, not against a periodic sample.
    oscillator_horizon = 8.0
    growing_horizon = 4.0
    for system_name, sign, horizon in (
        ("oscillator", -1.0, oscillator_horizon),
        ("growing", 1.0, growing_horizon),
    ):
        a, b, matrix, initial = _linear_system(sign)
        exact = _exact_linear(initial, horizon, sign)
        for h in DT_VALUES:
            for method in METHODS:
                counter = _WorkCounter()
                final = _integrate_split(
                    initial,
                    horizon,
                    h,
                    method,
                    lambda state, matrix=a: matrix @ state,
                    lambda state, matrix=b: matrix @ state,
                    lambda state, matrix=matrix: matrix @ state,
                    counter=counter,
                )
                error = float(np.linalg.norm(final - exact) / np.linalg.norm(exact))
                row = {
                    "experiment": "linear",
                    "system": system_name,
                    "method": method,
                    "dt": h,
                    "horizon": horizon,
                    "steps": int(round(horizon / h)),
                    "relative_error": error,
                    "final_x": float(final[0]),
                    "final_alpha": float(final[1]),
                }
                row.update(counter.snapshot())
                if system_name == "oscillator":
                    amplitude = float(np.linalg.norm(final) / np.linalg.norm(initial) - 1.0)
                    phase = math.atan2(-float(final[1]), float(final[0]))
                    phase_error = (phase - horizon + math.pi) % (2.0 * math.pi) - math.pi
                    row.update(
                        amplitude_error=amplitude,
                        phase_error=phase_error,
                    )
                else:
                    exact_amplification = float(np.linalg.norm(exact) / np.linalg.norm(initial))
                    numerical_amplification = float(np.linalg.norm(final) / np.linalg.norm(initial))
                    row.update(
                        exact_amplification=exact_amplification,
                        numerical_amplification=numerical_amplification,
                        amplification_ratio=numerical_amplification / exact_amplification,
                    )
                rows.append(row)
    return rows, {
        "oscillator_horizon": oscillator_horizon,
        "growing_horizon": growing_horizon,
        "initial_state": [1.0, 0.0],
    }


def _linear_amplification(method: str, h: float, sign: float) -> np.ndarray:
    a, b, matrix, _ = _linear_system(sign)
    identity = np.eye(2)
    return np.column_stack(
        [
            _integrate_split(
                identity[:, column], h, h, method, lambda state: a @ state,
                lambda state: b @ state, lambda state: matrix @ state
            )
            for column in range(2)
        ]
    )


def _stability_results() -> dict:
    q_values = (0.5, math.sqrt(3.0) * 0.95, math.sqrt(3.0) * 1.05, 1.9, 2.1)
    rows = []
    for q in q_values:
        for method in METHODS:
            amplification = _linear_amplification(method, q, -1.0)
            eigenvalues = np.linalg.eigvals(amplification)
            rows.append(
                {
                    "q": q,
                    "method": method,
                    "spectral_radius": float(np.max(np.abs(eigenvalues))),
                    "matrix": amplification.tolist(),
                }
            )
    return {
        "oscillator_q": rows,
        "predicted_boundaries": {
            "coupled_ssprk3_imaginary_axis": math.sqrt(3.0),
            "historical_lie_exact_subflows": 2.0,
            "strang_exact_subflows": 2.0,
        },
    }


def _particle_rhs_factory(
    sign: float = 1.0,
    sigma: float = 0.8,
):
    """Make one direct, smooth Gaussian-like velocity/Jacobian evaluator.

    The kernel is u = sign * (Gamma × r) exp(-|r|²/(2 sigma²)). It is not
    claimed to be a conserved vortex-particle Hamiltonian; it is a compact
    two-way coupling probe with the same evaluator used by every method.
    """
    sigma2 = sigma * sigma

    def velocity_and_gradient(position: np.ndarray, strength: np.ndarray):
        count = len(position)
        velocity = np.zeros((count, 3), dtype=np.float64)
        gradient = np.zeros((count, 3, 3), dtype=np.float64)
        for i in range(count):
            for j in range(count):
                if i == j:
                    continue
                r = position[i] - position[j]
                gamma = strength[j]
                q = math.exp(-float(r @ r) / (2.0 * sigma2))
                cross_matrix = np.array(
                    [[0.0, -gamma[2], gamma[1]],
                     [gamma[2], 0.0, -gamma[0]],
                     [-gamma[1], gamma[0], 0.0]],
                    dtype=np.float64,
                )
                raw = np.cross(gamma, r)
                contribution = sign * q * raw
                velocity[i] += contribution
                gradient[i] += sign * q * (cross_matrix - np.outer(raw, r) / sigma2)
        return velocity, gradient

    return velocity_and_gradient


def _particle_rhses(counter: _WorkCounter | None = None):
    evaluator = _particle_rhs_factory()

    def evaluate(position, strength, role: str):
        if counter is not None:
            counter.record_evaluator(role)
        return evaluator(position, strength)

    initial = np.array(
        [[0.0, 0.0, 0.0, 0.20, 0.40, 1.00],
         [1.0, 0.30, 0.40, -0.30, 0.60, 0.70]],
        dtype=np.float64,
    )

    def split_state(state):
        values = np.asarray(state, dtype=np.float64).reshape(2, 6)
        return values[:, :3], values[:, 3:]

    def rhs_full(state):
        position, strength = split_state(state)
        velocity, gradient = evaluate(position, strength, "full")
        rate = np.empty_like(position)
        for i in range(len(position)):
            rate[i] = gradient[i].T @ strength[i]
        return np.column_stack((velocity, rate)).reshape(-1)

    def rhs_a(state):
        position, strength = split_state(state)
        velocity, _ = evaluate(position, strength, "A")
        return np.column_stack((velocity, np.zeros_like(strength))).reshape(-1)

    def rhs_b(state):
        position, strength = split_state(state)
        _, gradient = evaluate(position, strength, "B")
        rate = np.empty_like(strength)
        for i in range(len(position)):
            rate[i] = gradient[i].T @ strength[i]
        return np.column_stack((np.zeros_like(position), rate)).reshape(-1)

    return initial.reshape(-1), rhs_a, rhs_b, rhs_full, evaluator


def _particle_results() -> tuple[list[dict], dict]:
    initial, rhs_a, rhs_b, rhs_full, evaluator = _particle_rhses()
    horizon = 0.5
    reference_step = 1.0e-4
    reference = _integrate_split(
        initial, horizon, reference_step, "strang_rk4", rhs_a, rhs_b, rhs_full
    )
    finer_reference = _integrate_split(
        initial, horizon, reference_step / 2.0, "strang_rk4", rhs_a, rhs_b, rhs_full
    )
    reference_difference = float(
        np.linalg.norm(reference - finer_reference) / max(np.linalg.norm(finer_reference), 1.0e-30)
    )
    position0, strength0 = initial.reshape(2, 6)[:, :3], initial.reshape(2, 6)[:, 3:]
    velocity0, gradient0 = evaluator(position0, strength0)
    rows = []
    for h in (0.05, 0.025, 0.0125, 0.00625):
        steps = int(round(horizon / h))
        for method in METHODS:
            counter = _WorkCounter()
            _, rhs_a, rhs_b, rhs_full, _ = _particle_rhses(counter)
            final = _integrate_split(
                initial, horizon, h, method, rhs_a, rhs_b, rhs_full, counter=counter
            )
            relative_error = float(np.linalg.norm(final - reference) / np.linalg.norm(reference))
            final_values = final.reshape(2, 6)
            strength_sum_drift = float(
                np.linalg.norm(final_values[:, 3:].sum(axis=0) - strength0.sum(axis=0))
            )
            row = {
                "experiment": "particle",
                "system": "two_particle_fixed_core",
                "method": method,
                "dt": h,
                "horizon": horizon,
                "steps": steps,
                "relative_error": relative_error,
                "strength_sum_drift": strength_sum_drift,
            }
            row.update(counter.snapshot())
            rows.append(row)
    return rows, {
        "horizon": horizon,
        "reference_method": "Strang RK4",
        "reference_step": reference_step,
        "reference_finer_step": reference_step / 2.0,
        "reference_relative_difference": reference_difference,
        "kernel_scope": "synthetic smooth Gaussian-like pair interaction",
        "production_vpm_kernel_used": False,
        "evaluator_counts_are_not_production_timings": True,
        "initial_velocity_norm": float(np.linalg.norm(velocity0)),
        "initial_stretching_rate_norm": float(
            np.linalg.norm(np.einsum("nij,nj->ni", np.swapaxes(gradient0, 1, 2), strength0))
        ),
        "strength_sum_is_not_claimed_conserved": True,
    }


def _write_raw(rows: list[dict], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summarize_work(rows: list[dict]) -> dict:
    """Summarize per-step counts from recorded rows, without a cost constant."""
    summary = {}
    for experiment in sorted({row["experiment"] for row in rows}):
        summary[experiment] = {}
        for method in METHODS:
            selected = [
                row
                for row in rows
                if row["experiment"] == experiment and row["method"] == method
            ]
            if not selected:
                continue
            summary[experiment][method] = {}
            for field in (
                "rhs_calls",
                "full_rhs_calls",
                "subproblem_rhs_calls",
                "a_rhs_calls",
                "b_rhs_calls",
                "evaluator_calls",
                "full_velocity_gradient_evaluator_calls",
                "subproblem_evaluator_calls",
            ):
                summary[experiment][method][f"{field}_per_step"] = sorted(
                    {int(row[field]) // int(row["steps"]) for row in selected}
                )
    return summary


def _plot_convergence(rows: list[dict]) -> None:
    linear = [row for row in rows if row["experiment"] == "linear"]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8), constrained_layout=True)
    for system, column, title in (
        ("oscillator", 0, "oscillator: terminal relative error"),
        ("growing", 1, "growing system: terminal relative error"),
    ):
        axis_dt = axes[0, column]
        axis_work = axes[1, column]
        for method, info in METHODS.items():
            selected = sorted(
                (row for row in linear if row["system"] == system and row["method"] == method),
                key=lambda row: row["dt"],
            )
            axis_dt.loglog(
                [row["dt"] for row in selected],
                [row["relative_error"] for row in selected],
                "o-",
                label=info["label"],
            )
            axis_work.loglog(
                [row["rhs_calls"] for row in selected],
                [row["relative_error"] for row in selected],
                "o-",
                label=info["label"],
            )
        axis_dt.set(xlabel="dt", ylabel="relative error", title=title)
        axis_work.set(xlabel="counted RHS evaluations", ylabel="relative error", title=f"{system}: error versus work")
        # The measured dt values are the five plotted points. Use them as the
        # major ticks and suppress log-minor labels so the top panels remain
        # readable without changing any numerical results.
        dt_ticks = [0.025, 0.05, 0.1, 0.2, 0.4]
        axis_dt.set_xticks(dt_ticks, labels=["0.025", "0.05", "0.1", "0.2", "0.4"])
        axis_dt.xaxis.set_minor_formatter(mticker.NullFormatter())
        axis_dt.grid(True, which="both", alpha=0.3)
        axis_work.grid(True, which="both", alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(FIGURES_DIR / "convergence.png", dpi=160)
    plt.close(fig)


def _plot_stability(stability: dict) -> None:
    rows = stability["oscillator_q"]
    fig, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    for method, info in METHODS.items():
        selected = [row for row in rows if row["method"] == method]
        axis.plot(
            [row["q"] for row in selected],
            [row["spectral_radius"] for row in selected],
            "o-",
            label=info["label"],
        )
    axis.axhline(1.0, color="black", linewidth=0.8)
    axis.axvline(math.sqrt(3.0), color="tab:red", linestyle="--", linewidth=0.9, label="SSPRK3 q=√3")
    axis.axvline(2.0, color="tab:green", linestyle=":", linewidth=1.0, label="split q=2")
    axis.set(
        xlabel="q = omega dt",
        ylabel="spectral radius",
        title="oscillator one-step amplification",
    )
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    fig.savefig(FIGURES_DIR / "stability.png", dpi=160)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    linear_rows, linear_meta = _linear_results()
    particle_rows, particle_meta = _particle_results()
    stability = _stability_results()
    rows = linear_rows + particle_rows
    _write_raw(rows, ASSESSMENT_DIR / "results.csv")
    payload = {
        "precision": "float64",
        "methods": METHODS,
        "work_accounting_from_recorded_rows": _summarize_work(rows),
        "linear": linear_meta,
        "particle": particle_meta,
        "stability": stability,
        "rows": rows,
    }
    (ASSESSMENT_DIR / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    _plot_convergence(rows)
    _plot_stability(stability)
    print(f"wrote {ASSESSMENT_DIR / 'results.csv'}")
    print(f"wrote {ASSESSMENT_DIR / 'results.json'}")
    print(f"wrote {FIGURES_DIR / 'convergence.png'}")
    print(f"wrote {FIGURES_DIR / 'stability.png'}")
    print(f"particle reference disagreement: {particle_meta['reference_relative_difference']:.3e}")


if __name__ == "__main__":
    main()
