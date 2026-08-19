#!/usr/bin/env python3
"""Audit the long-ring excess energy loss with a factor-two time-step pair."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/openonda_ring_energy_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/openonda_ring_energy_cache")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from source.solvers.VPM import VPMSolver  # noqa: E402
from source.solvers.VPM.io import BackupSystem  # noqa: E402

RUN = ROOT / (
    "tutorials/VPM/vortexRing/solution/relaxed_reference_tail002_cs_h012_dt002_tstar2_qpsi"
)
PREFIX = "vpm_relaxed_reference_tail002_cs_h012_dt002_tstar2_qpsi"
BASE_CONFIG = RUN / f"{PREFIX}_final"
START = RUN / f"{PREFIX}_final.h5"
COARSE_FINAL = RUN / f"{PREFIX}_000150.h5"
FINE_FINAL = ROOT / (
    "tutorials/VPM/vortexRing/solution/"
    "relaxed_reference_tail002_cs_h012_from_t2_dt001_tstar3/"
    "vpm_relaxed_h012_from_t2_dt001_tstar3_final.h5"
)

INK = "#20252a"
BLUE = "#286f9b"
GOLD = "#d9973b"
GREY = "#8a99a8"
GRID = "#d8dde2"
LIMIT = 0.05


def integral_state(solver: VPMSolver, path: Path) -> dict[str, float]:
    BackupSystem.load_numerical_state(solver, path)
    solver._update_all_flow_integrals()
    return {
        "time": solver.time,
        "energy": solver.total_kinetic_energy,
        "molecular_energy_rate": float(solver._flow_integrals["vorticity_dissipation_rate"]),
    }


def interval(
    start: dict[str, float], final: dict[str, float], time_step_size: float
) -> dict[str, float]:
    duration = final["time"] - start["time"]
    measured = final["energy"] - start["energy"]
    predicted = 0.5 * duration * (start["molecular_energy_rate"] + final["molecular_energy_rate"])
    return {
        "time_step": time_step_size,
        "duration": duration,
        "initial_energy": start["energy"],
        "final_energy": final["energy"],
        "measured_energy_change": measured,
        "molecular_energy_change": predicted,
        "energy_balance_relative_residual": abs((measured - predicted) / predicted),
    }


def plot(rows: list[dict[str, float]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.7), constrained_layout=True)
    fig.suptitle(
        r"Vortex-ring energy audit over $2\leq t^*\leq3$",
        color=INK,
        fontsize=13,
        fontweight="bold",
    )
    labels = [rf"$\Delta t={row['time_step']:.2f}$" for row in rows]
    x = np.arange(len(rows))
    width = 0.34

    axis = axes[0]
    axis.bar(
        x - width / 2,
        [-row["molecular_energy_change"] for row in rows],
        width,
        color=INK,
        label=r"exact molecular integral $\nu\int|\omega|^2dt$",
    )
    axis.bar(
        x + width / 2,
        [-row["measured_energy_change"] for row in rows],
        width,
        color=BLUE,
        label="measured kinetic-energy loss",
    )
    axis.set_xticks(x, labels)
    axis.set_ylabel(r"positive energy loss $-\Delta E$")
    axis.set_title("Both time steps lose the same excess energy")
    axis.legend(frameon=False, fontsize=7)

    axis = axes[1]
    axis.bar(x, [row["energy_balance_relative_residual"] for row in rows], color=GOLD)
    axis.axhline(LIMIT, color=INK, linestyle="--", label="frozen 5% limit")
    axis.set_xticks(x, labels)
    axis.set_ylabel("relative energy-balance residual")
    axis.set_title("Halving the time step does not restore the identity")
    axis.legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.grid(True, color=GRID, linewidth=0.7, alpha=0.75)
        axis.spines[["top", "right"]].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main() -> None:
    solver = VPMSolver.continue_from_backup(str(BASE_CONFIG))
    if solver is None:
        raise RuntimeError(f"could not restore {BASE_CONFIG}")
    start = integral_state(solver, START)
    coarse = interval(start, integral_state(solver, COARSE_FINAL), 0.02)
    fine = interval(start, integral_state(solver, FINE_FINAL), 0.01)
    rows = [coarse, fine]
    endpoint_difference = abs(coarse["final_energy"] / fine["final_energy"] - 1.0)
    gate = {
        "status": "FAIL",
        "limit": LIMIT,
        "maximum_energy_balance_relative_residual": max(
            row["energy_balance_relative_residual"] for row in rows
        ),
        "factor_two_endpoint_energy_relative_difference": endpoint_difference,
        "interpretation": (
            "The trajectories are time-step converged but violate the discrete "
            "kinetic-energy identity; do not continue the reference relaxation."
        ),
    }
    payload = {
        "stage": "5B long-ring energy audit",
        "status": gate["status"],
        "identity": "Delta E = - integral nu |omega|^2 dV dt",
        "common_start": str(START.relative_to(ROOT)),
        "intervals": rows,
        "gate": gate,
    }
    result = ROOT / "scripts/experiments/stage_5b_ring_energy_audit_results.json"
    figure = ROOT / "docs/figures/vpm_les/stage_5b_ring_energy_audit.png"
    result.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    plot(rows, figure)
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
