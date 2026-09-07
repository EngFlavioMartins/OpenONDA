"""Remeasure saved fields with a common FFT box and independent target velocities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import erf

from .render_study import states
from .. import setup
from source.solvers.vpm.numerics.fourier_integrals import (
    _grid_for_particles,
    gaussian_fourier_integrals,
)


def target_velocity(targets, state, chunk=32):
    """Direct unbounded Gaussian Biot–Savart sum at independent targets."""
    position = np.asarray(state["position"], dtype=float)
    strength = np.asarray(state["vortex_strength"], dtype=float)
    sigma = np.asarray(state["core_radius"], dtype=float)
    velocity = []
    for start in range(0, len(targets), chunk):
        displacement = targets[start : start + chunk, None, :] - position[None, :, :]
        radius = np.linalg.norm(displacement, axis=2)
        q = radius / sigma
        g = erf(q) - 2 / np.sqrt(np.pi) * q * np.exp(-q * q)
        factor = np.divide(g, 4 * np.pi * radius**3, out=np.zeros_like(g), where=radius > 1e-12)
        velocity.append(
            np.sum(np.cross(strength[None, :, :], displacement) * factor[..., None], axis=1)
        )
    return np.concatenate(velocity)


def audit(runs, output, spacing=0.05, selected_steps=None, padding=1.0):
    output.mkdir(parents=True, exist_ok=True)
    records = [
        (run, state)
        for run in runs
        for state in states(run)
        if selected_steps is None or int(state["step"]) in selected_steps
    ]
    if not records:
        raise ValueError("No requested snapshots")
    bounds, maximum_core = [], 0.0
    for _, state in records:
        p = np.array(state["position"], dtype=float)
        p[:, 0] -= np.average(p[:, 0], weights=np.linalg.norm(state["vortex_strength"], axis=1))
        state["centred_position"] = p
        bounds.extend((p.min(axis=0), p.max(axis=0)))
        maximum_core = max(maximum_core, float(np.max(state["core_radius"])))
    grid = _grid_for_particles(
        np.array(bounds), spacing, padding=int(np.ceil(max(padding, 4 * maximum_core) / spacing))
    )
    padded_nodes = int(np.prod(grid.shape)) * 8
    if padded_nodes > 16_000_000:
        raise ValueError(
            f"Audit needs {padded_nodes} padded nodes; select a shorter interval or coarser audit spacing"
        )
    rows = []
    for run, state in records:
        result = gaussian_fourier_integrals(
            state["centred_position"],
            state["vortex_strength"],
            state["core_radius"],
            state.get("particle_volume", np.ones(len(state["position"]))),
            grid=grid,
            radius_expansion_order=6,
        )
        rows.append(
            dict(
                run=run,
                step=int(state["step"]),
                time=float(state["time"]),
                t_star=float(state["time"]) * np.pi,
                energy=result.total_kinetic_energy,
                enstrophy=result.total_enstrophy,
                helicity=result.total_helicity,
                energy_order_error=abs(
                    result.total_kinetic_energy - result.previous_order_total_kinetic_energy
                )
                / max(abs(result.total_kinetic_energy), 1e-30),
                enstrophy_order_error=abs(
                    result.total_enstrophy - result.previous_order_total_enstrophy
                )
                / max(abs(result.total_enstrophy), 1e-30),
            )
        )
        print(run, int(state["step"]), rows[-1]["energy"], flush=True)
    table = pd.DataFrame(rows)
    table["energy_ratio"] = table.energy / table.groupby("run").energy.transform("first")
    table["enstrophy_ratio"] = table.enstrophy / table.groupby("run").enstrophy.transform("first")
    table.to_csv(output / "field_integrals.csv", index=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for name, group in table.groupby("run", sort=False):
        axes[0].plot(group.t_star, group.energy_ratio, "o-", label=name)
        axes[1].plot(group.t_star, group.enstrophy_ratio, "o-", label=name)
    for axis, quantity in zip(axes, ("E/E(initial sample)", "Z/Z(initial sample)")):
        axis.set(xlabel="t Γ0 / R0²", ylabel=quantity)
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Saved fields remeasured on one common Fourier box")
    fig.savefig(output / "field_integrals.png", dpi=160)
    plt.close(fig)
    (output / "audit.json").write_text(
        json.dumps(
            dict(
                runs=runs,
                spacing=spacing,
                origin=grid.origin.tolist(),
                shape=grid.shape,
                padded_nodes=padded_nodes,
                energy_definition="periodic Fourier on one common box; x translation removed",
                radius_expansion_order=6,
                normalization="first selected snapshot of each run; include step zero",
                note="Independent of runtime backend switches. Finite-box and audit-spacing convergence still required.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument(
        "--output", type=Path, default=setup.TUTORIAL_DIR / "figures" / "study" / "field_audit"
    )
    parser.add_argument("--spacing", type=float, default=0.05)
    parser.add_argument("--steps", nargs="+", type=int)
    parser.add_argument("--padding", type=float, default=1.0)
    args = parser.parse_args()
    audit(args.runs, args.output, args.spacing, args.steps, args.padding)
