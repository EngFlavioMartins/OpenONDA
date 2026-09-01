#!/usr/bin/env python3
"""Compare DIRECT, TRANSPOSED, and MIXED against one exact Euler benchmark."""

from __future__ import annotations

import csv
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.integrate import solve_ivp
import setup

from assets.core import AnalyticEvaluator, State, errors, integrate


class ABCFlow:
    """Steady A=B=C=1 Beltrami flow: curl(u)=u, an exact Euler solution."""

    name = "abc_beltrami_euler"

    def velocity(self, x, t):
        del t
        a, b, c = np.asarray(x).T
        return np.column_stack(
            (np.sin(c) + np.cos(b), np.sin(a) + np.cos(c), np.sin(b) + np.cos(a))
        )

    def gradient(self, x, t):
        del t
        a, b, c = np.asarray(x).T
        j = np.zeros((len(x), 3, 3))
        j[:, 0, 1] = -np.sin(b)
        j[:, 0, 2] = np.cos(c)
        j[:, 1, 0] = np.cos(a)
        j[:, 1, 2] = -np.sin(c)
        j[:, 2, 0] = -np.sin(a)
        j[:, 2, 1] = np.cos(b)
        return j


def write(rows):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with (setup.RESULTS / "formulation_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    setup.mkdirs()
    flow = ABCFlow()
    axis = np.linspace(0.2, 2 * np.pi - 0.2, 4)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    x0 = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    volume = (2 * np.pi / 4) ** 3
    g0 = volume * flow.velocity(x0, 0)
    packed = x0.ravel()
    n = len(x0)
    sol = solve_ivp(
        lambda t, v: flow.velocity(v.reshape(n, 3), t).ravel(),
        (0, 1),
        packed,
        method="DOP853",
        rtol=2e-13,
        atol=2e-15,
    )
    xref = sol.y[:, -1].reshape(n, 3)
    ref = State(xref, volume * flow.velocity(xref, 1))
    initial = State(x0, g0)
    rows = []
    j0 = flow.gradient(x0, 0)
    direct = np.einsum("nij,nj->ni", j0, g0)
    transposed = np.einsum("nji,nj->ni", j0, g0)
    mixed = 0.5 * (direct + transposed)
    identity_error = max(
        np.linalg.norm(direct - transposed), np.linalg.norm(direct - mixed)
    ) / np.linalg.norm(direct)
    for mode in setup.MODES:
        for steps in (5, 10, 20, 40, 80):
            evaluator = AnalyticEvaluator(flow)
            start = time.perf_counter()
            out = integrate("coupled_rk3", evaluator, initial, 1.0, steps, mode)
            metric = errors(out, ref)
            physical_defect = np.linalg.norm(
                out.gamma - volume * flow.velocity(out.x, 1)
            ) / np.linalg.norm(ref.gamma)
            rows.append(
                dict(
                    benchmark=flow.name,
                    reference="exact steady Euler Beltrami relation curl(u)=u",
                    mode=mode,
                    integrator="coupled_rk3",
                    steps=steps,
                    dt=1 / steps,
                    rate_identity_error_at_initial_state=identity_error,
                    physical_manifold_defect=physical_defect,
                    wall_time_s=time.perf_counter() - start,
                    **metric,
                    **evaluator.counts.__dict__,
                )
            )
    write(rows)
    print(f"wrote {len(rows)} common-reference formulation records")


if __name__ == "__main__":
    main()
