"""Locate peaks of azimuthally averaged vorticity, without using material labels."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter
from scipy.optimize import minimize
from source.solvers.vpm.diagnostics.axisymmetric_field import (
    azimuthal_vorticity,
    azimuthal_circulation,
)

if not __package__:
    from openonda.tutorial_runner import case_package
    from pathlib import Path as _CasePath

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

from .render_study import states
from .. import setup


def core_peaks(state):
    position = state["position"]
    weight = np.linalg.norm(state["vortex_strength"], axis=1)
    centre = np.average(position[:, 0], weights=weight)
    x = np.linspace(centre - 1.25, centre + 1.25, 21)
    r = np.linspace(0.15, 1.65, 16)
    xx, rr = np.meshgrid(x, r, indexing="ij")

    def field(points):
        return azimuthal_vorticity(position, state["vortex_strength"], state["core_radius"], points)

    grid = field(np.column_stack((xx.ravel(), rr.ravel()))).reshape(xx.shape)
    locations = np.argwhere((grid == maximum_filter(grid, size=3)) & (grid > grid.max() * 0.1))
    peaks = []
    for ix, ir in locations:
        seed = np.array([x[ix], r[ir]])
        result = minimize(
            lambda v: -field([v])[0],
            seed,
            method="Nelder-Mead",
            bounds=[(x[0], x[-1]), (r[0], r[-1])],
            options={"xatol": 1e-6, "fatol": 1e-7, "maxiter": 150},
        )
        point = result.x
        if all(np.linalg.norm(point - p[:2]) > 0.02 for p in peaks):
            peaks.append(np.r_[point, -result.fun])
    return sorted(peaks, key=lambda p: -p[2]), centre


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--steps", type=int, nargs="*")
    args = parser.parse_args()
    output = setup.TUTORIAL_DIR / "figures" / "study" / "core_diagnosis"
    output.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        rows = []
        for state in states(run):
            step = int(state["step"])
            if args.steps and step not in args.steps:
                continue
            peaks, centre = core_peaks(state)
            circulation = azimuthal_circulation(
                state["position"], state["vortex_strength"], state["core_radius"]
            )
            bridge_ratio = np.nan
            if len(peaks) >= 2:
                line = np.linspace(peaks[0][:2], peaks[1][:2], 101)
                bridge = azimuthal_vorticity(
                    state["position"], state["vortex_strength"], state["core_radius"], line
                )
                # A high straight-line saddle proves that the two local maxima
                # share a strong bridge; two peaks need not be two separate rings.
                bridge_ratio = float(bridge.min() / min(peaks[0][2], peaks[1][2]))
            print(
                run,
                step,
                [(round(p[0], 4), round(p[1], 4), round(p[2], 3)) for p in peaks],
                flush=True,
            )
            for rank, peak in enumerate(peaks):
                delta = 0.003
                offsets = (
                    np.array(
                        [
                            [0, 0],
                            [1, 0],
                            [-1, 0],
                            [0, 1],
                            [0, -1],
                            [1, 1],
                            [1, -1],
                            [-1, 1],
                            [-1, -1],
                        ]
                    )
                    * delta
                )
                values = azimuthal_vorticity(
                    state["position"],
                    state["vortex_strength"],
                    state["core_radius"],
                    peak[:2] + offsets,
                )
                widths = np.full(2, np.nan)
                if np.all(values > 0):
                    log = np.log(values)
                    cross = (log[5] - log[6] - log[7] + log[8]) / (4 * delta**2)
                    hessian = np.array(
                        [
                            [(log[1] + log[2] - 2 * log[0]) / delta**2, cross],
                            [cross, (log[3] + log[4] - 2 * log[0]) / delta**2],
                        ]
                    )
                    eigenvalues = np.linalg.eigvalsh(hessian)
                    if np.all(eigenvalues < 0):
                        widths = np.sqrt(-2 / eigenvalues)
                rows.append(
                    dict(
                        run=run,
                        step=step,
                        time=float(state["time"]),
                        peak_rank=rank,
                        x=peak[0],
                        radius=peak[1],
                        vorticity=peak[2],
                        n_peaks=len(peaks),
                        total_meridional_circulation=circulation,
                        strongest_peak_pair_bridge_ratio=bridge_ratio,
                        strength_centroid_x=centre,
                        core_minor_width=widths[0],
                        core_major_width=widths[1],
                        core_aspect_ratio=widths[1] / widths[0],
                    )
                )
        pd.DataFrame(rows).to_csv(output / f"{run}_peaks.csv", index=False)


if __name__ == "__main__":
    main()
