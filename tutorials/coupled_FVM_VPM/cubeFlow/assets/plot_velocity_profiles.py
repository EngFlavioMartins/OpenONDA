#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _plotutil import (
    CASE_DIR,
    COLORS,
    hybrid_pvd,
    load_forces,
    load_reference_forces,
    run_constants,
    save,
)
from _reference_util import particles_at, plot_frames, reference_velocity, sample_vtu
from _frames_util import D, EPS, U_INF, vpm_velocity

FIG = CASE_DIR / "figures"
REF_PVD = CASE_DIR / "referenceFlow" / "solution" / "referenceFlow.pvd"
FIGURE_FORMAT = "png"
FIGURE_DPI = 400
CD_MARK_EVERY = 40
PROFILE_MARK_EVERY = 10


def fig_velocity_profiles(t, hyb_vtu, ref_vtu, particles, box, forces):
    x = np.linspace(-3.0, 10.0, 521)
    cl = np.column_stack([x, np.full_like(x, EPS), np.full_like(x, EPS)])
    oa = np.column_stack([x, np.full_like(x, 0.75), np.full_like(x, EPS)])

    in_box = (x >= box["xmin"]) & (x <= box["xmax"])
    hyb_cl = sample_vtu(hyb_vtu, cl)["U"][:, 0]
    hyb_oa = sample_vtu(hyb_vtu, oa)["U"][:, 0]
    ref_cl = reference_velocity(ref_vtu, cl)[:, 0]
    ref_oa = reference_velocity(ref_vtu, oa)[:, 0]
    vpm_dom = (x >= -2.0) & (x <= 10.0)
    vpm_cl = np.full_like(x, np.nan)
    vpm_oa = np.full_like(x, np.nan)
    n_vpm = int(vpm_dom.sum())
    vpm_u = vpm_velocity(particles, np.vstack((cl[vpm_dom], oa[vpm_dom])))[:, 0]
    vpm_cl[vpm_dom] = vpm_u[:n_vpm]
    vpm_oa[vpm_dom] = vpm_u[n_vpm:]

    fig_height = 6.0 / 2.54
    fig_length = 12.5 / 2.54
    fig = plt.figure(figsize=(fig_length, 2 * fig_height), dpi=400)
    gs = GridSpec(2, 2, figure=fig)
    ax2 = fig.add_subplot(gs[0, 0])
    ax3 = fig.add_subplot(gs[0, 1])
    ax1 = fig.add_subplot(gs[1, :])

    ax1.plot(
        forces["hyb"]["time"] * U_INF / D,
        forces["hyb"]["Cd"],
        color=COLORS["hybrid"],
        marker="o",
        markevery=CD_MARK_EVERY,
        ms=2,
        label="FVM",
    )
    ax1.plot(
        forces["ref"]["time"] * U_INF / D,
        forces["ref"]["Cd"],
        color=COLORS["reference"],
        ls="-.",
        marker="D",
        markevery=CD_MARK_EVERY,
        ms=2,
        label="Reference FVM",
    )
    ax1.set(
        xlabel=r"$t U_\infty / D$",
        ylabel=r"$C_D$",
        xlim=(0, 20),
        ylim=(0.5, 2),
        title=r"Drag Coefficient, $C_D$",
    )
    ax1.legend(loc="upper right")

    ax2.axvspan(-0.5, 0.5, color=COLORS["background_strong"], zorder=1)
    for ax, hyb, ref, vpm, ttl, ylim in (
        (ax2, hyb_cl, ref_cl, vpm_cl, f"Centerline ($t={t:.2f}$)", (-1.2, 1.2)),
        (ax3, hyb_oa, ref_oa, vpm_oa, f"Off-axis $y=0.75D$ ($t={t:.2f}$)", (-0.5, 1.5)),
    ):
        ax.axvspan(box["xmin"], box["xmax"], color=COLORS["background_light"], zorder=0)
        ax.plot(
            x,
            ref / U_INF,
            color=COLORS["reference"],
            ls="-.",
            label="Reference FVM" if np.isfinite(ref).any() else "_nolegend_",
            zorder=2,
        )
        hx = np.where(in_box, hyb, np.nan)
        ax.plot(x, hx / U_INF, color=COLORS["hybrid"], ls="-", label="FVM", zorder=2)
        ax.plot(
            x,
            vpm / U_INF,
            color=COLORS["vpm"],
            ls="-",
            marker="o",
            ms=1.5,
            markevery=PROFILE_MARK_EVERY,
            label="VPM",
            zorder=2,
        )
        ax.set(xlabel=r"$x/D$", ylabel="", xlim=(-3, 10), ylim=ylim, title=ttl)
    ax2.set_ylabel(r"$u_x/U_\infty$")
    ax3.legend(loc="lower right")

    save(fig, f"velocity_profiles_t{t:.2f}", FIGURE_FORMAT, FIGURE_DPI)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    forces = {"hyb": load_forces(), "ref": load_reference_forces()}
    box = run_constants()["box"]

    for time, hybrid_vtu, reference_vtu in plot_frames(hybrid_pvd(), REF_PVD):
        fig_velocity_profiles(
            time,
            hybrid_vtu,
            reference_vtu,
            particles_at(time),
            box,
            forces,
        )
        print(f"  velocity_profiles t={time:.2f} done")


if __name__ == "__main__":
    main()
