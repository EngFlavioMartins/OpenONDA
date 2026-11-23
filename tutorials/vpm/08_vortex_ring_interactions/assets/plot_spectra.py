"""Energy spectra, dissipation spectra and velocity-gradient invariants of DNS and both LES.

``figures/spectra.png``: (a, b) the energy spectrum E(k) at the SPECTRUM_TIMES of each flow;
(c, d) the effective viscosity of each sink at BUDGET_TIME, ``nu_x(k) = D_x(k) / (2 k^2 E(k))``
over nu: the molecular operator on its 7-point stencil, Smagorinsky with its selective eddy
viscosity, and the spectral cusp, with the shape ``1 + (15.2/0.441) exp(-3.03 k_c/k)`` of
Chollet & Lesieur (1981) scaled to the cusp run's plateau. Vertical lines mark the
particle-core wavenumber 2/sigma and the lattice cut-off pi/h.

``figures/gradient_invariants.png``: joint PDF of R/Q_w^(3/2) and Q/Q_w over the vortical
nodes at INVARIANT_TIMES, with the zero-discriminant line 27R^2/4 + Q^3 = 0.
Run ``python -m assets.turbulence_statistics`` and ``python -m assets.energy_transfer`` first.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from openonda import plotting

import setup
from openonda.vpm.turbulence.spectral_cusp import CUSP_AMPLITUDE, CUSP_RATE, PLATEAU

from assets.postprocess import (
    CASE_DIR,
    FLOW_TITLES,
    FLOWS,
    MODEL_STYLES,
    figure_size,
    panel_label,
    save,
    shared_legend,
)

# Every run reaches these times; the first run stops at t = 6.36 (collision) and 12.24 (leapfrog)
SPECTRUM_TIMES = {"collision": (2.0, 4.0, 6.0), "leapfrog": (4.0, 8.0, 12.0)}
INVARIANT_TIMES = {"collision": 6.0, "leapfrog": 12.0}
BUDGET_TIME = {"collision": 5.5, "leapfrog": 11.5}
SHADES = (0.35, 0.65, 1.0)  # line darkness by time


def spectra(case):
    return pd.read_csv(CASE_DIR / "samples" / case / "spectra.csv")


def at_time(table, time):
    """Rows of the analysed time closest to ``time``, or None beyond the run."""
    times = np.unique(table["time"])
    if not len(times):
        return None
    nearest = times[np.argmin(np.abs(times - time))]
    if abs(nearest - time) > 0.51:
        return None
    selected = table[table["time"] == nearest]
    cutoff = np.pi / setup.PARTICLE_SPACING
    return selected[(selected["wavenumber"] > 0) & (selected["wavenumber"] <= cutoff)]


def shade(colour, level):
    rgb = np.array(plt.matplotlib.colors.to_rgb(colour))
    return tuple(1.0 - level * (1.0 - rgb))


def reference_lines(axis):
    sigma = setup.CORE_RADIUS_RATIO * setup.PARTICLE_SPACING
    for k in (2.0 / sigma, np.pi / setup.PARTICLE_SPACING):
        axis.axvline(k, color=plotting.COLORS["gray"], lw=0.6, ls=":")


def budget(case):
    return pd.read_csv(CASE_DIR / "samples" / case / "energy_transfer.csv")


def plot_spectra():
    plotting.set_thesis_style()
    fig, axes = plt.subplots(2, 2, figsize=figure_size(11.5), sharex=True)
    cutoff = np.pi / setup.PARTICLE_SPACING
    for column, flow in enumerate(FLOWS):
        upper, lower = axes[0, column], axes[1, column]
        for model, style in MODEL_STYLES.items():
            case = f"{flow}_{model}"
            table = spectra(case)
            for time, level in zip(SPECTRUM_TIMES[flow], SHADES, strict=True):
                rows = at_time(table, time)
                if rows is None:
                    continue
                upper.loglog(
                    rows["wavenumber"],
                    rows["energy"],
                    color=shade(style["color"], level),
                    ls=style["linestyle"],
                    lw=1.0,
                )
            rows = at_time(budget(case), BUDGET_TIME[flow])
            if rows is None:
                continue
            k = rows["wavenumber"].to_numpy()
            scale = 2.0 * setup.KINEMATIC_VISCOSITY * k**2 * rows["energy"].to_numpy()
            if model == "dns":
                lower.loglog(
                    k,
                    rows["molecular_dissipation"] / scale,
                    color=plotting.COLORS["gray"],
                    ls=":",
                    lw=1.0,
                )
                continue
            ratio = (rows["subgrid_dissipation"] + rows["selective_dissipation"]) / scale
            lower.loglog(
                k, ratio.where(ratio > 0), color=style["color"], ls=style["linestyle"], lw=1.0
            )
            if model == "les_cusp":
                shape = 1.0 + CUSP_AMPLITUDE / PLATEAU * np.exp(-CUSP_RATE * cutoff / k)
                plateau = float(np.median(ratio[k < 0.25 * cutoff]))
                lower.loglog(k, plateau * shape, color="k", lw=0.6)
        lower.text(
            0.04, 0.06, f"$t\\Gamma_0/R_0^2={BUDGET_TIME[flow]:g}$", transform=lower.transAxes
        )
        upper.set_title(FLOW_TITLES[flow])
        lower.set_xlabel("$kR_0$")
        for axis in (upper, lower):
            reference_lines(axis)
            axis.set_xlim(1.0, cutoff)
    axes[0, 0].set_ylabel("$E(k)$ [m$^6$/s$^2$]")
    axes[1, 0].set_ylabel("$\\nu_x(k)/\\nu$")
    for axis in axes[1]:
        axis.set_ylim(1e-2, 1e2)
    for index, axis in enumerate(axes.flat):
        panel_label(axis, index, right=True)
    handles = [
        Line2D([], [], color=style["color"], ls=style["linestyle"], lw=1.0)
        for style in MODEL_STYLES.values()
    ]
    names = [style["label"] for style in MODEL_STYLES.values()]
    handles += [
        Line2D([], [], color=plotting.COLORS["gray"], ls=":", lw=1.0),
        Line2D([], [], color="k", lw=0.6),
    ]
    names += ["Molecular, 7-point stencil", "Chollet \\& Lesieur (1981)"]
    shared_legend(fig, handles, names, ncol=2)
    fig.subplots_adjust(bottom=0.27, top=0.94, hspace=0.12, wspace=0.3)
    plotting.fit_thesis_y_label_margins(fig, fig.axes)
    save(fig, "spectra")
    plt.close(fig)


def plot_invariants():
    plotting.set_thesis_style()
    fig, axes = plt.subplots(3, 2, figsize=figure_size(15.0), sharex=True, sharey=True)
    for column, flow in enumerate(FLOWS):
        for row, model in enumerate(MODEL_STYLES):
            axis = axes[row, column]
            data = np.load(CASE_DIR / "samples" / f"{flow}_{model}" / "gradient_invariants.npz")
            edges = data["edges"]
            index = int(np.argmin(np.abs(data["times"] - INVARIANT_TIMES[flow])))
            centres = 0.5 * (edges[1:] + edges[:-1])
            histogram = data["histograms"][index].T
            axis.contourf(
                centres,
                centres,
                np.where(histogram > 0, histogram, np.nan),
                levels=np.logspace(-3, 1, 9),
                norm=LogNorm(1e-3, 10),
                cmap=plt.get_cmap(plotting.get_colormap("vorticity_magnitude")).reversed(),
            )
            r = np.linspace(edges[0], edges[-1], 400)
            axis.plot(r, -np.cbrt(27.0 / 4.0 * r**2), color="k", lw=0.6)
            axis.text(
                0.04,
                0.06,
                f"{MODEL_STYLES[model]['label']}, $t={data['times'][index]:.0f}$",
                transform=axis.transAxes,
            )
            axis.set_xlim(-1.5, 1.5)
            axis.set_ylim(-3.0, 3.0)
        axes[0, column].set_title(FLOW_TITLES[flow])
        axes[-1, column].set_xlabel("$R/Q_w^{3/2}$")
    for axis in axes[:, 0]:
        axis.set_ylabel("$Q/Q_w$")
    for index, axis in enumerate(axes.flat):
        panel_label(axis, index, right=True)
    fig.subplots_adjust(bottom=0.11, top=0.94, hspace=0.1, wspace=0.1)
    plotting.fit_thesis_y_label_margins(fig, fig.axes)
    save(fig, "gradient_invariants")
    plt.close(fig)


if __name__ == "__main__":
    plot_spectra()
    plot_invariants()
