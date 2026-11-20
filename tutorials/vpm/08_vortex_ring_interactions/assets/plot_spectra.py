"""Energy spectra, dissipation spectra and velocity-gradient invariants of DNS and LES.

``figures/spectra.png``: (a, b) the energy spectrum E(k) at the SPECTRUM_TIMES of each flow,
DNS dashed and LES solid; (c, d) the dissipation spectra of the LES, molecular
nu Omega(k) and sub-grid eps_sgs(k), with the DNS nu Omega(k) at the last of those times.
Vertical lines mark the particle-core wavenumber 2/sigma and the lattice cut-off pi/h.

``figures/gradient_invariants.png``: joint PDF of R/Q_w^(3/2) and Q/Q_w over the vortical
nodes at INVARIANT_TIMES, with the zero-discriminant line 27R^2/4 + Q^3 = 0.
Run ``python -m assets.turbulence_statistics`` first.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from openonda import plotting

import setup
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

SPECTRUM_TIMES = {"collision": (4.0, 10.0, 16.0), "leapfrog": (5.0, 15.0, 25.0)}
INVARIANT_TIMES = {"collision": 12.0, "leapfrog": 20.0}
SHADES = (0.35, 0.65, 1.0)  # line darkness by time


def spectra(case):
    return pd.read_csv(CASE_DIR / "samples" / case / "spectra.csv")


def at_time(table, time):
    """Rows of the analysed time closest to ``time``, or None beyond the run."""
    times = np.unique(table["time"])
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


def plot_spectra():
    plotting.set_thesis_style()
    fig, axes = plt.subplots(2, 2, figsize=figure_size(11.5), sharex=True)
    for column, flow in enumerate(FLOWS):
        times = SPECTRUM_TIMES[flow]
        for model, style in MODEL_STYLES.items():
            table = spectra(f"{flow}_{model}")
            for time, level in zip(times, SHADES):
                rows = at_time(table, time)
                if rows is None:
                    continue
                axes[0, column].loglog(
                    rows["wavenumber"],
                    rows["energy"],
                    color=shade(style["color"], level),
                    ls=style["linestyle"],
                    lw=1.0,
                    label=f"{style['label']}, $t={time:.0f}$",
                )
        k = np.array([4.0, 40.0])
        axes[0, column].loglog(k, 2e-3 * (k / 4.0) ** (-5.0 / 3.0), color="k", lw=0.6)
        axes[0, column].text(14.0, 1.2e-3, "$k^{-5/3}$", fontsize=8)
        last = times[-1]
        les = at_time(spectra(f"{flow}_les"), last)
        dns = at_time(spectra(f"{flow}_dns"), last)
        lower = axes[1, column]
        if les is not None:
            lower.loglog(les["wavenumber"], les["molecular_dissipation"], color=MODEL_STYLES["les"]["color"], lw=1.0, ls="-", label="LES, $\\nu\\Omega(k)$")
            lower.loglog(les["wavenumber"], les["subgrid_dissipation"].clip(lower=1e-12), color=MODEL_STYLES["les"]["color"], lw=1.0, ls=":", label="LES, $\\varepsilon_\\mathrm{sgs}(k)$")
        if dns is not None:
            lower.loglog(dns["wavenumber"], dns["molecular_dissipation"], color=MODEL_STYLES["dns"]["color"], lw=1.0, ls="--", label="DNS, $\\nu\\Omega(k)$")
        lower.text(0.04, 0.06, f"$t\\Gamma_0/R_0^2={last:.0f}$", transform=lower.transAxes)
        axes[0, column].set_title(FLOW_TITLES[flow])
        lower.set_xlabel("$kR_0$")
        for axis in axes[:, column]:
            reference_lines(axis)
    axes[0, 0].set_ylabel("$E(k)$ [m$^6$/s$^2$]")
    axes[1, 0].set_ylabel("Dissipation [m$^6$/s$^3$]")
    for index, axis in enumerate(axes.flat):
        panel_label(axis, index, right=True)
    handles, names = axes[1, 0].get_legend_handles_labels()
    shared_legend(fig, handles, names, ncol=3)
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.17, top=0.94, hspace=0.12, wspace=0.3)
    save(fig, "spectra")
    plt.close(fig)


def plot_invariants():
    plotting.set_thesis_style()
    fig, axes = plt.subplots(2, 2, figsize=figure_size(11.0), sharex=True, sharey=True)
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
                cmap=plotting.get_colormap("thesis_blue"),
            )
            r = np.linspace(edges[0], edges[-1], 400)
            axis.plot(r, -np.cbrt(27.0 / 4.0 * r**2), color="k", lw=0.6)
            axis.text(0.04, 0.06, f"{MODEL_STYLES[model]['label']}, $t={data['times'][index]:.0f}$", transform=axis.transAxes)
            axis.set_xlim(-1.5, 1.5)
            axis.set_ylim(-3.0, 3.0)
        axes[0, column].set_title(FLOW_TITLES[flow])
        axes[1, column].set_xlabel("$R/Q_w^{3/2}$")
    for axis in axes[:, 0]:
        axis.set_ylabel("$Q/Q_w$")
    for index, axis in enumerate(axes.flat):
        panel_label(axis, index, right=True)
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.11, top=0.94, hspace=0.1, wspace=0.1)
    save(fig, "gradient_invariants")
    plt.close(fig)


if __name__ == "__main__":
    plot_spectra()
    plot_invariants()
