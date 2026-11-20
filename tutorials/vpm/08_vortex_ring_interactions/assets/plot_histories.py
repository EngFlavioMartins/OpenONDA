"""Energy, enstrophy, dissipation and disturbance histories of the six runs.

Columns: head-on collision, leapfrogging. Rows: (a, b) kinetic energy; (c, d) enstrophy;
(e, f) the energy loss rate -dE/dt of each run with the sub-grid part eps_sgs of each LES;
(g, h) the largest azimuthal amplitude A_m of the two rings. A cross marks where a run
stopped at a state limit.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from openonda import plotting

from assets.postprocess import (
    FLOW_TITLES,
    FLOWS,
    MODEL_STYLES,
    energy_budget,
    figure_size,
    panel_label,
    read_table,
    run_status,
    save,
    shared_legend,
)


def stopped(case):
    return run_status(case)["status"] != "completed"


def plot():
    plotting.set_thesis_style()
    fig, axes = plt.subplots(4, 2, figsize=figure_size(15.5), sharex="col")
    for column, flow in enumerate(FLOWS):
        for model, style in MODEL_STYLES.items():
            case = f"{flow}_{model}"
            budget = energy_budget(case)
            time = budget["time"].to_numpy()
            modes = read_table(case, "azimuthal_modes")
            amplitude = modes.filter(like="mode_").max(axis=1).groupby(modes["time"]).max()
            series = (
                budget["total_kinetic_energy"] / budget["total_kinetic_energy"].iloc[0],
                budget["total_enstrophy"] / budget["total_enstrophy"].iloc[0],
                budget["energy_loss_rate"],
            )
            line = dict(color=style["color"], ls=style["linestyle"], lw=1.0)
            for row, values in enumerate(series):
                axes[row, column].plot(time, values, label=style["label"], **line)
            axes[3, column].semilogy(amplitude.index, amplitude.to_numpy(), **line)
            if model != "dns":
                axes[2, column].plot(
                    time,
                    budget["subgrid_dissipation"],
                    color=style["color"],
                    ls=":",
                    lw=1.0,
                    label=f"{style['label']}, $\\varepsilon_\\mathrm{{sgs}}$",
                )
            if stopped(case):
                for row, values in enumerate(series):
                    axes[row, column].plot(
                        time[-1], np.asarray(values)[-1], "x", color=style["color"], ms=5, mew=1
                    )
                axes[3, column].plot(
                    amplitude.index[-1], amplitude.iloc[-1], "x", color=style["color"], ms=5, mew=1
                )
        axes[0, column].set_title(FLOW_TITLES[flow])
        axes[-1, column].set_xlabel("$t\\Gamma_0/R_0^2$")
    labels = ("$E/E_0$", "$Z/Z_0$", "$-\\mathrm{d}E/\\mathrm{d}t$ [m$^5$/s$^3$]", "$\\max_m A_m$")
    for row, label in enumerate(labels):
        axes[row, 0].set_ylabel(label)
    for index, axis in enumerate(axes.flat):
        panel_label(axis, index, right=index in (0, 1))
    handles, names = axes[2, 0].get_legend_handles_labels()
    shared_legend(fig, handles, names, ncol=3)
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.12, top=0.95, hspace=0.12, wspace=0.3)
    save(fig, "histories")
    plt.close(fig)


if __name__ == "__main__":
    plot()
