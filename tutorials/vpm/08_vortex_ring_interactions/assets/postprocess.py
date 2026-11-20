"""Tables, run status and figure styles shared by the plots of this case."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from openonda import plotting

import setup

CASE_DIR = Path(__file__).resolve().parents[1]
FLOWS = ("collision", "leapfrog")
FLOW_TITLES = {"collision": "Head-on collision", "leapfrog": "Leapfrogging"}
MODEL_STYLES = {
    "dns": {"label": "DNS", "color": plotting.COLORS["dark"], "linestyle": "--", "marker": "o"},
    "les": {"label": "LES", "color": plotting.COLORS["amber"], "linestyle": "-", "marker": "s"},
}


def read_table(case, name):
    """One CSV table of ``samples/<case>``; a leading ``# time = t`` line is skipped."""
    path = CASE_DIR / "samples" / case / f"{name}.csv"
    with path.open(encoding="utf-8") as stream:
        skip = int(stream.readline().startswith("#"))
    return pd.read_csv(path, skiprows=skip)


def run_status(case):
    """Stop time and reason of one run, from its metadata."""
    metadata = json.loads((CASE_DIR / "solution" / case / "vpm_metadata.json").read_text())
    return metadata["run_status"]


def figure_size(height_cm):
    return (plotting.MAX_FIGURE_WIDTH_CM * plotting.CM, height_cm * plotting.CM)


def panel_label(axis, index, *, right=False):
    axis.text(
        0.96 if right else 0.04,
        0.94,
        f"({chr(97 + index)})",
        transform=axis.transAxes,
        ha="right" if right else "left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2, "alpha": 0.8},
        zorder=5,
    )


def shared_legend(fig, handles, labels, *, bottom=0.02, ncol=2):
    return fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(0.5, bottom),
        ncol=ncol,
        frameon=True,
        fancybox=True,
        borderaxespad=0,
        handlelength=1.8,
        handletextpad=0.5,
        columnspacing=1.0,
        labelspacing=0.25,
    )


def save(fig, name):
    path = CASE_DIR / "figures" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plotting.export_figure(fig, path, figure_format="png")


def energy_budget(case):
    """Flow integrals with the energy rates split into molecular, sub-grid and numerical parts.

    ``-dE/dt`` comes from the sampled energy, ``eps_mol = nu Z`` and
    ``eps_sgs = -viscous_kinetic_energy_rate - eps_mol``; the remainder
    ``eps_num = -dE/dt - eps_mol - eps_sgs`` is the energy the particle
    discretisation removes, mostly by regenerating the particles on the lattice.
    """
    table = read_table(case, "flow_integrals")
    table["molecular_dissipation"] = setup.KINEMATIC_VISCOSITY * table["total_enstrophy"]
    table["subgrid_dissipation"] = (
        -table["viscous_kinetic_energy_rate"] - table["molecular_dissipation"]
    )
    rate = -np.gradient(table["total_kinetic_energy"].to_numpy(), table["time"].to_numpy())
    table["energy_loss_rate"] = rate
    table["numerical_dissipation"] = (
        rate - table["molecular_dissipation"] - table["subgrid_dissipation"]
    )
    return table
