"""Compare rotor wake consistency using native flow-integral and force samples."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

from openonda import plotting as theme
from openonda.tutorial_runner import load_case_module


ROOT = Path(__file__).resolve().parents[1]
reader = load_case_module(ROOT.parent, "assets._quadcopter_plots")


def impulse_ratios(forces, integrals, parameters):
    """Compare wake impulse changes and integrated blade thrust over full revolutions."""
    period = parameters.period
    end = min(forces.time.max(), integrals.time.max())
    complete = int(np.floor((end + 1e-10) / period))
    cycles, ratios = [], []
    # The first saved integral is after startup; use complete sampled intervals.
    first = max(2, int(np.ceil(integrals.time.min() / period)) + 1)
    for cycle in range(first, complete + 1):
        start, finish = (cycle - 1) * period, cycle * period
        internal = forces.time[(forces.time > start + 1e-10) & (forces.time < finish - 1e-10)]
        times = np.r_[start, internal, finish]
        blade_impulse = trapezoid(np.interp(times, forces.time, forces.thrust), times)
        wake_impulse = np.interp([start, finish], integrals.time, integrals.linear_impulse_z)
        cycles.append(cycle)
        ratios.append(-parameters.density * np.diff(wake_impulse)[0] / blade_impulse)
    return cycles, ratios


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=("coarse", "relaxed"))
    parser.add_argument("--format", choices=("png", "pdf"), default="png")
    args = parser.parse_args()
    labels = {
        "coarse": ("No relaxation", theme.COLORS["FVMorange"]),
        "relaxed": ("Relaxation 0.3", theme.COLORS["VPMpurple"]),
        "relaxed_moments": ("Relaxation 0.3, preserve moments", theme.COLORS["TUDcyan"]),
    }
    theme.set_thesis_style()
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12.5 * theme.CM, 18 * theme.CM),
        constrained_layout=True,
        sharex=True,
    )
    for name in args.cases:
        p = reader.rotor_inputs(ROOT / "solution" / name / "vpm_metadata.json")
        names = (name, "continue_8", "continue_12") if name == "coarse" else (name,)
        forces = pd.concat([reader.performance(ROOT / "samples" / item, p) for item in names])
        integrals = pd.concat(
            [pd.read_csv(ROOT / "samples" / item / "flow_integrals.csv") for item in names]
        )
        forces = forces.sort_values("time")
        integrals = integrals.sort_values("time")
        label, color = labels[name]
        revolutions = integrals.time / p.period
        axes[0].plot(
            revolutions, integrals.vortex_strength_misalignment_degrees, color=color, label=label
        )
        axes[1].plot(revolutions, integrals.vorticity_divergence_error, color=color)
        cycles, ratios = impulse_ratios(forces, integrals, p)
        axes[2].plot(cycles, ratios, "-o", color=color, markersize=3)
    axes[0].set(ylabel="Misalignment [°]", title="Particle strength versus induced vorticity")
    axes[0].legend()
    axes[1].set(ylabel="Normalized divergence")
    axes[2].axhline(1, color=theme.COLORS["DarkText"], linestyle="--", linewidth=1)
    axes[2].set(
        xlabel="Revolutions",
        ylabel=r"$-\rho\,\Delta I_z\,/\,\int T\,dt$",
        title="Wake impulse / blade impulse",
    )
    axes[2].set_xlim(left=0)
    (ROOT / "figures").mkdir(exist_ok=True)
    theme.save_fig(
        fig,
        ROOT / "figures/rotor_health.png",
        figure_format=args.format,
        dpi=theme.DEFAULT_DPI,
        bbox_inches=None,
    )


if __name__ == "__main__":
    main()
