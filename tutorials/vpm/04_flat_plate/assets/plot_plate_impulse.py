#!/usr/bin/env python3
"""Compare native surface loads with the change in bound-plus-wake impulse."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

from openonda.tutorial_runner import case_package

__package__ = case_package(Path(__file__).resolve().parents[1]) + ".assets"
from ._plot_theme import (
    validation_subplots,
    validation_legend,
    FIG_DIR,
    SAMPLES_DIR,
    color,
    save_fig,
)
from .results import parameters

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--case", default="exp_static_aoa08")
parser.add_argument("--format", choices=("png", "pdf", "both"), default="both")
parser.add_argument("--dpi", type=int, default=400)
parser.add_argument("--figure", default="plate_impulse")
args = parser.parse_args()

physics = parameters(SAMPLES_DIR.parent, args.case)
samples = SAMPLES_DIR / args.case
force = pd.read_csv(samples / "vlm_forces.csv")
flow = pd.read_csv(samples / "flow_integrals.csv")
flow = flow[(flow.time >= force.time.min()) & (flow.time <= force.time.max())]
clock = flow.time.to_numpy()
force_clock = force.time.to_numpy()
total = force[[f"force_{a}" for a in "xyz"]].to_numpy()
pressure = force[[f"unsteady_force_{a}" for a in "xyz"]].to_numpy()

# The pressure-time force is a backward difference over each accepted interval.
# Integrate that term with its own interval weights; KJ is an endpoint load.
kj = cumulative_trapezoid(total - pressure, force_clock, axis=0, initial=0)
pressure_impulse = np.vstack(
    (
        np.zeros(3),
        np.cumsum(
            pressure[1:] * np.diff(force_clock)[:, None],
            axis=0,
        ),
    )
)
load_impulses = []
for values in (kj, kj + pressure_impulse):
    sampled = np.column_stack([np.interp(clock, force_clock, values[:, i]) for i in range(3)])
    load_impulses.append(sampled - sampled[0])
fluid = physics["density"] * flow[[f"coupled_linear_impulse_{a}" for a in "xyz"]].to_numpy()
measured = -(fluid - fluid[0])
stream = physics["reference_velocity"] / physics["speed"]
lift = np.array([0.0, 0.0, 1.0]) - stream[2] * stream
lift /= np.linalg.norm(lift)

fig, axes = validation_subplots(3, height_cm=18, sharex=True, outer=0.115, top_padding_cm=0.54)
for axis, direction, label in zip(axes[:2], (lift, stream), ("Lift", "Drag"), strict=True):
    axis.plot(
        clock,
        measured @ direction,
        color=color("TUDdark"),
        marker="D",
        markevery=80,
        ms=2.5,
        label="Fluid impulse",
        lw=1.5,
    )
    axis.plot(
        clock,
        load_impulses[0] @ direction,
        "-",
        marker="o",
        markevery=80,
        ms=2.5,
        color=color("vpm"),
        label="Kutta--Joukowski only",
        lw=1.5,
    )
    axis.plot(
        clock,
        load_impulses[1] @ direction,
        "-",
        marker="s",
        markevery=80,
        ms=2.5,
        color=color("TUDcyan"),
        label="Total surface load",
        lw=1.8,
    )
    axis.set_ylabel(label + r" impulse [N\,s]")
    residual = (load_impulses[1] - measured) @ direction
    axes[2].plot(
        clock,
        100 * residual / abs(measured[-1] @ direction),
        label=label,
        color=color("vpm" if label == "Lift" else "TUDcyan"),
        ls="-",
        marker="o" if label == "Lift" else "s",
        markevery=80,
        ms=2.5,
    )
    discrepancy = (load_impulses[1][-1] - measured[-1]) @ direction
    print(
        f"{label} impulse: fluid={measured[-1] @ direction:.8g} N s; surface-fluid={discrepancy:+.8g} N s"
    )
validation_legend(fig, axes[0], ncol=1)
axes[2].set_ylabel(r"Residual [\%]")
axes[2].legend(frameon=False, loc="upper left", ncol=2)
print("Residual = total surface load minus fluid impulse")
print(args.case.replace("_", r"\_"))
print("Normalized by each final fluid-impulse magnitude")
axes[-1].set_xlabel("Time [s]")
save_fig(fig, FIG_DIR / f"{args.figure}.png", figure_format=args.format, dpi=args.dpi)
