#!/usr/bin/env python3
"""Compare native surface loads with the change in bound-plus-wake impulse."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

from openonda.tutorial_runner import case_package

__package__ = case_package(Path(__file__).resolve().parents[1]) + ".assets"
from ._plot_theme import FIG_DIR, SAMPLES_DIR, centered_subplots_adjust, cm, color, save_fig
from .results import parameters

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--case", default="exp_static_aoa08")
parser.add_argument("--format", choices=("png", "pdf"), default="png")
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

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12.5 * cm(), 10.8 * cm()))
centered_subplots_adjust(fig, outer=0.17, bottom=0.125, top=0.92, hspace=0.12)
for axis, direction, label in zip(axes, (lift, stream), ("Lift", "Drag"), strict=True):
    axis.plot(clock, measured @ direction, color=color("ref"), label="Fluid impulse", lw=1.5)
    axis.plot(
        clock,
        load_impulses[0] @ direction,
        "--",
        color=color("vpm"),
        label="Kutta--Joukowski",
        lw=1.5,
    )
    axis.plot(
        clock,
        load_impulses[1] @ direction,
        ":",
        color=color("TUDcyan"),
        label="Total surface load",
        lw=1.8,
    )
    axis.set_ylabel(label + r" impulse [N\,s]")
    discrepancy = (load_impulses[1][-1] - measured[-1]) @ direction
    print(
        f"{label} impulse: fluid={measured[-1] @ direction:.8g} N s; surface-fluid={discrepancy:+.8g} N s"
    )
axes[0].legend(loc="upper left")
axes[0].set_title("Surface force and fluid impulse")
axes[-1].set_xlabel("Time [s]")
save_fig(fig, FIG_DIR / f"{args.figure}.png", figure_format=args.format, dpi=args.dpi)
