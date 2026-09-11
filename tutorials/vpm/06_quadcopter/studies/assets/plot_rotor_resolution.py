"""Compare isolated-rotor results using native VPM/VLM samples."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openonda import plotting as theme
from openonda.tutorial_runner import load_case_module

root = Path(__file__).resolve().parents[1]
reader = load_case_module(root.parent, "assets._quadcopter_plots")
p = reader.rotor_inputs(root / "solution/coarse/vpm_metadata.json")
bem = reader.bem_reference(p)
scale = p.density * np.pi * p.radius**2 * (p.omega * p.radius) ** 2
reference = {
    "CT": bem.attrs["thrust"] / scale,
    "CP": bem.attrs["power"] / (scale * p.omega * p.radius),
}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--format", choices=("png", "pdf"), default="png")
args = parser.parse_args()
(root / "figures").mkdir(exist_ok=True)
theme.set_thesis_style()
fig, rows = plt.subplots(4, 1, figsize=(12.5 * theme.CM, 23 * theme.CM), constrained_layout=True)
axes = np.array([[rows[0], rows[2]], [rows[1], rows[3]]])
for name, label, color in (
    ("coarse", "$4\\times12$ panels, 3.75°", theme.COLORS["VPMpurple"]),
    ("time_refined", "$4\\times12$ panels, 1.875°", theme.COLORS["TUDcyan"]),
    ("mesh_refined", "$8\\times24$ panels, 3.75°", theme.COLORS["FVMorange"]),
):
    data = reader.performance(root / "samples" / name, p)
    data = data[data.time <= 3 * p.period + 1e-10].copy()
    complete = int((data.time.max() + 1e-10) // p.period)
    data["cycle"] = np.ceil((data.time - 1e-10) / p.period).astype(int)
    means = (
        data[(data.cycle > 0) & (data.cycle <= complete)]
        .groupby("cycle")[["CT", "CP", "thrust", "input_power"]]
        .mean()
    )
    for row, key in enumerate(("CT", "CP")):
        axes[row, 0].plot(data.revolutions, data[key], color=color, lw=1.1, label=label)
        axes[row, 1].plot(means.index, means[key], "-o", color=color, lw=1.2, ms=4)
for row, key in enumerate(("CT", "CP")):
    for ax in axes[row]:
        ax.axhline(reference[key], color="0.3", ls="--", lw=1, label="Attached-flow BEM")
        ax.set_xlim(0, 3.1)
    axes[row, 0].set_ylabel(rf"${key[0]}_{key[1]}$")
    axes[row, 1].set_ylabel(rf"Revolution mean ${key[0]}_{key[1]}$")
axes[0, 0].legend()
axes[0, 0].set_title("Native force and shaft-power samples")
axes[0, 1].set_title("Complete revolutions only")
axes[1, 0].set_xlabel("Revolutions")
axes[1, 1].set_xlabel("Revolution")
fig.suptitle("Isolated rotor: temporal and spatial resolution")
fig.savefig(
    root / "figures" / f"rotor_resolution.{args.format}", dpi=theme.DEFAULT_DPI, bbox_inches=None
)
