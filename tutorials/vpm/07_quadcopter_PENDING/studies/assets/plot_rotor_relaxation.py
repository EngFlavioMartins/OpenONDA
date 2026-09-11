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
references = {
    "CT": bem.attrs["thrust"] / scale,
    "CP": bem.attrs["power"] / (scale * p.omega * p.radius),
}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--format", choices=("png", "pdf"), default="png")
args = parser.parse_args()
(root / "figures").mkdir(exist_ok=True)
theme.set_thesis_style()
fig, rows = plt.subplots(4, 1, figsize=(12.5 * theme.CM, 25 * theme.CM), constrained_layout=True)
axes = np.array([[rows[0], rows[2]], [rows[1], rows[3]]])
for name, label, color in (
    ("relaxed", "3.75°, relaxation 0.3", theme.COLORS["VPMpurple"]),
    ("relaxed_time_refined", "1.875°, relaxation 0.3", theme.COLORS["TUDcyan"]),
    ("coarse", "3.75°, no relaxation", theme.COLORS["FVMorange"]),
    ("time_refined", "1.875°, no relaxation", theme.COLORS["AccentGreen"]),
):
    data = reader.performance(root / "samples" / name, p)
    if name == "coarse":
        for continuation_name in ("continue_8", "continue_12"):
            continuation = reader.performance(root / "samples" / continuation_name, p)
            data = pd.concat((data, continuation)).sort_values("time").drop_duplicates("time")
    complete = int((data.time.max() + 1e-10) // p.period)
    data["cycle"] = np.ceil((data.time - 1e-10) / p.period).astype(int)
    means = data[data.cycle <= complete].groupby("cycle")[["CT", "CP"]].mean()
    for row, key in enumerate(("CT", "CP")):
        axes[row, 0].plot(data.revolutions, data[key], color=color, lw=1, label=label)
        axes[row, 1].plot(means.index, means[key], "-o", color=color, lw=1.2, ms=4)
for row, key in enumerate(("CT", "CP")):
    for ax in axes[row]:
        ax.axhline(references[key], color="0.3", ls="--", lw=1, label="Attached-flow BEM")
        ax.set_xlim(left=0)
    axes[row, 0].set_ylabel(rf"${key[0]}_{key[1]}$")
    axes[row, 1].set_ylabel(rf"Revolution mean ${key[0]}_{key[1]}$")
axes[0, 0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.03))
for ax in axes[:, 0]:
    for restart in (6, 8):
        ax.axvline(restart, color="0.6", ls=":", lw=0.8)
axes[1, 0].annotate(
    "Native restart",
    xy=(6, 0.04),
    xycoords=("data", "axes fraction"),
    xytext=(4, 0),
    textcoords="offset points",
    rotation=90,
    va="bottom",
    color="0.4",
)
axes[0, 1].set_title("Complete revolutions only")
axes[1, 0].set_xlabel("Revolutions")
axes[1, 1].set_xlabel("Revolution")
fig.suptitle("Isolated rotor: relaxation comparison")
fig.savefig(
    root / "figures" / f"rotor_relaxation.{args.format}", dpi=theme.DEFAULT_DPI, bbox_inches=None
)
