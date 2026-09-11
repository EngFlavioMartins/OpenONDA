"""Compare isolated-rotor results using native VPM/VLM samples."""

import argparse
from pathlib import Path
import json
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
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--format", choices=("png", "pdf"), default="png")
args = parser.parse_args()
(root / "figures").mkdir(exist_ok=True)
theme.set_thesis_style()
fig, axes = plt.subplots(2, 1, figsize=(12.5 * theme.CM, 14 * theme.CM), constrained_layout=True)
for name, label, color in (
    ("coarse", "$4\\times12$, 3.75°", theme.COLORS["VPMpurple"]),
    ("time_refined", "$4\\times12$, 1.875°", theme.COLORS["TUDcyan"]),
    ("mesh_refined", "$8\\times24$, 3.75°", theme.COLORS["FVMorange"]),
):
    metadata = json.loads((root / "solution" / name / "vpm_metadata.json").read_text())
    surface = metadata["configuration"]["numerics"]["vlm"]["surfaces"][0]
    motion = surface["kinematics"]
    center = np.asarray(motion["rotation_centre"]["values"])
    omega = motion["angular_speed"]
    span = pd.read_csv(root / "samples" / name / f"vlm_spanwise_{surface['name']}.csv")
    chord = pd.read_csv(root / "samples" / name / f"vlm_chordwise_{surface['name']}.csv")
    span = span[(span.time > 2 * p.period + 1e-10) & (span.time <= 3 * p.period + 1e-10)].copy()
    chord = chord[(chord.time > 2 * p.period + 1e-10) & (chord.time <= 3 * p.period + 1e-10)].copy()
    # Undo the recorded rigid rotation; the authored quarter chord is radial y.
    theta = omega * chord.time
    chord["radius"] = -np.sin(theta) * (chord.bound_x - center[0]) + np.cos(theta) * (
        chord.bound_y - center[1]
    )
    radii = chord.groupby("station_id").radius.mean()
    span["thrust_per_radius"] = span.section_force_z / span.spanwise_station_width
    profile = span.groupby("station_id").agg(
        thrust_per_radius=("thrust_per_radius", "mean"),
        circulation=("circulation_magnitude", "mean"),
        section_force_z=("section_force_z", "mean"),
    )
    profile = profile.join(radii).sort_values("radius")
    axes[0].plot(
        profile.radius / p.radius,
        profile.thrust_per_radius,
        "o-",
        ms=3,
        lw=1,
        color=color,
        label=label,
    )
    axes[1].plot(
        profile.radius / p.radius, profile.circulation, "o-", ms=3, lw=1, color=color, label=label
    )
axes[0].plot(
    bem.normalized_radial_position,
    bem.thrust_per_radius / p.n_blades,
    "--",
    color="0.3",
    label="Attached-flow BEM",
)
axes[1].plot(
    bem.normalized_radial_position, bem.circulation, "--", color="0.3", label="Attached-flow BEM"
)
axes[0].set(ylabel="Thrust per blade and radius [N/m]", title="Axial blade loading")
axes[1].set(ylabel=r"Circulation $|\Gamma|$ [m$^2$/s]", title="Blade circulation")
for ax in axes:
    ax.set_xlabel(r"$r/R$")
axes[0].legend()
fig.suptitle("Third-revolution mean; native VLM samples")
fig.savefig(
    root / "figures" / f"rotor_loading.{args.format}", dpi=theme.DEFAULT_DPI, bbox_inches=None
)
