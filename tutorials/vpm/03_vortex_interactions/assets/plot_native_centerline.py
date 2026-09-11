#!/usr/bin/env python3
"""Plot full-azimuth particle centerline proxies from one native checkpoint.

These strength-weighted conditional means describe particle geometry, not a
reconstructed vorticity isosurface. They complement meridional slices, which
can intersect identical phases of the imposed azimuthal disturbance.
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from openonda import plotting as theme


ROOT = Path(__file__).resolve().parents[1]


def centerline(position, strength, bins=128):
    """Return equally spaced azimuth bins and centered axial/radial means."""
    weight = np.linalg.norm(strength, axis=1)
    centre = np.average(position, axis=0, weights=weight)
    relative = position - centre
    angle = np.arctan2(relative[:, 2], relative[:, 1]) % (2 * np.pi)
    radius = np.linalg.norm(relative[:, 1:], axis=1)
    index = np.minimum((angle * bins / (2 * np.pi)).astype(int), bins - 1)
    bin_weight = np.bincount(index, weights=weight, minlength=bins)
    if np.any(bin_weight <= 0):
        raise ValueError("Incomplete azimuthal coverage: reduce the bin count")
    axial = np.bincount(index, weights=weight * relative[:, 0], minlength=bins) / bin_weight
    radial = np.bincount(index, weights=weight * radius, minlength=bins) / bin_weight
    angle = (np.arange(bins) + 0.5) * 360 / bins
    return angle, axial - axial.mean(), radial - radial.mean()


def render(run, step, bins):
    solution = ROOT / "solution" / run
    metadata = json.loads((solution / "vpm_metadata.json").read_text())
    radius = float(metadata["configuration"]["initial_conditions"][0]["radius"])
    path = solution / f"vpm_{step:06d}.h5"
    with h5py.File(path, "r") as saved:
        time = float(saved["solver"].attrs["time"])
        position = saved["particles/position"][:].astype(float)
        strength = saved["particles/vortex_strength"][:].astype(float)
        group = saved["particles/group_id"][:]
    if not np.isfinite(position).all() or not np.isfinite(strength).all():
        raise ValueError("Native particle geometry or strength is not finite")

    theme.set_thesis_style()
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=theme.figure_size("stacked"))
    fig.subplots_adjust(left=0.20, right=0.97, top=0.82, bottom=0.13, hspace=0.30)
    for identity in np.unique(group):
        selected = group == identity
        angle, axial, radial = centerline(position[selected], strength[selected], bins)
        label = f"Lineage {int(identity)}"
        axes[0].plot(angle, axial / radius, label=label)
        axes[1].plot(angle, radial / radius, label=label)
    for ax in axes:
        ax.axhline(0, color=theme.PALETTE["gray"], linewidth=0.6, zorder=0)
        ax.set_xlim(0, 360)
        ax.set_xticks((0, 90, 180, 270, 360))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
    axes[0].set_ylabel(r"$\delta x/R_0$")
    axes[1].set_ylabel(r"$\delta r/R_0$")
    axes[1].set_xlabel(r"Azimuth $\theta$ [degrees]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.55, 0.91), ncol=2)
    fig.suptitle(rf"Native centerline proxy: $t={time:g}$ s", y=0.98)
    output = ROOT / "figures" / run / "centerline"
    output.mkdir(parents=True, exist_ok=True)
    stem = output / f"native_centerline_{step:06d}"
    for extension in theme.EXPORT_FORMATS:
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=theme.DEFAULT_DPI)
    plt.close(fig)
    print(f"Saved {stem}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--bins", type=int, default=128)
    args = parser.parse_args()
    if args.bins < 2:
        parser.error("--bins must be at least 2")
    render(args.run, args.step, args.bins)
