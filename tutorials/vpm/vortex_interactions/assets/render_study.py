"""Render recorded particle geometry; no invented intermediate simulation states."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .. import setup
from ..study import STUDY_DIR


def states(run):
    if run == "leapfrog_baseline" and not (STUDY_DIR / run).exists():
        import h5py

        rings = [setup.create_ring(x, group).build() for group, x in enumerate((-0.5, 0.5))]
        result = [
            {
                "time": 0.0,
                "step": 0,
                **{
                    key: np.concatenate([getattr(ring, key) for ring in rings])
                    for key in ("position", "vortex_strength", "core_radius", "group_id")
                },
            }
        ]
        for path in sorted((setup.TUTORIAL_DIR / "solution" / "baseline").glob("*.h5")):
            if int(path.stem.rsplit("_", 1)[1]) % 100:
                continue
            with h5py.File(path) as f:
                result.append(
                    {
                        "time": float(f["solver"].attrs["time"]),
                        "step": int(f["solver"].attrs["step"]),
                        **{
                            key: f["particles"][key][:]
                            for key in ("position", "vortex_strength", "core_radius", "group_id")
                        },
                    }
                )
        return result
    return [
        dict(np.load(path))
        for path in sorted(
            (STUDY_DIR / run / "samples" / "diagnostics" / "particles").glob("*.npz")
        )
    ]


def draw(axis, state, *, centre=False):
    p = np.array(state["position"], dtype=float)
    strength = np.linalg.norm(state["vortex_strength"], axis=1)
    if centre:
        p[:, 0] -= np.average(p[:, 0], weights=strength)
    for group, color in ((0, "#277da8"), (1, "#d35b3e")):
        indices = np.flatnonzero(state["group_id"] == group)
        if not len(indices):
            continue
        # Deterministic strongest-particle subset makes ring cores readable.
        selected = indices[np.argsort(strength[indices])[-min(3500, len(indices)) :]]
        axis.scatter(
            p[selected, 0],
            p[selected, 1],
            p[selected, 2],
            s=1.0,
            alpha=0.35,
            color=color,
            rasterized=True,
        )
    radius = max(1.5, float(np.quantile(np.hypot(p[:, 1], p[:, 2]), 0.99)))
    midpoint = 0.5 * (p[:, 0].min() + p[:, 0].max())
    axis.set(
        xlim=(midpoint - radius, midpoint + radius),
        ylim=(-radius, radius),
        zlim=(-radius, radius),
        xlabel="x/R0",
        ylabel="y/R0",
        zlabel="z/R0",
    )
    axis.tick_params(labelsize=7, pad=0)
    for label in (axis.xaxis.label, axis.yaxis.label, axis.zaxis.label):
        label.set_fontsize(8)
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=18, azim=-58)
    axis.set_title(f"t Γ0/R0² = {float(state['time']) * np.pi:.2f}  |  N = {len(p):,}", fontsize=9)


def render(run, output):
    snapshots = states(run)
    if not snapshots:
        raise ValueError(f"No particle snapshots for {run}")
    output.mkdir(parents=True, exist_ok=True)
    selected = np.unique(
        np.round(np.linspace(0, len(snapshots) - 1, min(4, len(snapshots)))).astype(int)
    )
    fig = plt.figure(figsize=(4.7 * len(selected), 5))
    fig.subplots_adjust(left=0.015, right=0.96, bottom=0.12, top=0.82, wspace=0.06)
    for index, frame in enumerate(selected):
        draw(fig.add_subplot(1, len(selected), index + 1, projection="3d"), snapshots[frame])
    fig.suptitle(
        run + " — recorded particle cores (colors are particle labels)", fontsize=12, y=0.97
    )
    fig.savefig(output / f"{run}_evolution.png", dpi=160)
    plt.close(fig)
    frames = []
    for state in snapshots:
        fig = plt.figure(figsize=(6, 5), constrained_layout=True)
        draw(fig.add_subplot(111, projection="3d"), state, centre=True)
        fig.suptitle(run + " — translating view of recorded particle cores", fontsize=10)
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=100)
        plt.close(fig)
        buffer.seek(0)
        frames.append(Image.open(buffer).convert("RGB"))
    frames[0].save(
        output / f"{run}.gif", save_all=True, append_images=frames[1:], duration=450, loop=0
    )
    print(output / f"{run}_evolution.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--output", type=Path, default=setup.TUTORIAL_DIR / "figures" / "study")
    args = parser.parse_args()
    for run in args.runs:
        render(run, args.output)
