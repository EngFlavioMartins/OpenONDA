"""Render reconstructed Gaussian vorticity, rather than particle locations."""

from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from .render_study import states
from .. import setup


def vorticity(targets, state, chunk=32):
    position = np.asarray(state["position"], dtype=float)
    strength = np.asarray(state["vortex_strength"], dtype=float)
    sigma = np.asarray(state["core_radius"], dtype=float)
    result = []
    for start in range(0, len(targets), chunk):
        d = targets[start : start + chunk, None, :] - position[None, :, :]
        weights = np.exp(-np.sum(d * d, axis=2) / sigma**2) / (np.pi**1.5 * sigma**3)
        result.append(weights @ strength)
    return np.concatenate(result)


def render(run, output, steps):
    selected = [s for s in states(run) if int(s["step"]) in steps]
    if not selected:
        raise ValueError("No requested snapshots")
    centres = [
        np.average(s["position"][:, 0], weights=np.linalg.norm(s["vortex_strength"], axis=1))
        for s in selected
    ]
    axial_extent = max(
        1.5,
        max(
            float(np.max(np.abs(s["position"][:, 0] - centre)))
            + 3 * float(np.max(s["core_radius"]))
            for s, centre in zip(selected, centres)
        ),
    )
    radial_extent = max(
        1.8,
        max(
            float(np.max(np.hypot(s["position"][:, 1], s["position"][:, 2])))
            + 3 * float(np.max(s["core_radius"]))
            for s in selected
        ),
    )
    x = np.linspace(-axial_extent, axial_extent, 160)
    y = np.linspace(0, radial_extent, 160)
    xx, yy = np.meshgrid(x, y)
    fields = []
    for s, centre in zip(selected, centres):
        targets = np.column_stack((xx.ravel() + centre, yy.ravel(), np.zeros(xx.size)))
        fields.append(np.linalg.norm(vorticity(targets, s), axis=1).reshape(xx.shape))
        print(run, int(s["step"]), float(fields[-1].max()), flush=True)
    maximum = max(float(f.max()) for f in fields)
    fig, axes = plt.subplots(
        1, len(fields), figsize=(4 * len(fields), 3.8), squeeze=False, constrained_layout=True
    )
    for ax, s, f, centre in zip(axes.flat, selected, fields, centres):
        im = ax.pcolormesh(
            xx,
            yy,
            f,
            norm=LogNorm(vmin=maximum * 1e-3, vmax=maximum, clip=True),
            cmap="magma",
            shading="auto",
        )
        ax.contour(
            xx,
            yy,
            f,
            levels=[maximum * 0.05, maximum * 0.2, maximum * 0.5],
            colors="white",
            linewidths=0.55,
            alpha=0.7,
        )
        ax.set(xlabel="(x − strength centroid)/R0", ylabel="y/R0", aspect="equal")
        ax.set_title(
            f"t Γ0/R0² = {float(s['time']) * np.pi:.2f}\nx centroid = {centre:.2f} R0", fontsize=9
        )
    fig.colorbar(im, ax=axes.ravel().tolist(), label="|ω| [1/s], common scale", shrink=0.8)
    fig.suptitle(run + " — reconstructed vorticity in the z=0 meridional plane", fontsize=11)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / (run + "_vorticity.png"), dpi=160)
    plt.close(fig)
    np.savez_compressed(
        output / (run + "_vorticity_planes.npz"),
        x=x,
        y=y,
        time=[float(s["time"]) for s in selected],
        centre_x=centres,
        vorticity_magnitude=np.array(fields),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    p.add_argument("--steps", nargs="+", type=int, default=[0, 300, 600, 1200])
    p.add_argument("--output", type=Path, default=setup.TUTORIAL_DIR / "figures" / "study")
    a = p.parse_args()
    render(a.run, a.output, a.steps)
