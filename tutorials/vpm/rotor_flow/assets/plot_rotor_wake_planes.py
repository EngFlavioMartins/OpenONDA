#!/usr/bin/env python3
"""Compare native downstream velocity samples with an ideal actuator-disk wake."""

from __future__ import annotations

if not __package__:
    from pathlib import Path
    from openonda.tutorial_runner import case_package

    __package__ = case_package(Path(__file__).resolve().parents[1]) + ".assets"

from defusedxml import ElementTree
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from openonda import plotting as theme
from openonda.rotor_theory import (
    actuator_disk_velocity_ratio,
    axial_induction_factor_from_thrust_coefficient,
)
from ._common import FIGURES_DIR, build_arg_parser, read_operating_point, rotor_inputs


def relative_drift(values):
    """Compare two halves of a time window, normalized by its mean magnitude."""
    values = np.asarray(values)
    half = len(values) // 2
    return abs(values[:half].mean() - values[half:].mean()) / max(abs(values.mean()), 1e-12)


def plane_profiles(p, rotations=3):
    """Read published PVD times and velocity arrays; no solver state reconstruction."""
    edges = np.linspace(0, 1.25, 33)
    radius = 0.5 * (edges[1:] + edges[:-1])
    records = []
    for pvd in sorted(p.samples_dir.glob("slice_x*m.pvd")):
        frames = ElementTree.parse(pvd).findall(".//DataSet")
        end = max(float(frame.attrib["timestep"]) for frame in frames)
        selected = [
            frame
            for frame in frames
            if float(frame.attrib["timestep"]) > end - rotations * p.rotation_period
        ]
        profiles, means, times = [], [], []
        for frame in selected:
            grid = pv.read(pvd.parent / frame.attrib["file"])
            points = np.asarray(grid.points)
            r = np.linalg.norm(points[:, 1:], axis=1) / p.rotor_radius
            velocity = np.asarray(grid["velocity"])[:, 0] / p.freestream_speed
            indices = np.digitize(r, edges) - 1
            valid = (indices >= 0) & (indices < len(radius))
            sums = np.bincount(indices[valid], weights=velocity[valid], minlength=len(radius))
            counts = np.bincount(indices[valid], minlength=len(radius))
            profiles.append(
                np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
            )
            means.append(velocity[r <= 1].mean())
            times.append(float(frame.attrib["timestep"]))
        records.append(
            dict(
                name=pvd.stem,
                x=points[0, 0],
                radius=radius,
                mean=np.nanmean(profiles, axis=0),
                times=np.asarray(times),
                disc_means=np.asarray(means),
            )
        )
    if not records:
        raise FileNotFoundError(f"No published rotor wake planes in {p.samples_dir}")
    return sorted(records, key=lambda row: row["x"])


def main():
    args = build_arg_parser(__doc__).parse_args()
    theme.set_thesis_style()
    p = rotor_inputs()
    ct, _ = read_operating_point()
    induction = axial_induction_factor_from_thrust_coefficient(ct)
    fig, ax = plt.subplots(figsize=theme.figure_size("stacked"), constrained_layout=True)
    for row in plane_profiles(p):
        drift = relative_drift(row["disc_means"])
        start, end = row["times"][[0, -1]] / p.rotation_period
        label = (
            f"{row['x'] / p.rotor_radius:g}R; rev {start:.1f}–{end:.1f}; drift {100 * drift:.1f}\\%"
        )
        ax.plot(row["radius"], row["mean"], ls="-" if drift <= 0.01 else ":", label=label)
        print(f"{row['name']}: {len(row['times'])} frames; disc-mean velocity drift {drift:.2%}")
    ax.axhline(1, color="0.5", lw=0.7, label="Freestream")
    if np.isfinite(induction) and induction < 0.5:
        r = np.linspace(0, 1.25, 500)
        ax.plot(
            r,
            actuator_disk_velocity_ratio(induction, r),
            color="0.2",
            ls="--",
            label=f"Ideal far wake, CT={ct:.3f}",
        )
        ax.plot([0, 1], [1 - induction] * 2, color="0.5", ls=":", label="Ideal disk velocity")
    else:
        ax.text(
            0.02, 0.03, f"CT={ct:.3f}: outside ideal low-induction branch", transform=ax.transAxes
        )
    ax.set(
        xlabel=r"$r/R$",
        ylabel=r"$\langle u_x/U_\infty\rangle_{t,\theta}$",
        xlim=(0, 1.25),
        title="Time and azimuthal mean",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22))
    theme.save_fig(
        fig,
        FIGURES_DIR / "rotor_wake_planes.png",
        figure_format=args.format,
        dpi=args.dpi,
        bbox_inches=None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
