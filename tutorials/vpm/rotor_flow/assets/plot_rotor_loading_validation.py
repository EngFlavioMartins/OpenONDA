#!/usr/bin/env python3
"""Time-averaged blade circulation and section lift against the recorded-geometry BEM reference."""

if not __package__:
    from pathlib import Path
    from openonda.tutorial_runner import case_package

    __package__ = case_package(Path(__file__).resolve().parents[1]) + ".assets"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ._common import build_arg_parser, load_theme, FIGURES_DIR, rotor_inputs, bem_reference


def main():
    args = build_arg_parser(__doc__).parse_args()
    inputs = rotor_inputs()
    surface = inputs.metadata["configuration"]["numerics"]["vlm"]["surfaces"][0]["name"]
    span = pd.read_csv(inputs.samples_dir / f"vlm_spanwise_{surface}.csv")
    cutoff = span.time.max() - 6 * inputs.rotation_period
    span = span[span.time >= cutoff]
    start, end = span.time.agg(["min", "max"]) / inputs.rotation_period
    chord = pd.read_csv(inputs.samples_dir / f"vlm_chordwise_{surface}.csv")
    chord = chord[chord.time >= cutoff]
    positions = chord.groupby(["step", "station_id"])[["bound_y", "bound_z"]].mean()
    positions["radius"] = np.linalg.norm(positions, axis=1)
    sampled = span.merge(positions[["radius"]], on=["step", "station_id"])
    sampled = (
        sampled.groupby("station_id")
        .agg(
            radius=("radius", "mean"),
            circulation=("circulation_magnitude", "mean"),
            cl=("section_lift_coefficient_from_circulation", "mean"),
        )
        .sort_values("radius")
    )
    bem = bem_reference()
    colors, theme = load_theme()
    fig, axes = plt.subplots(2, 1, figsize=theme.figure_size("stacked"), constrained_layout=True)
    circulation_scale = inputs.freestream_speed * inputs.rotor_radius
    for axis, actual, reference in [
        (axes[0], sampled.circulation / circulation_scale, bem.circulation / circulation_scale),
        (axes[1], sampled.cl, bem.lift_coefficient),
    ]:
        axis.plot(
            sampled.radius / inputs.rotor_radius,
            actual,
            "o-",
            ms=3,
            color=colors["VPMpurple"],
            label=f"Mean, rev {start:.1f}–{end:.1f}",
        )
        axis.plot(
            bem.normalized_radial_position, reference, "--", color=colors["reference"], label="BEM"
        )
        axis.set_xlabel(r"Radius, $r/R$")
        axis.legend()
    axes[0].set(ylabel=r"Circulation, $\Gamma/(U_\infty R)$", title="Blade circulation")
    axes[1].set(ylabel=r"Section lift, $c_l$", title="Local aerodynamic loading")
    theme.save_fig(
        fig,
        FIGURES_DIR / "rotor_loading_validation.png",
        figure_format=args.format,
        dpi=args.dpi,
        bbox_inches=None,
    )


if __name__ == "__main__":
    main()
