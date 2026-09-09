#!/usr/bin/env python3
"""Rotor thrust/power histories and the ideal actuator-disk operating envelope."""

if not __package__:
    from pathlib import Path
    from openonda.tutorial_runner import case_package

    __package__ = case_package(Path(__file__).resolve().parents[1]) + ".assets"

import numpy as np
import matplotlib.pyplot as plt
from ._common import (
    build_arg_parser,
    load_theme,
    FIGURES_DIR,
    performance,
    bem_reference,
    rotor_inputs,
)


def main():
    args = build_arg_parser(__doc__).parse_args()
    colors, theme = load_theme()
    data, bem = performance(), bem_reference()
    fig, axes = plt.subplots(2, 1, figsize=theme.figure_size("stacked"), constrained_layout=True)
    for key, ink in [("CT", colors["VPMpurple"]), ("CP", colors["TUDcyan"])]:
        axes[0].plot(data.nominal_revolutions, data[key], color=ink, label=rf"$C_{key[1]}$")
        reference = bem.attrs["thrust_coefficient" if key == "CT" else "power_coefficient"]
        axes[0].axhline(reference, color=ink, ls="--", label=rf"BEM $C_{key[1]}$")
    axes[0].set(
        xlabel="Nominal revolutions", ylabel="Coefficient", title="Rotor thrust and shaft power"
    )
    axes[0].legend(ncol=2)
    ct = np.linspace(0, 1, 300)
    axes[1].plot(
        ct,
        0.5 * ct * (1 + np.sqrt(1 - ct)),
        "--",
        color=colors["reference"],
        label="Ideal steady disk",
    )
    axes[1].plot(data.CT, data.CP, color=colors["TUDcyan"], lw=1, alpha=0.8, label="VLM/VPM")
    tail = data[data.time > data.time.max() - 6 * rotor_inputs().rotation_period]
    start, end = tail.nominal_revolutions.agg(["min", "max"])
    axes[1].scatter(
        tail.CT.mean(),
        tail.CP.mean(),
        color=colors["VPMpurple"],
        s=24,
        label=f"Mean, rev {start:.1f}–{end:.1f}",
        zorder=5,
    )
    axes[1].scatter(
        bem.attrs["thrust_coefficient"],
        bem.attrs["power_coefficient"],
        marker="x",
        color=colors["reference"],
        s=28,
        label="BEM",
        zorder=5,
    )
    axes[1].set(xlabel=r"Thrust, $C_T$", ylabel=r"Power, $C_P$", title="Operating point")
    axes[1].legend()
    theme.save_fig(
        fig,
        FIGURES_DIR / "rotor_performance.png",
        figure_format=args.format,
        dpi=args.dpi,
        bbox_inches=None,
    )


if __name__ == "__main__":
    main()
