#!/usr/bin/env python3
"""Rotor performance after the first impulsive sample and steady references."""

if not __package__:
    from pathlib import Path
    from openonda.tutorial_runner import case_package

    __package__ = case_package(Path(__file__).resolve().parents[1]) + ".assets"

import numpy as np
from ._common import (
    build_arg_parser,
    load_theme,
    FIGURES_DIR,
    performance,
    bem_reference,
    rotor_inputs,
    read_operating_point,
    rotor_subplots,
    save_rotor_figure,
    OPERATING_WINDOW_REVOLUTIONS,
)


def main():
    args = build_arg_parser(__doc__).parse_args()
    colors, _ = load_theme()
    data, bem, p = performance(), bem_reference(), rotor_inputs()
    end = float(data.time.max())
    start = end - OPERATING_WINDOW_REVOLUTIONS * p.rotation_period
    mean = read_operating_point(window_start=start, window_end=end)
    tail = data[data.time >= start]
    fig, axes = rotor_subplots(2, height_cm=11)
    # Presentation request: omit exactly the initial impulsive load sample.
    # Keep the native history and operating-window calculations unchanged.
    displayed = data.iloc[1:]
    for key, ink, reference_key, reference_style in [
        ("CT", colors["VPMpurple"], "thrust_coefficient", "--"),
        ("CP", colors["TUDcyan"], "power_coefficient", ":"),
    ]:
        reference = bem.attrs[reference_key]
        axes[0].plot(
            displayed.nominal_revolutions, displayed[key], color=ink, label=rf"VLM+VPM $C_{key[1]}$"
        )
        axes[0].axhline(
            reference,
            color=colors["reference"],
            ls=reference_style,
            label=rf"BEM $C_{key[1]}$",
        )
    axes[0].set(
        ylabel=r"$C_T,\ C_P$", xlabel="Nominal revolutions", xlim=(0, end / p.rotation_period)
    )
    axes[0].legend(
        loc="lower right",
        ncol=2,
        frameon=False,
        columnspacing=0.8,
        handlelength=1.5,
        handletextpad=0.5,
    )
    ct = np.linspace(0, 1, 300)
    axes[1].plot(
        ct,
        0.5 * ct * (1 + np.sqrt(1 - ct)),
        "--",
        color=colors["reference"],
        label="Ideal actuator disk",
    )
    axes[1].plot(tail.CT, tail.CP, color=colors["VPMpurple"], lw=1)
    axes[1].plot(*mean, "o", color=colors["VPMpurple"], zorder=5, label="VLM+VPM")
    reference = (bem.attrs["thrust_coefficient"], bem.attrs["power_coefficient"])
    axes[1].plot(*reference, "x", color=colors["reference"], zorder=5, label="BEM", linestyle="--")
    axes[1].set(xlabel=r"$C_T$", ylabel=r"$C_P$")
    axes[1].legend(loc="upper left", frameon=False)
    save_rotor_figure(
        fig, FIGURES_DIR / "rotor_performance.png", figure_format=args.format, dpi=args.dpi
    )


if __name__ == "__main__":
    main()
