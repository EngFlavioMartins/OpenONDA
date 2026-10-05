"""Plot ``delta_wing_circulation_history.png``."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from openonda import plotting as _theme

from .postprocess import FIGURES_DIR, _save_figure, flow_integrals


def plot_circulation(samples_arg=None, destination=FIGURES_DIR, figure_format="png"):
    "Export the sampled vortex-strength history figure."
    _theme.set_thesis_style()
    data = flow_integrals(samples_arg)
    fig, ax = plt.subplots(figsize=(12.5 * _theme.CM, 7.0 * _theme.CM))
    _theme.centered_subplots_adjust(fig, outer=0.101, bottom=0.2, top=0.915)
    ax.plot(data.time, data.vortex_strength_magnitude_sum, color=_theme.COLORS["vpm"])
    ax.set(xlabel="Time [s]", ylabel="$\\sum_p |\\boldsymbol{\\Gamma}_p|$ [m$^3$/s]")
    _save_figure(fig, (ax,), destination / "delta_wing_circulation_history.png", figure_format)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=_theme.FORMAT_CHOICES, default="both")
    parser.add_argument(
        "--samples",
        type=Path,
        action="append",
        help="sample directory; repeat in sparse-to-dense order for a continuation",
    )
    args = parser.parse_args()
    plot_circulation(args.samples, FIGURES_DIR, args.format)


if __name__ == "__main__":
    main()
