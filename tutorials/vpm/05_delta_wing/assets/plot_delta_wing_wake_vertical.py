"""Plot ``delta_wing_wake_vertical.png``."""

import argparse
from pathlib import Path

from openonda import plotting as _theme

from .postprocess import FIGURES_DIR, _plot_wake_field


def plot_wake_vertical(
    samples_arg=None, destination=FIGURES_DIR, figure_format="png", solution_dirs=None
):
    "Export the vertical wake-field figure from native wake-plane samples."
    _plot_wake_field(
        samples_arg, destination, figure_format, "vertical", solution_dirs=solution_dirs
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=_theme.FORMAT_CHOICES, default="both")
    parser.add_argument(
        "--samples",
        type=Path,
        action="append",
        help="sample directory; repeat in sparse-to-dense order for a continuation",
    )
    parser.add_argument(
        "--solution",
        type=Path,
        action="append",
        help="matching solution directory for each --samples directory",
    )
    args = parser.parse_args()
    plot_wake_vertical(args.samples, FIGURES_DIR, args.format, args.solution)


if __name__ == "__main__":
    main()
