"""Plot one isolated rotor using its native metadata and force samples."""

import argparse
from pathlib import Path

from openonda.tutorial_runner import load_case_module

root = Path(__file__).resolve().parents[1]
reader = load_case_module(root.parent, "assets._quadcopter_plots")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--case", default="continue_12")
parser.add_argument("--format", choices=("png", "pdf"), default="png")
args = parser.parse_args()
reader.plot_performance(
    root / "samples" / args.case,
    root / "figures" / args.case,
    args.format,
    metadata_path=root / "solution" / args.case / "vpm_metadata.json",
)
