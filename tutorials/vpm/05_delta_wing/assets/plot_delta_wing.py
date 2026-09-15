#!/usr/bin/env python3
"""Plot native delta-wing results, distinguishing partial and completed runs."""

from pathlib import Path

from openonda.tutorial_runner import case_package

__package__ = case_package(Path(__file__).resolve().parents[1]) + ".assets"

import argparse
import json

from ._delta_wing_plots import CASE_DIR, FIGURES_DIR, SAMPLES_DIR, _theme
from ._delta_wing_plots import plot_circulation, plot_forces, plot_wake
from ._delta_wing_plots import load_accepted_lineage
from .finalize_delta_wing_lineage import finalize_lineage
from .render_delta_wing_gif import DEFAULT_OUTPUT, render


def plot_results(figure_format="png"):
    metadata_path = CASE_DIR / "solution/vpm_metadata.json"
    manifest = CASE_DIR / "assets/delta_wing_accepted_lineage.json"
    segments = None
    if manifest.is_file() and len(json.loads(manifest.read_text()).get("segments", [])) == 2:
        segments = load_accepted_lineage(manifest, include_active=True)
        metadata_path = segments[-1]["solution_path"] / "vpm_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    status = metadata["lifecycle"]["status"]
    complete = status == "completed"
    if complete:
        # Full-run products retain the native ownership and completion checks.
        finalize_lineage()
        samples = None
        destination = FIGURES_DIR
    else:
        samples = segments[-1]["samples_path"] if segments is not None else SAMPLES_DIR
        destination = FIGURES_DIR / "partial"
        state = metadata["state"]
        print(f"Run status: {status}; saved state step {state['step']}, t={state['time']:g} s.")
        print(
            "Writing partial diagnostics; cycle-mean wake and final animation require a completed run."
        )
    plot_forces(samples, destination, figure_format, partial=not complete)
    plot_circulation(samples, destination, figure_format, partial=not complete)
    plot_wake(
        samples,
        destination,
        figure_format,
        solution_dirs=[metadata_path.parent] if not complete else None,
        partial=not complete,
    )
    if complete:
        render(DEFAULT_OUTPUT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=_theme.EXPORT_FORMATS, default="png")
    args = parser.parse_args()
    plot_results(args.format)


if __name__ == "__main__":
    main()
