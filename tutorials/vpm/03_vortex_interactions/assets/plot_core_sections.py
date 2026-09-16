"""Plot recorded VPM meridional planes in the shared thesis style.

Reads SurfaceSampler VTS/PVD output only. No particle reconstruction, time
interpolation, or azimuthal averaging is performed by this plotting utility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

if not __package__:
    from openonda.tutorial_runner import case_package
    from pathlib import Path as _CasePath

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

from .. import setup
from .postprocess import (
    CASES,
    case_style,
    discover_core_sections,
    figure_size,
    load_metadata,
    plot_style_metadata,
    read_core_section,
    save_figure,
    theme as plotting_theme,
)


def arrange_records(records, runs, times):
    """Arrange exact saved planes into a time-by-method grid.

    Missing run/time combinations remain ``None``. Two files claiming the same
    run and physical time are rejected because choosing one silently would make
    the comparison depend on directory ordering.
    """
    lookup = {}
    for record in records:
        key = (record["run"], record["time"])
        if key in lookup:
            raise ValueError(f"Duplicate core section for {key[0]} at t={key[1]:g}")
        lookup[key] = record
    return [
        [
            next(
                (
                    record
                    for (record_run, record_time), record in lookup.items()
                    if record_run == run and np.isclose(record_time, time, atol=1e-8, rtol=0)
                ),
                None,
            )
            for run in runs
        ]
        for time in times
    ]


def render(records, output, auxiliary_output, runs, times, formats=("png",)):
    theme = plotting_theme()
    theme.set_thesis_style()
    output.mkdir(parents=True, exist_ok=True)
    auxiliary_output.mkdir(parents=True, exist_ok=True)
    if not records or not runs or not times:
        print("No core_section.pvd samples found; new runs record these through setup.py.")
        return

    grid = arrange_records(records, runs, times)
    fig, axes = plt.subplots(
        len(times),
        len(runs),
        sharex=True,
        sharey=True,
        squeeze=False,
        figsize=figure_size(12.5),
    )
    fig.subplots_adjust(
        left=0.11,
        right=0.79,
        bottom=0.12,
        top=0.82,
        wspace=0.04,
        hspace=0.06,
    )
    cmap = LinearSegmentedColormap.from_list("core_vorticity", ["white", theme.COLORS["TUDdark"]])
    levels = np.linspace(0, 1, 41)
    contour_levels = np.array([0.05, 0.1, 0.2, 0.4, 0.8])
    panels = []
    field = None

    for column, run in enumerate(runs):
        label = case_style(run)["label"].replace(
            "Selective eddy viscosity", "Selective\neddy\nviscosity"
        )
        label = label.replace("Pedrizzetti ", "Pedrizzetti\n").replace(
            "Particle splitting", "Particle\nsplitting"
        )
        axes[0, column].set_title(label, color=case_style(run)["color"], pad=4)

    for row, (time, row_records) in enumerate(zip(times, grid, strict=True)):
        for column, (run, record) in enumerate(zip(runs, row_records, strict=True)):
            ax = axes[row, column]
            ax.set(
                xlim=(-1.0, 1.0),
                ylim=(0.35, 1.55),
                yticks=(0.6, 1.0, 1.4),
                aspect="equal",
            )
            ax.tick_params(
                bottom=row == len(times) - 1,
                labelbottom=row == len(times) - 1,
                left=column == 0,
                labelleft=column == 0,
            )
            panel = {"run": run, "requested_time": time, "available": record is not None}
            if record is None:
                ax.add_patch(
                    plt.Rectangle(
                        (0, 0),
                        1,
                        1,
                        transform=ax.transAxes,
                        facecolor=theme.COLORS["MaskGray"],
                        edgecolor="none",
                        clip_on=False,
                    )
                )
                ax.set_axis_off()
                panels.append(panel)
                continue

            x, r, omega = read_core_section(record["path"])
            weights = np.abs(omega).sum(axis=1)
            centre = float(np.average(x, weights=weights)) if weights.sum() else float(x.mean())
            local_x = x - centre
            field = ax.contourf(local_x, r, omega.T, levels=levels, cmap=cmap, extend="max")
            available_contours = contour_levels[
                (contour_levels > omega.min()) & (contour_levels < omega.max())
            ]
            if len(available_contours):
                ax.contour(
                    local_x,
                    r,
                    omega.T,
                    levels=available_contours,
                    colors=case_style(run)["color"],
                    linewidths=0.35,
                )

            source = record["path"]
            source_name = (
                str(source.relative_to(setup.TUTORIAL_DIR))
                if source.is_relative_to(setup.TUTORIAL_DIR)
                else str(source)
            )
            panel.update(
                {
                    "recorded_time": record["time"],
                    "recorded_run": run,
                    "run_status": load_metadata(run).get("lifecycle", {}).get("status"),
                    "axial_centre_over_radius": centre,
                    "source": source_name,
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
            )
            panels.append(panel)

    if field is None:
        plt.close(fig)
        print("No requested core-section times are available.")
        return

    fig.supxlabel(r"$(x-x_c)/R_0$", y=0.045)
    fig.supylabel(r"$r/R_0$", x=0.018)
    left_plot_edge = min(axis.get_position().x0 for axis in axes.flat)
    colorbar_width = 0.014
    cax = fig.add_axes([1.0 - left_plot_edge - colorbar_width, 0.31, colorbar_width, 0.38])
    fig.colorbar(field, cax=cax, ticks=[0, 0.5, 1], label=r"$\omega_\theta/\omega_0$")
    fig.canvas.draw()
    fig.text(
        0.83,
        0.835,
        r"$t\Gamma_0/R_0^2$",
        ha="center",
        va="bottom",
        color=theme.COLORS["DarkText"],
    )
    for row, time in enumerate(times):
        position = axes[row, -1].get_position()
        nondimensional_time = time * setup.RING_CIRCULATION / setup.RING_RADIUS**2
        fig.text(
            0.83,
            0.5 * (position.y0 + position.y1),
            rf"${nondimensional_time:.2g}$",
            ha="center",
            va="center",
            color=theme.COLORS["DarkText"],
        )
    stem = "core_sections"
    save_figure(fig, output / stem, (*axes.flat, cax), formats, fit_margins=False)
    exports = []
    for fmt in formats:
        exported = output / f"{stem}.{fmt}"
        exports.append(
            {"file": exported.name, "sha256": hashlib.sha256(exported.read_bytes()).hexdigest()}
        )
    plt.close(fig)
    print(f"Saved {output / stem}", flush=True)

    manifest = {
        "style": plot_style_metadata(),
        "generator": "assets/plot_core_sections.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "layout": {"rows": len(times), "columns": len(runs), "runs": runs, "times": times},
        "coordinate": "(x - vorticity-weighted section centre) / R0",
        "panels": panels,
        "exports": exports,
    }
    (auxiliary_output / "core_sections.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", type=Path, default=setup.TUTORIAL_DIR / "samples")
    parser.add_argument("--output", type=Path, default=setup.TUTORIAL_DIR / "figures")
    parser.add_argument(
        "--auxiliary-output", type=Path, default=setup.TUTORIAL_DIR / "figures/auxiliary"
    )
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--times", nargs="+", type=float)
    parser.add_argument(
        "--include-final",
        action="store_true",
        help="Include each selected run's actual final saved time",
    )
    parser.add_argument("--format", choices=("png", "pdf", "both"), default="png")
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove the case-local figures directory before rendering",
    )
    args = parser.parse_args()
    if args.clean_output:
        output = args.output.resolve()
        expected = (setup.TUTORIAL_DIR / "figures").resolve()
        if output != expected:
            parser.error("--clean-output is restricted to this case's figures directory")
        shutil.rmtree(output, ignore_errors=True)
    runs = args.runs or list(CASES)
    records = discover_core_sections(args.samples_dir, runs)
    times = sorted(set(args.times or (record["time"] for record in records)))
    if args.times:
        final_times = (
            {
                run: max(r["time"] for r in records if r["run"] == run)
                for run in {r["run"] for r in records}
            }
            if args.include_final
            else {}
        )
        times = sorted(set(times) | set(final_times.values()))
        records = [
            r
            for r in records
            if any(np.isclose(r["time"], t, atol=1e-8, rtol=0) for t in times)
            or r["time"] == final_times.get(r["run"])
        ]
    render(
        records,
        args.output,
        args.auxiliary_output,
        runs,
        times,
        ("pdf", "png") if args.format == "both" else (args.format,),
    )


if __name__ == "__main__":
    main()
