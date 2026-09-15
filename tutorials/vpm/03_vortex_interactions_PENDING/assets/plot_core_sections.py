"""Plot recorded VPM meridional planes in the shared thesis style.

Reads SurfaceSampler VTS/PVD output only. No particle reconstruction, time
interpolation, or azimuthal averaging is performed by this plotting utility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import defusedxml.ElementTree as ET
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pyvista as pv

if not __package__:
    from openonda.tutorial_runner import case_package
    from pathlib import Path as _CasePath

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

from .. import setup
from .postprocess import (
    CASES,
    _theme,
    case_style,
    figure_size,
    load_metadata,
    plot_style_metadata,
    save_figure,
)


def read_plane(path):
    """Recover the rectangular z=0 half-plane independently of VTK ordering."""
    grid = pv.read(path)
    points = np.asarray(grid.points)
    if not np.allclose(points[:, 2], 0, atol=1e-7) or np.min(points[:, 1]) < -1e-7:
        raise ValueError(f"Expected z=0, y>=0 meridional half-plane: {path}")
    x, ix = np.unique(points[:, 0], return_inverse=True)
    r, ir = np.unique(points[:, 1], return_inverse=True)
    if len(points) != len(x) * len(r) or len(np.unique(ix * len(r) + ir)) != len(points):
        raise ValueError(f"Incomplete or duplicated rectangular plane: {path}")
    omega = np.empty((len(x), len(r)))
    omega[ix, ir] = np.asarray(grid.point_data["vorticity"])[:, 2]
    if not np.isfinite(omega).all():
        raise ValueError(f"Non-finite sampled vorticity: {path}")
    return (
        x / setup.RING_RADIUS,
        r / setup.RING_RADIUS,
        omega / (setup.RING_CIRCULATION / (np.pi * setup.CORE_RADIUS**2)),
    )


def discover(samples_dir, runs=None):
    records = []
    for index in sorted(samples_dir.glob("*/core_section.pvd")):
        recorded_name = index.parent.name
        name = recorded_name
        if name not in CASES or (runs and name not in runs):
            continue
        for entry in ET.parse(index).findall(".//DataSet"):
            path = index.parent / entry.attrib["file"]
            if not path.is_file():
                continue
            records.append(
                dict(
                    run=name,
                    recorded_run=recorded_name,
                    label=case_style(name)["label"],
                    time=float(entry.attrib["timestep"]),
                    path=path,
                )
            )
    if runs:
        run_order = {name: index for index, name in enumerate(runs)}
        records.sort(
            key=lambda record: (
                record["time"],
                run_order.get(record["run"], len(run_order)),
            )
        )
    return records


def render(records, output, formats=("png",)):
    theme = _theme()
    theme.set_thesis_style()
    output.mkdir(parents=True, exist_ok=True)
    exports = []
    for record in records:
        x, r, omega = read_plane(record["path"])
        weights = np.abs(omega).sum(axis=1)
        centre = float(np.average(x, weights=weights)) if weights.sum() else float(x.mean())
        style = case_style(record["run"])
        cmap = LinearSegmentedColormap.from_list(
            f"{record['run']}_vorticity", ["white", style["color"]]
        )
        levels = np.linspace(0, 1, 41)
        contours = np.array([0.05, 0.1, 0.2, 0.4, 0.8])
        contours = contours[(contours > omega.min()) & (contours < omega.max())]

        fig, ax = plt.subplots(figsize=figure_size(5.6))
        fig.subplots_adjust(left=0.16, right=0.78, bottom=0.235, top=0.885)
        field = ax.contourf(x, r, omega.T, levels=levels, cmap=cmap, extend="max")
        if len(contours):
            ax.contour(x, r, omega.T, levels=contours, colors=style["color"], linewidths=0.45)
        ax.set(
            xlim=(max(x[0], centre - 1.3), min(x[-1], centre + 1.3)),
            ylim=(0.35, min(r[-1], 1.55)),
            aspect="equal",
            xlabel=r"$x/R_0$",
            ylabel=r"$r/R_0$",
        )
        time = record["time"] * setup.RING_CIRCULATION / setup.RING_RADIUS**2
        ax.set_title(rf"{record['label'].splitlines()[0]}, $t\Gamma_0/R_0^2={time:.2f}$", pad=3)
        cax = fig.add_axes([0.82, 0.25, 0.02, 0.62])
        fig.colorbar(field, cax=cax, ticks=[0, 0.5, 1], label=r"$\omega_\theta/\omega_0$")
        # Choose horizontal margins from the rendered y text first. Preserve
        # equal physical x/y scale, then derive the compact canvas height.
        for _ in range(3):
            outer = theme.thesis_y_label_margin(fig, ax)
            plot_width_cm = (1 - 2 * outer - 0.06) * theme.MAX_FIGURE_WIDTH_CM
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            plot_height_cm = plot_width_cm * (ymax - ymin) / (xmax - xmin)
            height_cm = 1.35 + plot_height_cm + 0.70
            fig.set_size_inches(*figure_size(height_cm))
            bottom = 1.35 / height_cm
            height = plot_height_cm / height_cm
            ax.set_position([outer, bottom, 1 - 2 * outer - 0.06, height])
            cax.set_position([1 - outer - 0.02, bottom, 0.02, height])
        stem = f"core_section_{record['run']}_t{record['time']:g}"
        save_figure(fig, output / stem, (ax, cax), formats, fit_margins=False)
        source = record["path"]
        source_name = (
            str(source.relative_to(setup.TUTORIAL_DIR))
            if source.is_relative_to(setup.TUTORIAL_DIR)
            else str(source)
        )
        run_status = load_metadata(record["run"]).get("lifecycle", {}).get("status")
        for fmt in formats:
            exported = output / f"{stem}.{fmt}"
            exports.append(
                {
                    "file": exported.name,
                    "sha256": hashlib.sha256(exported.read_bytes()).hexdigest(),
                    "run": record["run"],
                    "recorded_run": record["recorded_run"],
                    "run_status": run_status,
                    "time": record["time"],
                    "source": source_name,
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
            )
        plt.close(fig)
        print(f"Saved {output / stem}", flush=True)
    if exports:
        manifest = {
            "style": plot_style_metadata(),
            "generator": "assets/plot_core_sections.py",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "exports": exports,
        }
        (output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not records:
        print("No core_section.pvd samples found; new runs record these through setup.py.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", type=Path, default=setup.TUTORIAL_DIR / "samples")
    parser.add_argument("--output", type=Path, default=setup.TUTORIAL_DIR / "figures/core_sections")
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--times", nargs="+", type=float)
    parser.add_argument(
        "--include-final",
        action="store_true",
        help="Include each selected run's actual final saved time",
    )
    parser.add_argument("--format", choices=("png", "pdf", "both"), default="both")
    args = parser.parse_args()
    records = discover(args.samples_dir, args.runs)
    if args.times:
        final_times = (
            {
                run: max(r["time"] for r in records if r["run"] == run)
                for run in {r["run"] for r in records}
            }
            if args.include_final
            else {}
        )
        records = [
            r
            for r in records
            if any(np.isclose(r["time"], t, atol=1e-8, rtol=0) for t in args.times)
            or r["time"] == final_times.get(r["run"])
        ]
    render(records, args.output, ("pdf", "png") if args.format == "both" else (args.format,))


if __name__ == "__main__":
    main()
