"""Plot recorded VPM meridional planes in the shared thesis style.

Reads SurfaceSampler VTS/PVD output only. No particle reconstruction, time
interpolation, or azimuthal averaging is performed by this plotting utility.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
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
from .study import STUDY_DIR
from .ring_metrics import _theme, load_metadata, load_study_metadata, metadata_settings


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


def discover(samples_dir, study_dir, runs=None):
    records = []
    indexes = list(samples_dir.glob("*/core_section.pvd"))
    indexes += list(study_dir.glob("*/samples/diagnostics/core_section.pvd"))
    for index in sorted(indexes):
        is_study = index.parent.name == "diagnostics"
        folder = index.parents[2] if is_study else index.parent
        name = folder.name
        if runs and name not in runs:
            continue
        metadata = load_study_metadata(folder) if is_study else load_metadata(name)
        settings = metadata_settings(metadata) if metadata else {}
        scenario = settings.get("scenario", "unknown")
        if scenario == "leapfrog" and settings.get("amplitude", 0.0) > 0:
            scenario = "seeded_breakdown"
        label = setup.CASE_LABELS.get(name, name.replace("_", " "))
        if name == "cs_breakdown_filter_cs020_cpu_t6_step080":
            label = r"Control $C_s=.20$"
        elif name == "cs_breakdown_filter_cs000_cpu_t6_step080":
            label = r"Molecular $C_s=0$"
        elif name == "cs_breakdown_filter_cs020_cpu_t6_step300_continuation":
            label = r"Control $C_s=.20$"
        elif name == "cs_breakdown_filter_cs000_cpu_t6_step300_continuation":
            label = r"Molecular $C_s=0$"
        elif name == "cs_breakdown_filter_cs020_cpu_t6_step300_tail":
            label = r"Control $C_s=.20$"
        elif name == "cs_breakdown_filter_cs020_cpu_t6_step320_discriminator":
            label = r"Control $C_s=.20$"
        elif name == "cs_breakdown_filter_cs000_cpu_t6_step320_discriminator":
            label = r"Molecular $C_s=0$"
        if settings:
            scheme = settings.get("diffusion", "CS")
            if scheme == "GBD":
                scheme += "/" + settings.get("gbd_remeshing", "M4_PRIME").replace(
                    "M4_PRIME", "M4'"
                ).replace("LAGRANGE6", "Lagrange-6")
            label = (
                f"{settings.get('method', 'baseline').replace('p_moments', 'weak realignment').replace('_', ' ').capitalize()}\n"
                f"{scheme}, {settings.get('integrator', 'SSPRK3')}\n"
                rf"$h/R_0={settings['spacing']:g}$, $\Delta t={settings['dt']:g}$"
            )
            if name == "cs_breakdown_coverage_h05_fixed_sigma_cpu_t6_continuation":
                label = label.replace("Baseline", "Fixed-core coverage")
            elif name == "cs_breakdown_filter_cs020_cpu_t6_step080":
                label = label.replace("Baseline", r"Control $C_s=.20$")
            elif name == "cs_breakdown_filter_cs000_cpu_t6_step080":
                label = label.replace("Baseline", r"Molecular $C_s=0$")
            elif name == "cs_breakdown_filter_cs020_cpu_t6_step300_continuation":
                label = label.replace("Baseline", r"Control $C_s=.20$")
            elif name == "cs_breakdown_filter_cs000_cpu_t6_step300_continuation":
                label = label.replace("Baseline", r"Molecular $C_s=0$")
            elif name == "cs_breakdown_filter_cs020_cpu_t6_step300_tail":
                label = label.replace("Baseline", r"Control $C_s=.20$")
            elif name == "cs_breakdown_filter_cs020_cpu_t6_step320_discriminator":
                label = label.replace("Baseline", r"Control $C_s=.20$")
            elif name == "cs_breakdown_filter_cs000_cpu_t6_step320_discriminator":
                label = label.replace("Baseline", r"Molecular $C_s=0$")
        for entry in ET.parse(index).findall(".//DataSet"):
            records.append(
                dict(
                    run=name,
                    label=label,
                    scenario=scenario,
                    reynolds_number=settings.get("reynolds_number", np.nan),
                    time=float(entry.attrib["timestep"]),
                    path=index.parent / entry.attrib["file"],
                )
            )
    if runs:
        run_order = {name: index for index, name in enumerate(runs)}
        records.sort(
            key=lambda record: (
                record["scenario"],
                record["time"],
                run_order.get(record["run"], len(run_order)),
            )
        )
    return records


def render(records, output, formats=("pdf", "png")):
    theme = _theme()
    theme.set_thesis_style()
    cmap = LinearSegmentedColormap.from_list(
        "openonda_signed_vorticity", [theme.PALETTE["purple"], "white", theme.PALETTE["teal"]]
    )
    groups = defaultdict(list)
    for record in records:
        groups[(record["scenario"], record["reynolds_number"], round(record["time"], 9))].append(
            record
        )
    output.mkdir(parents=True, exist_ok=True)
    for (scenario, reynolds_number, time), group in sorted(groups.items()):
        # Two panels per page retain the thesis font size and physical width.
        for start in range(0, len(group), 2):
            page = group[start : start + 2]
            fig, axes = plt.subplots(
                len(page),
                1,
                squeeze=False,
                figsize=theme.figure_size("stacked" if len(page) == 2 else "wide"),
            )
            fig.subplots_adjust(
                left=0.13,
                right=0.76,
                bottom=0.19,
                top=0.80 if len(page) == 2 else 0.72,
                hspace=1.1,
            )
            for ax, record in zip(axes.flat, page):
                x, r, omega = read_plane(record["path"])
                weights = np.abs(omega).sum(axis=1)
                centre = float(np.average(x, weights=weights)) if weights.sum() else float(x.mean())
                field = ax.contourf(
                    x, r, omega.T, levels=np.linspace(-1, 1, 81), cmap=cmap, extend="both"
                )
                positive = np.array([0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.8])
                levels = np.r_[-positive[::-1], positive]
                levels = levels[(levels > omega.min()) & (levels < omega.max())]
                if len(levels):
                    ax.contour(
                        x,
                        r,
                        omega.T,
                        levels=levels,
                        colors=theme.PALETTE["dark"],
                        linewidths=0.45,
                        negative_linestyles="dashed",
                    )
                ax.set(
                    xlim=(max(x[0], centre - 1.3), min(x[-1], centre + 1.3)),
                    ylim=(0, min(r[-1], 1.8)),
                    aspect="equal",
                    xlabel=r"$x/R_0$",
                    ylabel=r"$r/R_0$",
                )
                ax.set_title(record["label"], fontsize=theme.THESIS_FONT_SIZE_PT)
            cax = fig.add_axes([0.80, 0.23, 0.025, 0.48])
            fig.colorbar(
                field, cax=cax, ticks=[-1, -0.5, 0, 0.5, 1], label=r"$\omega_\theta/\omega_0$"
            )
            scenario_label = {
                "seeded_breakdown": "Seeded ring interaction",
                "leapfrog": "Unperturbed ring interaction",
                "collision": "Counter-rotating ring interaction",
                "unknown": "Ring interaction",
            }[scenario]
            fig.suptitle(
                rf"{scenario_label}, $Re_\Gamma={reynolds_number:g}$: $t\Gamma_0/R_0^2={time * setup.RING_CIRCULATION / setup.RING_RADIUS**2:.2f}$",
                y=0.98,
            )
            fig.text(
                0.15,
                0.015,
                r"$z=0$, $y\geq0$; $\omega=\nabla\times u$, $\omega_0=\Gamma_0/(\pi a_0^2)$",
                fontsize=9,
            )
            stem = f"core_sections_{scenario}_Re{reynolds_number:g}_t{time:g}_{start // 2 + 1}"
            for fmt in formats:
                fig.savefig(output / f"{stem}.{fmt}", dpi=theme.DEFAULT_DPI, bbox_inches=None)
            plt.close(fig)
            print(f"Saved {output / stem}", flush=True)
    if not records:
        print("No core_section.pvd samples found; new runs record these through setup.py.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", type=Path, default=setup.TUTORIAL_DIR / "samples")
    parser.add_argument("--study-dir", type=Path, default=STUDY_DIR)
    parser.add_argument("--output", type=Path, default=setup.TUTORIAL_DIR / "figures/core_sections")
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--times", nargs="+", type=float)
    parser.add_argument("--format", choices=("png", "pdf", "both"), default="both")
    args = parser.parse_args()
    records = discover(args.samples_dir, args.study_dir, args.runs)
    if args.times:
        records = [
            r
            for r in records
            if any(np.isclose(r["time"], t, atol=1e-8, rtol=0) for t in args.times)
        ]
    render(records, args.output, ("pdf", "png") if args.format == "both" else (args.format,))


if __name__ == "__main__":
    main()
