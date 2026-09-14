#!/usr/bin/env python3
"""Record comparison provenance and expose the raw fine-reference force anomaly."""

from __future__ import annotations

if not __package__:
    from pathlib import Path as _CasePath
    from openonda.tutorial_runner import case_package

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

import argparse
import csv
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
from . import _plotutil as util


def mesh_summary(path):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata"]))
        wall = next(b for b in meta["boundary"] if b["name"] == "cube")
        owners = archive["owners"][wall["start_face"] : wall["start_face"] + wall["n_faces"]]
        return {
            "cells": meta["n_cells"],
            "cube_faces": wall["n_faces"],
            "finest_cartesian_spacing": float(np.min(archive["cell_sizes"])),
            "cube_adjacent_cartesian_spacings": np.unique(archive["cell_sizes"][owners]).tolist(),
        }


def audit():
    limits = util.validate_plot_inputs()
    coupled, reference = util.comparison_configurations()
    end = min(limits.values())
    forces = util.load_forces("reference")
    selected = (forces["time"] >= 1) & (forces["time"] <= end + util.TIME_ATOL)
    indices = np.flatnonzero(selected)
    if not len(indices):
        indices = np.flatnonzero(forces["time"] <= end + util.TIME_ATOL)
    peaks = indices[np.argsort(forces["drag_coefficient"][indices])[-3:][::-1]]
    records = []
    for i in peaks:
        records.append(
            {
                "step": int(forces["step"][i]),
                "time": float(forces["time"][i]),
                "Cd": float(forces["drag_coefficient"][i]),
                "accepted_dt": float(forces["accepted_time_step_size"][i]),
            }
        )
    pressure = []
    peak_step = records[0]["step"]
    with (util.CASE_DIR / "reference_flow/solution/fine/diagnostics.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["step"] in (peak_step - 1, peak_step, peak_step + 1):
                pressure.append(
                    {
                        k: row[k]
                        for k in (
                            "step",
                            "time",
                            "time_step_size",
                            "min_kinematic_pressure",
                            "max_kinematic_pressure",
                            "max_velocity_magnitude",
                            "max_continuity_error",
                        )
                    }
                )
    meshes = {
        "coupled": mesh_summary(util.SOLUTION / "mesh.npz"),
        "fine": mesh_summary(util.CASE_DIR / "reference_flow/solution/fine/mesh.npz"),
    }
    if (
        meshes["coupled"]["cube_adjacent_cartesian_spacings"]
        != meshes["fine"]["cube_adjacent_cartesian_spacings"]
    ):
        raise ValueError("The cube-adjacent Cartesian spacings differ")
    before = next((row for row in pressure if row["step"] == peak_step - 1), None)
    at_peak = next((row for row in pressure if row["step"] == peak_step), None)
    ratios = None
    if before and at_peak:
        span_before = before["max_kinematic_pressure"] - before["min_kinematic_pressure"]
        span_peak = at_peak["max_kinematic_pressure"] - at_peak["min_kinematic_pressure"]
        if span_before > 0 and at_peak["time_step_size"] > 0:
            ratios = {
                "pressure_span_increase": span_peak / span_before,
                "timestep_reduction": before["time_step_size"] / at_peak["time_step_size"],
            }
    metrics_path = util.FIGURES / "field_differences.csv"
    field_metrics = []
    if metrics_path.is_file():
        with metrics_path.open() as stream:
            field_metrics = [
                row
                for row in csv.DictReader(stream)
                if np.isclose(float(row["time"]), end, rtol=0, atol=util.TIME_ATOL)
            ]
    return {
        "reference_samples": str(util.REFERENCE_SAMPLES),
        "reference_grid": "fine",
        "comparison_end_time": end,
        "meshes": meshes,
        "matched_configuration_sections": [
            "transport",
            "initial conditions",
            "schemes",
            "PIMPLE",
            "turbulence",
            "linear tolerances",
            "force normalization",
        ],
        "coupled_time_config": coupled["time"],
        "reference_time_config": reference["time"],
        "largest_reference_Cd_after_t1": records,
        "reference_pressure_around_largest_Cd": pressure,
        "ratios_at_largest_Cd": ratios,
        "latest_field_differences": field_metrics,
        "sampling": util.comparison_manifest()["method"],
        "definition": "100 * norm(u_test - u_comparison, 2) / U_inf; all three components",
        "rms": "sqrt(sum(area_weight * difference_percent**2) / sum(area_weight))",
        "support": "z=0, common finite fluid quads in [-1.5,1.5]^2; no extrapolation or body halo",
        "display": "bilinear velocity interpolation, then vector norm; full native maximum colour range",
    }


def plot_force_audit(report, fmt, dpi):
    util._THEME.set_thesis_style()
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=util.figure_size(11.8), dpi=dpi)
    util._THEME.centered_subplots_adjust(fig, outer=0.17, bottom=0.14, top=0.84, hspace=0.58)
    for source, style in (("reference", "-."), ("fvm", "-")):
        data = util.load_forces(source)
        keep = data["time"] <= report["comparison_end_time"] + util.TIME_ATOL
        axes[0].plot(
            data["time"][keep],
            data["drag_coefficient"][keep],
            color=util.colour(source),
            ls=style,
            label=util.label(source),
        )
        axes[1].semilogy(
            data["time"][keep],
            data["accepted_time_step_size"][keep],
            color=util.colour(source),
            ls=style,
        )
    axes[0].set(ylabel=r"$C_D$", title="(a) Raw wall drag")
    axes[0].yaxis.set_major_locator(MaxNLocator(4))
    axes[1].set(
        ylabel=r"$\Delta t$ [s]",
        xlabel="Flow time [s]",
        title="(b) Accepted timestep at force samples",
        xlim=(0, report["comparison_end_time"]),
    )
    axes[1].xaxis.set_major_locator(MaxNLocator(6))
    axes[1].xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    for ax in axes:
        ax.axvline(report["largest_reference_Cd_after_t1"][0]["time"], color=".4", lw=0.5, ls=":")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2, frameon=False
    )
    util.save(fig, "reference_force_audit", fmt, dpi)
    plt.close(fig)


def write_report(report):
    root = util.FIGURES
    root.mkdir(exist_ok=True)
    (root / "comparison_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    peak = report["largest_reference_Cd_after_t1"][0]
    mesh = report["meshes"]
    ratios = report["ratios_at_largest_Cd"]
    excursion = (
        ratios is not None
        and ratios["pressure_span_increase"] > 10
        and ratios["timestep_reduction"] > 10
    )
    observation = "Neighbouring pressure/timestep ratios are unavailable."
    if ratios:
        observation = (
            f"At this sample the pressure span increases by a factor of "
            f"{ratios['pressure_span_increase']:.3g}, and the timestep decreases "
            f"by a factor of {ratios['timestep_reduction']:.3g} relative to the preceding step."
        )
    verdict = (
        "The reference drag is not a trustworthy accuracy benchmark until this "
        "pressure/timestep excursion is resolved and checked."
        if excursion
        else "This diagnostic does not establish that the reference is converged or accurate."
    )
    latest_table = "No field metrics have been generated at this comparison time."
    if report["latest_field_differences"]:
        labels = {
            "reference_fvm_fields": "Reference FVM / Coupled FVM (primary near field)",
            "reference_vpm_fields": "Reference FVM / VPM (auxiliary overlap field)",
            "velocity_fields": "Coupled FVM / VPM (overlap consistency)",
        }
        latest_table = "| Comparison | RMS [% U_inf] | Sampled max [% U_inf] | Area [D^2] |\n"
        latest_table += "|---|---:|---:|---:|\n"
        for row in report["latest_field_differences"]:
            latest_table += (
                f"| {labels[row['figure']]} | {float(row['rms_percent']):.3f} | "
                f"{float(row['sampled_max_percent']):.3f} | {float(row['covered_area_D2']):.4f} |\n"
            )
    text = f"""# Cube comparison audit

Reference: reference_flow/samples/fine/ and reference_flow/solution/fine/.
Comparison ends at t={report["comparison_end_time"]:g} s. Reference data after
this time are not used in the figures. No simulation was advanced by plotting.

## What is matched

- Density, viscosity, initial conditions, FVM spatial/time schemes, turbulence
  closure, PIMPLE correctors/relaxation, linear tolerances and force definitions
  agree in the saved configurations.
- Both meshes have cube-adjacent Cartesian spacing
  {mesh["fine"]["finest_cartesian_spacing"]:.6g} m (requested fine target: 0.06 m).
  The coupled mesh has {mesh["coupled"]["cells"]:,} cells and
  {mesh["coupled"]["cube_faces"]:,} cube faces; the reference has
  {mesh["fine"]["cells"]:,} cells and {mesh["fine"]["cube_faces"]:,} cube faces.
  Their fitted wall cells and outer boundaries are not identical.
- Both FVM velocity fields use the existing 3D affine reconstruction with 12
  native volume-centroid neighbours and its documented IDW fallback. MPI
  global cell IDs are checked for complete, unique coverage.
- Only coincident saved physical states are used (absolute tolerance 1e-9 s).
  The reference saves full fields at 1 s intervals, so matched profile/field
  figures use that cadence. No interpolation between times is performed.
  The original force histories retain their 0.05 s sampling.
- Forces are the raw pressure-plus-viscous cube-wall forces, normalized by
  0.5 rho U_inf^2 D^2, with rho=1, U_inf=1 and D=1. No smoothing, outlier removal
  or drag-axis clipping is used.

## Meaning of the differences

Colour shows 100 ||u_test - u_comparison||_2 / U_inf, including u_x, u_y and u_z.
The velocity panels themselves show u_x/U_inf. Normalizing by freestream speed
avoids division by a vanishing local velocity. The word “difference” is used
because a finite-mesh reference is not an exact solution.

field_differences.csv records the area-weighted sampled RMS, sampled maximum,
valid-node count and covered area for each figure. Each valid grid rectangle
distributes one quarter of its area to each vertex. The RMS is a quadrature
estimate on this sampled z=0 plane, not a 3D volume norm. All three fields must
be finite at a point, giving every pairing the same support; the cube and
rectangles with missing corners are excluded.
Unresolved strips adjacent to the wall are not counted as zero error. Covered
area is reported so this limitation is visible.

Error contours use bilinearly interpolated velocity vectors; their difference
is formed before taking the vector norm. Metrics are calculated on the original
sample grid (spacing about 0.12 m), independently of display resolution.
Contours cannot recover structures absent from those samples. No nearest-point
extrapolation or 95th-percentile colour clipping is used. Colour ranges are
shared between the two velocity panels of each figure and may change with time.

reference_fvm_fields_* compares the primary near-body solution with fine FVM.
reference_vpm_fields_* and velocity_fields_* diagnose VPM in the overlap
region, where VPM is auxiliary. They are not whole-domain hybrid error maps.
The line profiles include the sampled outer wake. A z=0 section of 3D fields
does not establish accuracy everywhere in three dimensions.

At the latest compared time, t={report["comparison_end_time"]:g} s:

{latest_table}

These are instantaneous sampled differences, not time-averaged error estimates.

## Reference drag anomaly and remaining validation limits

The largest saved reference Cd after t=1 within the compared interval is
{peak["Cd"]:.9g} at t={peak["time"]:g} s, with accepted dt={peak["accepted_dt"]:.9g} s.
reference_force_audit.* shows raw Cd and the accepted timestep; the vertical
line marks this sample. comparison_audit.json records neighbouring pressure
extrema where solver diagnostics are available. These are observations, not
a completed diagnosis of the pressure/timestep algorithm.

{observation}
{verdict}
The diagnostic flags simultaneous pressure-span growth and timestep reduction
above a factor of ten; this is a screening rule, not a convergence criterion,
and it never filters the plotted data. Hiding suspect points would invalidate
the comparison. The fine run uses adaptive timesteps and the coupled FVM uses fixed
0.01 s steps, so temporal error is not isolated even though the schemes match.
The saved fine grid alone supplies no mesh/time convergence or statistical
uncertainty estimate. These figures can document the comparison and anomaly;
they do not yet support a claim of validated hybrid accuracy.

## Figure contract

Vector PDFs are exactly 125 mm wide, with embedded NewPX text/math fonts at
10.95 pt for main text. Include at natural size. PNG previews are 400 dpi.
All axes use equal outer side margins; the shared thesis validator checks font
sizes, a minimum 5 pt text-to-canvas clearance and text overlap before saving.
Line widths are 1.1 pt (primary), 1.0 pt (reference), and 0.5 pt (axes).
Captions belong in the thesis/paper; identify the slice, time, normalization,
common support, auxiliary-field status and finite-reference limitations.
"""
    (root / "comparison_audit.md").write_text(text)
    print(
        f"Reference drag check: Cd={peak['Cd']:.6g}, t={peak['time']:g}, dt={peak['accepted_dt']:.6g}"
    )
    print(verdict)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=util.EXPORT_FORMATS, default="pdf")
    parser.add_argument("--dpi", type=int, default=util.FIGURE_DPI)
    args = parser.parse_args()
    report = audit()
    write_report(report)
    plot_force_audit(report, args.format, args.dpi)


if __name__ == "__main__":
    main()
