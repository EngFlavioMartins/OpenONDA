#!/usr/bin/env python3
"""Plot completed, source-recorded 3D cube experiments without fitted alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(root, name):
    path = root / name / "cube-coupled-trial.json"
    report = json.loads(path.read_text())
    if report["status"] != "complete" or report["spatial_dimensions"] != 3:
        raise ValueError(f"Incomplete or non-3D experiment: {path}")
    if not report["identical_native_shared_cells"]:
        raise ValueError(f"Unmatched FVM mesh: {path}")
    return report


def live_plot(root):
    cases = (
        ("cube-3d-coupled-mixed", "Mixed, ΔT=0.05", "#1f77b4", "-"),
        ("cube-3d-coupled-mixed-dt001", "Mixed, ΔT=0.01", "#1f77b4", "--"),
        ("cube-3d-coupled-mixed-pressure", "Mixed + VPM ∇p, ΔT=0.05", "#d95f02", "-"),
        ("cube-3d-coupled-mixed-pressure-dt001", "Mixed + VPM ∇p, ΔT=0.01", "#d95f02", "--"),
    )
    finer = root / "cube-3d-coupled-mixed-h00625/cube-coupled-trial.json"
    if finer.exists() and json.loads(finer.read_text())["status"] == "complete":
        cases += ((finer.parent.name, "Mixed, h=0.0625, ΔT=0.05", "#00856a", "-"),)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.subplots_adjust(bottom=0.2, top=0.86, hspace=0.4, wspace=0.27)
    for index, (name, label, color, style) in enumerate(cases):
        report = load(root, name)
        if report.get("frozen_renewals", 0):
            raise ValueError("A frozen transfer must not be plotted as flow evolution")
        rows = report["comparison"]
        t = np.array([r["physical_time"] for r in rows])
        for axis, key, factor, title in (
            (axes[0, 0], "fvm_velocity_rms_over_Uinf", 100, "Shared FVM volume: velocity RMS error / U∞ (%)"),
            (axes[0, 1], "fvm_near_body_velocity_rms_over_Uinf", 100, "Near cube: velocity RMS error / U∞ (%)"),
            (axes[1, 0], "hybrid_drag_coefficient", 1, "Drag coefficient (area = 1)"),
            (axes[1, 1], "vpm_sampled_velocity_rms_over_Uinf", 100, "VPM at 256 held-out cells: RMS error / U∞ (%)"),
        ):
            axis.plot(t, [factor * r[key] for r in rows], color=color, ls=style, label=label)
            axis.set(title=title, xlabel="Physical flow time t U∞/D")
            axis.grid(alpha=0.2)
        if index == 1:
            axes[1, 0].plot(t, [r["full_drag_coefficient"] for r in rows], "k:", label="Full FVM reference")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(
        "Live fully 3D cube coupling: identical native FVM cells, small 3D × 3D × 3D box\n"
        "Coarse wall cells (0.125); h=0.125 unless stated; Euler FVM dt=0.01"
    )
    fig.savefig(root / "cube-3d-live-comparison.png", dpi=180)
    plt.close(fig)


def frozen_plot(root):
    cases = (
        ("cube-3d-frozen-renewal", "Current renewal, prune 0.05", "#1f77b4", "-"),
        ("cube-3d-frozen-renewal-no-prune", "Current renewal, no pruning", "#1f77b4", "--"),
        ("cube-3d-frozen-renewal-residual", "Residual candidate, prune 0.05", "#a03c8c", "-"),
        ("cube-3d-frozen-renewal-residual-no-prune", "Residual candidate, no pruning", "#a03c8c", "--"),
        ("cube-3d-frozen-renewal-residual-no-prune-cap1", "Residual candidate, no second correction/pruning", "#a03c8c", ":"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    fig.subplots_adjust(bottom=0.27, top=0.78, wspace=0.27)
    for name, label, color, style in cases:
        report = load(root, name)
        rows = report["comparison"]
        if not all(r["elapsed_flow_time"] == 0 for r in rows):
            raise ValueError("Frozen comparison unexpectedly advanced time")
        iterations = [r["frozen_renewal_iteration"] for r in rows]
        axes[0].plot(iterations, [100 * r["vpm_sampled_velocity_rms_over_Uinf"] for r in rows], color=color, ls=style, label=label)
        axes[1].plot(iterations, [r["particle_strength_l1"] for r in rows], color=color, ls=style)
    axes[0].set(title="VPM velocity error at held-out native cells", ylabel="RMS error / U∞ (%)")
    axes[1].set(title="Total particle-strength magnitude", ylabel="Σ |Γp| (m³/s)")
    for axis in axes:
        axis.set(xlabel="Renewal applications at fixed physical time")
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(
        "Frozen 3D cube at t U∞/D = 0.5: transfer only\n"
        "No FVM advance, advection, stretching or diffusion; residual candidate rejected for default use"
    )
    fig.savefig(root / "cube-3d-frozen-renewal.png", dpi=180)
    plt.close(fig)


def medium_plot(root):
    cases = (
        ("cube-3d-medium-coupled-mixed", "Mixed", "#1f77b4"),
        ("cube-3d-medium-coupled-mixed-pressure", "Mixed + ∇p, accepted history", "#d95f02"),
        ("cube-3d-medium-pressure-predicted-history", "Mixed + ∇p, predictor history (experiment)", "#00856a"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.subplots_adjust(bottom=0.21, top=0.86, hspace=0.4, wspace=0.27)
    for index, (name, label, color) in enumerate(cases):
        rows = load(root, name)["comparison"]
        t = [r["physical_time"] for r in rows]
        for axis, key, factor, title in (
            (axes[0, 0], "fvm_velocity_rms_over_Uinf", 100, "Shared FVM volume: velocity RMS error / U∞ (%)"),
            (axes[0, 1], "fvm_near_body_velocity_rms_over_Uinf", 100, "Near cube: velocity RMS error / U∞ (%)"),
            (axes[1, 0], "hybrid_drag_coefficient", 1, "Drag coefficient (area = 1)"),
        ):
            axis.plot(t, [factor * r[key] for r in rows], color=color, label=label)
            axis.set(title=title, xlabel="Physical flow time t U∞/D")
        if index == 0:
            axes[1, 0].plot(t, [r["full_drag_coefficient"] for r in rows], "k:", label="Full FVM reference")
        if index > 0:
            valid = [r for r in rows if r.get("pressure_audit_temporal_valid")]
            axes[1, 1].plot([r["physical_time"] for r in valid], [r["pressure_predicted_normal_rms_error"] for r in valid], color=color)
            if index == 1:
                axes[1, 1].plot([r["physical_time"] for r in valid], [r["pressure_accepted_normal_rms_error"] for r in valid], color=color, ls="--", label="Post-transfer audit (not imposed on FVM)")
    axes[1, 1].set(title="Normal pressure-gradient RMS error (U∞²/D)", xlabel="Physical flow time t U∞/D")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    extra_handles, extra_labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles + extra_handles, labels + extra_labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(
        "Fully 3D medium cube: 16,936 identical near-body FVM cells in a 3D × 3D × 3D box\n"
        "Wall request and particle spacing = 0.0625; FVM dt=0.01; coupling ΔT=0.05"
    )
    fig.savefig(root / "cube-3d-medium-comparison.png", dpi=180)
    plt.close(fig)


def sgs_plot(root):
    cases = (
        ("cube-3d-medium-coupled-mixed", "LES, mixed", "#1f77b4", "-"),
        ("cube-3d-medium-laminar-coupled-mixed", "Laminar, mixed", "#1f77b4", "--"),
        ("cube-3d-medium-coupled-mixed-pressure", "LES, mixed + VPM ∇p", "#d95f02", "-"),
        ("cube-3d-medium-laminar-coupled-pressure", "Laminar, mixed + VPM ∇p", "#d95f02", "--"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.subplots_adjust(bottom=0.17, top=0.86, hspace=0.4, wspace=0.28)
    for name, label, color, style in cases:
        rows = load(root, name)["comparison"]
        t = [r["physical_time"] for r in rows]
        for axis, key, title in (
            (axes[0, 0], "fvm_velocity_rms_over_Uinf", "Shared FVM velocity RMS error / U∞ (%)"),
            (axes[0, 1], "fvm_near_body_velocity_rms_over_Uinf", "Near-body velocity RMS error / U∞ (%)"),
            (axes[1, 1], "vpm_sampled_velocity_rms_over_Uinf", "VPM sampled velocity RMS error / U∞ (%)"),
        ):
            axis.plot(t, [100 * r[key] for r in rows], color=color, ls=style, label=label)
            axis.set(title=title, xlabel="Physical flow time t U∞/D")
        axes[1, 0].plot(t, [100 * r["drag_coefficient_difference"] / r["full_drag_coefficient"]
                            for r in rows], color=color, ls=style)
    axes[1, 0].set(title="Drag difference from each case's full FVM reference (%)",
                   xlabel="Physical flow time t U∞/D")
    axes[1, 0].axhline(0, color="black", lw=0.6, alpha=0.5)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=10)
    fig.suptitle("Fully 3D medium cube: SGS removal leaves comparable coupling errors\n"
                 "Matched cells and time policy; independent fresh laminar and LES warmups; original convection assembly")
    fig.savefig(root / "cube-3d-sgs-isolation.png", dpi=180)
    plt.close(fig)


def mixed_convection_plot(root):
    cases = (
        ("cube-3d-medium-laminar-coupled-mixed", "Original explicit mixed-face convection", "#1f77b4", "--"),
        ("cube-3d-medium-laminar-implicit-convection-mixed", "Implicit tangential dependence", "#00856a", "-"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.subplots_adjust(bottom=0.16, top=0.86, hspace=0.4, wspace=0.28)
    reference, times = None, None
    for name, label, color, style in cases:
        rows = load(root, name)["comparison"]
        t = [r["physical_time"] for r in rows]
        drag = [r["full_drag_coefficient"] for r in rows]
        if reference is None:
            reference, times = drag, t
        else:
            np.testing.assert_allclose(t, times, rtol=0, atol=1e-14)
            np.testing.assert_allclose(drag, reference, rtol=0, atol=1e-12)
        for axis, key, scale, title in (
            (axes[0, 0], "fvm_velocity_rms_over_Uinf", 100, "Shared FVM velocity RMS error / U∞ (%)"),
            (axes[0, 1], "fvm_near_body_velocity_rms_over_Uinf", 100, "Near-body velocity RMS error / U∞ (%)"),
            (axes[1, 0], "hybrid_drag_coefficient", 1, "Drag coefficient"),
        ):
            axis.plot(t, [scale * r[key] for r in rows], color=color, ls=style, label=label)
            axis.set(title=title, xlabel="Physical flow time t U∞/D")
        axes[1, 1].plot(t, [100 * r["drag_coefficient_difference"] / r["full_drag_coefficient"]
                            for r in rows], color=color, ls=style)
    axes[1, 0].plot(times, reference, "k:", label="Full FVM reference")
    axes[1, 1].set(title="Relative drag difference from full FVM (%)",
                   xlabel="Physical flow time t U∞/D")
    axes[1, 1].axhline(0, color="black", lw=0.6, alpha=0.5)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1, frameon=False, fontsize=10)
    fig.suptitle("Mixed-boundary convection: matched fully 3D medium laminar cube\n"
                 "Same seed, cells, reference and transfer; h=0.0625; FVM dt=0.01; coupling ΔT=0.05")
    fig.savefig(root / "cube-3d-mixed-convection.png", dpi=180)
    plt.close(fig)


def joint_reconstruction_plot(root):
    study = json.loads((root / "cube-3d-joint-reconstruction-full-velocity-fit/joint-reconstruction-3d.json").read_text())
    boundary = json.loads((root / "cube-3d-native-curl-and-boundary/native-curl-3d.json").read_text())
    assert study["status"] == "complete" and study["spatial_dimensions"] == 3
    assert boundary["spatial_dimensions"] == 3
    rows = study["results"]
    other = {row["name"]: row for row in boundary["results"]}
    labels = ["ωV\nseed", "Original\nω fit", "Regularized\nω fit", "Raw ω\n+ velocity", "curl(u)\n+ velocity", "Velocity\nonly"]
    assert len(labels) == len(rows)
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.subplots_adjust(top=0.85, bottom=0.12, hspace=0.5, wspace=0.28)
    axes[0, 0].bar(x - 0.18, [100 * r["fit_velocity_rms_over_Uinf"] for r in rows], 0.36,
                   label="Fit cells", color="#a8b7c7")
    axes[0, 0].bar(x + 0.18, [100 * r["held_velocity_rms_over_Uinf"] for r in rows], 0.36,
                   label="Independent cells", color="#23588a")
    axes[0, 0].set(title="Velocity error: fitting does not guarantee reconstruction", ylabel="RMS / U∞ (%)")
    axes[0, 0].legend(frameon=False, fontsize=9)
    for offset, key, label, color in (
        (-0.26, "held_raw_vorticity_relative_error", "Gaussian sum", "#d98d45"),
        (0, "held_velocity_curl_relative_error", "Continuous curl", "#23588a"),
        (0.26, "held_native_curl_relative_error", "FVM curl stencil", "#27876e"),
    ):
        values = [r.get(key, other[r["name"]].get(key)) for r in rows]
        axes[0, 1].bar(x + offset, 100 * np.asarray(values), 0.26, label=label, color=color)
    axes[0, 1].set(title="Three different vorticity measurements", ylabel="Independent relative RMS error (%)")
    axes[0, 1].legend(frameon=False, fontsize=8.5)
    for axis, key, factor, title, units in (
        (axes[1, 0], "boundary_normal_velocity_rms_error_over_Uinf", 100,
         "Normal velocity supplied to the small FVM", "Boundary RMS / U∞ (%)"),
        (axes[1, 1], "boundary_tangential_normal_gradient_rms_error", 1,
         "Tangential normal gradient supplied to the small FVM", "Boundary RMS error (U∞ / D)"),
    ):
        axis.bar(x, [factor * other[r["name"]][key] for r in rows], color="#23588a", width=0.65)
        axis.axhline(factor * other[rows[0]["name"]][key], color="#666666", ls="--", lw=1, label="Seed")
        axis.set(title=title, ylabel=units)
        axis.legend(frameon=False, fontsize=9)
    for axis in axes.flat:
        axis.set_xticks(x, labels, fontsize=9)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    fig.suptitle("Frozen fully 3D cube: joint reconstruction remains unqualified\n"
                 "1,256 fit cells + 1,256 independent cells; h = 0.125; 68 outer particles fixed", fontsize=14)
    fig.text(0.5, 0.035,
             "Volume-weighted cell errors; area-weighted boundary errors. No time advance.\n"
             "All candidates use at most 2× the donor strength magnitude; original unregularized fit uses 2.62×.",
             ha="center", fontsize=10)
    fig.savefig(root / "cube-3d-joint-reconstruction.png", dpi=180)
    plt.close(fig)


def reconstruction_followup_plot(root):
    families = (
        ("cube-3d-native-reconstruction", ["Native\ncurl", "+ velocity", "+ wall", "+ moments"]),
        ("cube-3d-overlap-reconstruction", ["Overlap\nu\n+ moments", "Overlap\nω + u\n+ moments", "Overlap\nω + u"]),
        ("cube-3d-all-donor-reconstruction", ["All donor\nω", "+ velocity", "+ moments"]),
    )
    names, normal, gradient = [], [], []
    reference = None
    for folder, labels in families:
        source = json.loads((root / folder / "native-reconstruction-3d.json").read_text())
        audit = json.loads((root / (folder + "-boundary") / "native-curl-3d.json").read_text())
        assert source["status"] == "complete" and source["spatial_dimensions"] == 3
        assert audit["boundary_faces"] == 864
        seed = audit["results"][0]
        values = np.array([seed["boundary_normal_velocity_rms_error_over_Uinf"],
                           seed["boundary_tangential_normal_gradient_rms_error"]])
        if reference is None:
            reference = values
        else:
            np.testing.assert_allclose(values, reference, rtol=0, atol=1e-13)
        rows = audit["results"][2:]
        assert len(rows) == len(labels)
        names += labels
        normal += [row["boundary_normal_velocity_rms_error_over_Uinf"] / reference[0] for row in rows]
        gradient += [row["boundary_tangential_normal_gradient_rms_error"] / reference[1] for row in rows]
    x = np.arange(len(names))
    fig, axis = plt.subplots(figsize=(13, 6))
    fig.subplots_adjust(bottom=0.24, top=0.81)
    axis.axhspan(0, 1, color="#eef4ef", zorder=0)
    axis.bar(x - 0.19, normal, 0.38, label="Normal-velocity error", color="#23588a")
    axis.bar(x + 0.19, gradient, 0.38, label="Tangential normal-gradient error", color="#9d659c")
    axis.axhline(1, color="#555555", ls="--", lw=1.1, label="ωV seed")
    axis.set_xticks(x, names, fontsize=9)
    axis.set(ylabel="Boundary RMS error / corresponding seed error", ylim=(0, 2.3))
    for divider in (3.5, 6.5):
        axis.axvline(divider, color="#aaaaaa", lw=0.8)
    axis.set_axisbelow(True)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(loc="upper left", frameon=False, ncol=3, fontsize=9)
    fig.suptitle("Frozen 3D cube: no tested variant improves both boundary quantities\n"
                 "Same 864 coupling faces remain outside every fit; smaller is better", fontsize=14)
    fig.text(0.5, 0.055,
             "Fixed 2,138-source basis and 68 preserved outer particles; Σ|Γ| ≤ 2× donor value.\n"
             "Native curl, wall data, particle moments, overlap focus and all-donor sampling tested separately.",
             ha="center", fontsize=10)
    fig.savefig(root / "cube-3d-reconstruction-followup.png", dpi=180)
    plt.close(fig)


def cell_integral_plot(root):
    folders = ("cube-3d-cell-integral-point-control", "cube-3d-cell-integral-reconstruction")
    studies = [json.loads((root / name / "native-reconstruction-3d.json").read_text()) for name in folders]
    audits = [json.loads((root / (name + "-boundary") / "native-curl-3d.json").read_text()) for name in folders]
    assert [s["raw_observation"] for s in studies] == ["point", "cell_integral"]
    for s, a in zip(studies, audits, strict=True):
        assert s["status"] == "complete" and s["spatial_dimensions"] == 3
        assert s["fit_cells"] == 2512 and s["held_cells"] == 328 and a["boundary_faces"] == 864
    names = ["raw_vorticity", "raw_vorticity_velocity", "raw_vorticity_velocity_moments"]
    x = np.arange(len(names))
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8))
    fig.subplots_adjust(top=0.84, bottom=0.13, hspace=0.42, wspace=0.27)
    panels = (
        (studies, "fit_cell_integrated_raw_vorticity_relative_error", 100,
         "Cell circulation residual: fitted cells", "Relative RMS (%)", False),
        (studies, "held_velocity_rms_over_Uinf", 100,
         "Velocity error: unused outer-layer cells", "RMS / U∞ (%)", False),
        (audits, "boundary_normal_velocity_rms_error_over_Uinf", 1,
         "Normal velocity: independent boundary faces", "RMS error / seed error", True),
        (audits, "boundary_native_flux_tangential_normal_gradient_rms_error", 1,
         "Native tangential derivative: boundary faces", "RMS error / seed error", True),
    )
    for axis, (data, key, scale, title, ylabel, normalize) in zip(axes.flat, panels, strict=True):
        rows = [{r["name"]: r for r in s["results"]} for s in data]
        reference = rows[0]["donor_volume_vorticity"][key]
        np.testing.assert_allclose(rows[1]["donor_volume_vorticity"][key], reference, rtol=0, atol=1e-13)
        factor = 1 / reference if normalize else scale
        for i, (label, color) in enumerate((("Point fit", "#23588a"), ("Cell-integral fit", "#c57636"))):
            values = [rows[i][name][key] * factor for name in names]
            axis.bar(x + (i - 0.5) * 0.36, values, 0.36, color=color, label=label)
        axis.axhline(reference * factor, ls="--", color="#555555", lw=1.1, label="ωV seed")
        axis.set_xticks(x, ["ω", "ω + velocity", "ω + velocity\n+ moments"])
        axis.set(title=title, ylabel=ylabel)
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(loc="best", frameon=False, fontsize=9)
    fig.suptitle("Cell integrals versus point samples on the same 3D particle basis\n"
                 "2,512 training cells; 328 unused cells; 864 coupling faces excluded from fitting", fontsize=14)
    fig.text(0.5, 0.035, "Same particle positions, radii, outer particles, penalty and strength budget.\n"
             "The circulation residual integrates raw Gaussian vorticity; it does not integrate the velocity curl.",
             ha="center", fontsize=10)
    fig.savefig(root / "cube-3d-cell-integral-comparison.png", dpi=180)
    plt.close(fig)


def integrated_velocity_curl_plot(root):
    data = json.loads((root / "cube-3d-integrated-velocity-curl-comparison/integrated-velocity-curl-comparison.json").read_text())
    assert data["spatial_dimensions"] == 3 and data["held_cells"] == 328 and data["boundary_faces"] == 864
    families = (("point_raw", "Raw point", "#23588a"),
                ("integrated_raw", "Raw integral", "#c57636"),
                ("integrated_velocity_curl", "Velocity-curl integral", "#3b8367"))
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    fig.subplots_adjust(top=0.84, bottom=0.15, hspace=0.43, wspace=0.29)
    panels = (
        ("held_velocity_rms_over_Uinf", "Velocity error: unused cells", "RMS / U∞ (%)", 100, False),
        ("held_cell_integrated_velocity_curl_relative_error", "Integrated velocity-curl error: unused cells", "Error RMS / target RMS", 1, False),
        ("boundary_normal_velocity_rms_error_over_Uinf", "Normal velocity: independent boundary faces", "RMS error / seed error", 1, True),
        ("boundary_native_flux_tangential_normal_gradient_rms_error", "Native tangential derivative: boundary faces", "RMS error / seed error", 1, True),
    )
    x = np.arange(3)
    for axis, (key, title, ylabel, factor, normalize) in zip(axes.flat, panels, strict=True):
        baseline = None
        for index, (family, label, color) in enumerate(families):
            rows = [r for r in data["records"] if r["family"] == family]
            seed = next(r for r in rows if r["objective_index"] == -2)[key]
            if baseline is None:
                baseline = seed
            np.testing.assert_allclose(seed, baseline, rtol=0, atol=1e-13)
            scale = 1/baseline if normalize else factor
            values = [next(r for r in rows if r["objective_index"] == i)[key] * scale for i in range(3)]
            axis.bar(x+(index-1)*0.25, values, 0.25, color=color, label=label)
        axis.axhline(1 if normalize else baseline*factor, color="#555555", ls="--", lw=1.1, label="ωV seed")
        axis.set_xticks(x, ["Vorticity", "+ velocity", "+ velocity\n+ moments"])
        axis.set(title=title, ylabel=ylabel)
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925), frameon=False, ncol=4, fontsize=9)
    fig.suptitle("Three different vorticity observations on the same fully 3D particle basis", fontsize=14)
    fig.text(0.5, 0.04, "2,512 fitted cells; 328 unused cells; all 864 coupling faces excluded from fitting.\n"
             "Unused-cell target vorticity RMS is 0.0010693; strength budget and particle geometry are fixed.",
             ha="center", fontsize=10)
    fig.savefig(root / "cube-3d-integrated-velocity-curl-comparison.png", dpi=180)
    plt.close(fig)


def panel_resolution_plot(root):
    counts = (108, 432, 1728, 6912)
    studies = [json.loads((root / f"cube-3d-panel-resolution-{n}/cube-panel-resolution-3d.json").read_text()) for n in counts]
    for n, data in zip(counts, studies, strict=True):
        assert data["status"] == "complete" and data["spatial_dimensions"] == 3 and data["panels"] == n
        assert data["wall_samples"] == 1536
    names = (("donor_volume_vorticity", "ωV seed", "#23588a"),
             ("integrated_velocity_curl", "Curl fit", "#c57636"),
             ("integrated_velocity_curl_velocity", "Curl + velocity fit", "#3b8367"),
             ("integrated_velocity_curl_velocity_moments", "+ moments", "#9d659c"))
    panels = (
        ("independent_wall_normal_velocity_rms_over_Uinf", "Wall penetration: fresh common samples", "Normal-velocity RMS / U∞ (%)", 100),
        ("near_body_cell_velocity_rms_over_Uinf", "Near-body cell-velocity error", "Velocity RMS / U∞ (%)", 100),
        ("boundary_normal_velocity_rms_error_over_Uinf", "Normal velocity: coupling faces", "Error RMS / U∞ (%)", 100),
        ("boundary_native_flux_tangential_normal_gradient_rms_error", "Native tangential derivative: coupling faces", "Error RMS (U∞/D)", 1),
    )
    x = np.arange(len(counts))
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    fig.subplots_adjust(top=0.84, bottom=0.14, hspace=0.38, wspace=0.28)
    for axis, (key, title, ylabel, scale) in zip(axes.flat, panels, strict=True):
        for name, label, color in names:
            values = [next(r for r in data["results"] if r["name"] == name)[key]*scale for data in studies]
            axis.plot(x, values, "o-", label=label, color=color)
        axis.set_xticks(x, [str(n) for n in counts])
        axis.set(title=title, ylabel=ylabel, xlabel="Source panels")
        axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Refining the body panels with every particle state held fixed", fontsize=14)
    fig.text(0.5, 0.035, "Same 1,536 wall samples and 864 coupling faces at all four surface resolutions.\n"
             "The source-panel solve imposes normal velocity; the particle fits are not repeated.", ha="center", fontsize=10)
    fig.savefig(root / "cube-3d-panel-resolution.png", dpi=180)
    plt.close(fig)


def native_face_followup_plot(root):
    cases = (
        ("cube-3d-medium-laminar-implicit-convection-oracle", "Interpolated gradients", "#777777", "-"),
        ("cube-3d-medium-laminar-native-flux-initialized-oracle", "Native face derivatives", "#23588a", "-"),
        ("cube-3d-medium-laminar-native-gradient-lsq-oracle", "Native velocity derivative + LSQ ∇p", "#9d659c", "--"),
    )
    modes = ("vorticity_mixed", "vorticity_mixed_pressure_gradient")
    metrics = (
        ("velocity_rms_over_Uinf", "Shared volume: velocity RMS / U∞ (%)"),
        ("near_body_velocity_rms_over_Uinf", "Near cube: velocity RMS / U∞ (%)"),
        ("drag_coefficient_difference", "Drag error / full-FVM drag (%)"),
    )
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.15, top=0.87, hspace=0.35, wspace=0.3)
    reference, summary = None, []
    for folder, label, color, style in cases:
        report = json.loads((root / folder / "cube-boundary-oracle.json").read_text())
        assert report["spatial_dimensions"] == 3 and report["identical_native_shared_cells"]
        assert report["steps"] == 100 and report["small_cells"] == 16936
        assert report["numerics"]["sgs"] == "none"
        for column, mode in enumerate(modes):
            rows = [r for r in report["results"] if r["mode"] == mode]
            if not rows:
                continue
            t = np.array([r["physical_time"] for r in rows])
            full_drag = np.array([r["full_drag_coefficient"] for r in rows])
            if reference is None:
                reference = np.column_stack((t, full_drag))
            else:
                np.testing.assert_array_equal(np.column_stack((t, full_drag)), reference)
            summary.append({"source": folder, **rows[-1]})
            for row, (key, title) in enumerate(metrics):
                values = 100 * np.array([r[key] for r in rows])
                if row == 2:
                    values /= full_drag
                axes[row, column].plot(t, values, label=label, color=color, ls=style, lw=1.6)
                axes[row, column].set(ylabel=title)
                axes[row, column].grid(alpha=0.2)
    for column, title in enumerate(("Mixed velocity + flux-consistent pressure",
                                     "Mixed velocity + prescribed pressure gradient")):
        axes[0, column].set_title(title, fontsize=11)
        axes[2, column].set_xlabel("Physical time t U∞/D")
        axes[2, column].axhline(0, color="#999999", lw=0.7)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.055),
               frameon=False, ncol=3, fontsize=9)
    fig.suptitle("Fully 3D medium cube: native face traces do not remove the boundary error\n"
                 "16,936 identical small-domain cells; 53,752-cell full reference; laminar Re = 1000",
                 fontsize=14)
    fig.text(0.5, 0.018,
             "Reference-FVM traces only; no particles. Box ≈ 3D × 3D × 3D, wall spacing 0.0625D, Δt = 0.01D/U∞.\n"
             "Native face-value controls nearly overlap native derivatives and are tabulated in the report.",
             ha="center", fontsize=9)
    fig.savefig(root / "cube-3d-native-face-followup.png", dpi=180)
    plt.close(fig)
    (root / "native-face-followup-summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def outflow_convection_plot(root):
    cases = (
        ("cube-3d-medium-laminar-mixed-state-baseline", "Current mixed convection", "#777777", "native"),
        ("cube-3d-medium-laminar-mixed-outflow", "Experimental linearUpwind outflow", "#23588a", "outflow_linear_upwind"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.19, top=0.85, hspace=0.4, wspace=0.29)
    reference, summary = None, []
    for folder, label, color, policy in cases:
        report = load(root, folder)
        assert report["mixed_boundary_convection"] == policy
        assert report["fvm_numerics"]["sgs"] == "none"
        assert report["particle_spacing"] == 0.0625 and report["requested_coupling_steps"] == 20
        rows = report["comparison"]
        t = np.array([r["physical_time"] for r in rows])
        full_drag = np.array([r["full_drag_coefficient"] for r in rows])
        if reference is None:
            reference = np.column_stack((t, full_drag))
        else:
            np.testing.assert_array_equal(np.column_stack((t, full_drag)), reference)
        summary.append({"source": folder, **rows[-1]})
        for axis, key, title in (
            (axes[0, 0], "fvm_velocity_rms_over_Uinf", "Shared FVM volume: velocity RMS / U∞ (%)"),
            (axes[0, 1], "fvm_near_body_velocity_rms_over_Uinf", "Near cube: velocity RMS / U∞ (%)"),
            (axes[1, 0], "drag_coefficient_difference", "Drag difference / full-FVM drag (%)"),
            (axes[1, 1], "vpm_sampled_velocity_rms_over_Uinf", "VPM at independent cells: velocity RMS / U∞ (%)"),
        ):
            values = 100 * np.array([r[key] for r in rows])
            if key == "drag_coefficient_difference":
                values /= full_drag
            axis.plot(t, values, color=color, label=label, lw=1.6)
            axis.set(title=title, xlabel="Physical time t U∞/D")
            axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.075), frameon=False, ncol=2)
    fig.suptitle("Fully 3D live cube: testing the outflow convection change\n"
                 "Same 16,936 inherited FVM cells and initial particle field; laminar Re = 1000", fontsize=14)
    fig.text(0.5, 0.025,
             "Particle-supplied mixed velocity + flux-consistent pressure; the reference is observational only.\n"
             "h = σ = 0.0625D; FVM Δt = 0.01D/U∞; coupling ΔT = 0.05D/U∞; 20 coupling steps.",
             ha="center", fontsize=9)
    fig.savefig(root / "cube-3d-outflow-convection-live.png", dpi=180)
    plt.close(fig)
    (root / "outflow-convection-live-summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def volume_overlap_plot(root):
    report = json.loads((root / "cube-3d-volume-overlap/cube-volume-overlap-3d.json").read_text())
    if report["status"] != "complete" or report["spatial_dimensions"] != 3:
        raise ValueError("A completed fully 3D overlap qualification is required")
    base = json.loads((root / "cube-3d-native-volume-induction/cube-native-volume-induction-3d.json").read_text())
    seed = next(r for r in base["records"] if r["representation"] == "gaussian_seed_threshold_0.02" and r["completion"] == "panel108")
    rows = report["records"]
    labels = ["Gaussian\ncontrol", "Volume to\nFVM boundary", "Volume to\ninner 1.25", "Taper\n0.75–1.25", "Taper\n1.0–1.25"]
    colors = ["#757575", "#c97932", "#386cb0", "#228669", "#4aa58b"]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.9))
    fig.subplots_adjust(left=0.09, bottom=0.14, top=0.84, hspace=0.45, wspace=0.3)
    axes[0, 0].bar(x, [100*r["boundary_normal_velocity_rms_error_over_Uinf"] for r in rows], color=colors, width=0.65)
    axes[0, 0].axhline(100*seed["boundary_normal_velocity_rms_error_over_Uinf"], color="k", ls="--", lw=1, label="Original pruned seed")
    axes[0, 0].set(title="Normal-velocity improvement survives the overlap", ylabel="Boundary RMS error / U∞ (%)", ylim=(0, None))
    for side, label, color, marker in (("exterior_half", "Positive-normal one-sided estimate", "#c97932", "^"),
                                      ("centred_half", "Centred estimate", "#757575", "o"),
                                      ("interior_half", "Negative-normal one-sided estimate", "#386cb0", "v")):
        axes[0, 1].plot(x, [r["boundary_native_tangential_gradient_rms_error"][side] for r in rows],
                        color=color, marker=marker, label=label)
    axes[0, 1].axhline(seed["boundary_native_tangential_gradient_rms_error"], color="k", ls="--", lw=1)
    axes[0, 1].set(title="Derivative benefit depends on the sharp-interface trace", ylabel="Boundary RMS error / (U∞/D)", ylim=(0, None))
    axes[1, 0].bar(x, [100*r["cell_velocity_rms_over_Uinf"]["near_body"] for r in rows], color=colors, width=0.65)
    axes[1, 0].axhline(100*seed["cell_velocity_rms_over_Uinf"]["near_body"], color="k", ls="--", lw=1)
    axes[1, 0].set(title="Near-body velocity: 256 common native cells", ylabel="Velocity RMS error / U∞ (%)", ylim=(0, None))
    axes[1, 1].plot(x, [r["all_face_one_sided_tangential_gradient_jump_rms"] for r in rows], "o-", color="#9d659c")
    axes[1, 1].set(title="Removing volume support from the cut removes ambiguity", ylabel="RMS difference of derivative estimates (U∞/D)", yscale="log")
    for axis in axes.flat:
        axis.set_xticks(x, labels, fontsize=9)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    handles, names = axes[0, 1].get_legend_handles_labels()
    h, n = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles+h, names+n, loc="lower center", bbox_to_anchor=(0.5, 0.027), ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Fully 3D volume/particle overlap inside the same small FVM domain\n"
                 "Fixed cell circulations and 68 exterior particles; same 108 body panels; all 864 face derivatives checked at two step sizes")
    fig.text(0.5, 0.008, "Side labels refer to normal offsets from aggregate FVM face centres. The native-boundary jump is checked separately on actual fan triangles.",
             ha="center", fontsize=8)
    fig.savefig(root / "cube-3d-volume-overlap.png", dpi=180)
    plt.close(fig)


def partitioned_volume_plot(root):
    base = json.loads((root / "cube-3d-native-volume-induction/cube-native-volume-induction-3d.json").read_text())
    partition = json.loads((root / "cube-3d-partitioned-volume-induction/cube-partitioned-volume-induction-3d.json").read_text())
    for report in (base, partition):
        if report["status"] != "complete" or report["spatial_dimensions"] != 3:
            raise ValueError("Completed fully 3D induction results are required")
    lookup = {r["representation"]: r for report in (base, partition) for r in report["records"]
              if r["completion"] == "panel108"}
    states = (("gaussian_seed_threshold_0.02", "Original Gaussian seed", "#757575"),
              ("native_preserved_circulation", "Volume cells everywhere (diagnostic)", "#386cb0"),
              ("near_volume_outer_gaussian_all", "Small volume domain + all exterior Gaussians", "#4aa58b"),
              ("near_volume_outer_gaussian_seed", "Small volume domain + 68 exterior seed particles", "#228669"),
              ("near_gaussian_outer_volume_control", "Gaussians near body + volumes outside (control)", "#c97932"))
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.3))
    fig.subplots_adjust(left=0.1, bottom=0.2, top=0.85, hspace=0.37, wspace=0.3)
    metrics = (("near_body", "Near-body velocity: 256 native cells", "Velocity RMS error / U∞ (%)", 100),
               ("held_outer", "Outer layer: 328 unused native cells", "Velocity RMS error / U∞ (%)", 100),
               ("boundary_normal_velocity_rms_error_over_Uinf", "864 coupling faces: normal velocity", "RMS error / U∞ (%)", 100),
               ("boundary_native_tangential_gradient_rms_error", "864 coupling faces: centred native derivative", "RMS error / (U∞/D)", 1))
    for axis, (key, title, ylabel, factor) in zip(axes.flat, metrics, strict=True):
        for index, (name, label, color) in enumerate(states):
            row = lookup[name]
            value = row["cell_velocity_rms_over_Uinf"][key] if key in ("near_body", "held_outer") else row[key]
            axis.bar(index, factor*value, color=color, label=label, width=0.65)
            axis.text(index, factor*value, f"{factor*value:.3g}", ha="center", va="bottom", fontsize=9)
        axis.set(title=title, ylabel=ylabel, xticks=[], ylim=(0, None))
        axis.margins(y=0.15)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Sharp volume/particle partitions: velocity improves, but the derivative needs a trace convention\n"
                 "Frozen 3D cube at t = 0.5; 2,840 near-volume cells; same 108 body panels and measurement points")
    fig.text(0.5, 0.008, "The plotted derivative is centred. Exterior and interior traces differ for the sharp partition; see the separate overlap qualification.",
             ha="center", fontsize=9)
    fig.savefig(root / "cube-3d-partitioned-volume-induction.png", dpi=180)
    plt.close(fig)


def native_volume_plot(root):
    path = root / "cube-3d-native-volume-induction/cube-native-volume-induction-3d.json"
    report = json.loads(path.read_text())
    if report["status"] != "complete" or report["spatial_dimensions"] != 3:
        raise ValueError("A completed fully 3D volume-induction comparison is required")
    states = (("native_constant_omega", "Cell volumes\nconstant ω"),
              ("native_preserved_circulation", "Cell volumes\npreserve ωV"),
              ("gaussian_sigma_0.0625", "Gaussian\nσ = 0.0625"),
              ("gaussian_sigma_0.125", "Gaussian\nσ = 0.125"),
              ("gaussian_sigma_0.25", "Gaussian\nσ = 0.25"),
              ("gaussian_seed_threshold_0.02", "Old pruned seed\nσ = 0.125"))
    variants = (("freestream", "+ freestream", "#757575", "o"),
                ("outer_completion", "+ full outer-boundary term", "#386cb0", "s"),
                ("outer_and_native_divergence", "+ outer term and native divergence", "#c97932", "^"),
                ("panel108", "+ freestream and 108 body panels", "#228669", "D"))
    metrics = (("near_body", "Near-body velocity: 256 native cells", "Velocity RMS error / U∞ (%)"),
               ("held_outer", "Outer layer: 328 unused native cells", "Velocity RMS error / U∞ (%)"),
               ("boundary_normal_velocity_rms_error_over_Uinf", "Coupling faces: normal velocity", "Normal-velocity RMS error / U∞ (%)"),
               ("boundary_native_tangential_gradient_rms_error", "Coupling faces: centred native derivative", "Derivative RMS error / (U∞/D)"))
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
    fig.subplots_adjust(bottom=0.17, top=0.85, hspace=0.5, wspace=0.28)
    lookup = {(row["representation"], row["completion"]): row for row in report["records"]}
    for axis, (key, title, ylabel) in zip(axes.flat, metrics, strict=True):
        for index, (variant, label, color, marker) in enumerate(variants):
            values = []
            for state, _ in states:
                row = lookup[state, variant]
                value = row["cell_velocity_rms_over_Uinf"][key] if key in ("near_body", "held_outer") else row[key]
                values.append(value if key == "boundary_native_tangential_gradient_rms_error" else 100*value)
            axis.plot(np.arange(len(states))+(index-1.5)*0.12, values, linestyle="none", marker=marker,
                      markersize=5, label=label, color=color)
        axis.set_xticks(np.arange(len(states)), [label for _, label in states], fontsize=8)
        axis.set(title=title, ylabel=ylabel, ylim=(0, None))
        axis.grid(axis="y", alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.04), ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Frozen fully 3D cube: integrate native cells before replacing them by Gaussian particles\n"
                 "Same t = 0.5 velocity field; 17,592 full-domain source cells; no strength fitting or time advance")
    fig.text(0.5, 0.012, "Only the old-seed control prunes cells. Native divergence is a diagnostic; it is not an incompressible VPM correction.",
             ha="center", fontsize=9)
    fig.savefig(root / "cube-3d-native-volume-induction.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--plots", nargs="+", choices=("live", "frozen", "medium", "sgs", "mixed_convection", "joint_reconstruction", "reconstruction_followup", "cell_integral", "integrated_velocity_curl", "panel_resolution", "native_face_followup", "outflow_convection", "native_volume", "partitioned_volume", "volume_overlap"),
                        default=("live", "frozen", "medium"))
    args = parser.parse_args()
    for name in args.plots:
        globals()[name + "_plot"](args.results)
