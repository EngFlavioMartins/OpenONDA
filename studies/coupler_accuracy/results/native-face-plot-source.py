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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--plots", nargs="+", choices=("live", "frozen", "medium", "sgs", "mixed_convection", "joint_reconstruction", "reconstruction_followup", "native_face_followup", "outflow_convection"),
                        default=("live", "frozen", "medium"))
    args = parser.parse_args()
    for name in args.plots:
        globals()[name + "_plot"](args.results)
