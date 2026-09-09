"""Score coherent field-core trajectories against the digitized LBM reference.

Identity is initialized by axial order and continued by minimum total movement
in the meridional plane. Stop at an unresolved pair or a strong bridge; do not
assign material identities to peaks after merger. No phase or distance fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import maximum_filter
from scipy.interpolate import RegularGridInterpolator

if not __package__:
    from openonda.tutorial_runner import case_package
    from pathlib import Path as _CasePath

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

from .. import setup
from .study import STUDY_DIR
from .ring_metrics import _theme, load_metadata, load_study_metadata, metadata_settings
from .plot_core_sections import discover, read_plane


def sampled_peaks(x, r, omega):
    """Locate positive core maxima on a recorded SurfaceSampler grid.

    Coordinates are grid-resolved (no new field evaluation). A bridge is a
    linear interpolation of these same samples. Edge maxima invalidate tracking
    because the sampling domain may have clipped a core.
    """
    if not np.isfinite(omega).all():
        raise ValueError("Non-finite sampler field")
    maximum = float(np.max(omega))
    if maximum <= 0:
        return [
            dict(
                x=np.nan,
                radius=np.nan,
                vorticity=np.nan,
                n_peaks=0,
                strongest_peak_pair_bridge_ratio=np.nan,
                peak_rank=0,
            )
        ]
    indices = np.argwhere((omega == maximum_filter(omega, size=3)) & (omega > 0.1 * maximum))
    indices = sorted(indices, key=lambda ij: -omega[tuple(ij)])
    clipped = any(i in (0, len(x) - 1) or j in (0, len(r) - 1) for i, j in indices)
    bridge = np.nan
    if len(indices) >= 2:
        pair = np.array([[x[i], r[j]] for i, j in indices[:2]])
        values = RegularGridInterpolator((x, r), omega)(np.linspace(*pair, 101))
        bridge = float(values.min() / min(omega[tuple(ij)] for ij in indices[:2]))
    return [
        dict(
            x=float(x[i]),
            radius=float(r[j]),
            vorticity=float(omega[i, j]),
            n_peaks=-1 if clipped else len(indices),
            strongest_peak_pair_bridge_ratio=bridge,
            peak_rank=rank,
        )
        for rank, (i, j) in enumerate(indices)
    ]


def sampler_history(run):
    records = discover(setup.TUTORIAL_DIR / "samples", STUDY_DIR, [run])
    rows, sources = [], []
    for record in sorted(records, key=lambda entry: entry["time"]):
        path = record["path"]
        x, r, omega = read_plane(path)
        step = int(path.stem.rsplit("_", 1)[1])
        rows.extend(
            dict(run=run, time=record["time"], step=step, **peak)
            for peak in sampled_peaks(x, r, omega)
        )
        sources.append(
            dict(
                file=path.name,
                time=record["time"],
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    if not rows:
        raise ValueError(f"No SurfaceSampler core sections for {run}")
    return pd.DataFrame(rows), sources


def sample_directory(run):
    """Resolve official tutorial output first, then retained research output."""
    official = setup.TUTORIAL_DIR / "samples" / run
    return official if official.is_dir() else STUDY_DIR / run / "samples/diagnostics"


def coherent_tracks(peaks, bridge_limit=0.5):
    """Conservatively terminate identities when two distinct cores are lost.

    The bridge threshold is an explicit diagnostic cutoff, not a physical
    merger criterion. The caller must provide snapshots in a resolved cadence.
    """
    previous = None
    rows = []
    reason = "all supplied samples contain a separated pair"
    for step, group in peaks.sort_values("time").groupby("step", sort=False):
        if len(group) != 2 or not (group.n_peaks == 2).all():
            reason = f"step {step}: not exactly two resolved peaks"
            break
        if group.strongest_peak_pair_bridge_ratio.max() >= bridge_limit:
            reason = f"step {step}: bridge ratio reached {bridge_limit:g}"
            break
        if previous is None:
            if step != 0:
                raise ValueError("Core identity requires the initial step-zero field")
            group = group.sort_values("x", ascending=False)
        else:
            points = group[["x", "radius"]].to_numpy()
            distance = np.linalg.norm(previous[:, None, :] - points[None, :, :], axis=2)
            _, assignment = linear_sum_assignment(distance)
            group = group.iloc[assignment]
        previous = group[["x", "radius"]].to_numpy()
        for ring, (_, row) in enumerate(group.iterrows(), 1):
            rows.append({**row.to_dict(), "ring": ring})
    return pd.DataFrame(rows), reason


def radius_score(tracks, reference, xmin, xmax, samples=501):
    """Equal-ring RMS error on one uniform-x interval, without extrapolation."""
    if not np.isfinite([xmin, xmax]).all() or xmax <= xmin or samples < 2:
        raise ValueError("Require an increasing finite interval and at least two samples")
    x = np.linspace(xmin, xmax, samples)
    errors = []
    for ring in (1, 2):
        track = tracks[tracks.ring == ring].sort_values("time")
        ref = reference[reference.ring == ring].sort_values("x_over_R0")
        tx = track.x.to_numpy()
        rx = ref.x_over_R0.to_numpy() - 2.5
        if len(tx) < 2 or np.any(np.diff(tx) <= 0):
            raise ValueError("Radius-versus-distance scoring needs monotone axial tracks")
        if xmin < max(tx[0], rx[0]) or xmax > min(tx[-1], rx[-1]):
            raise ValueError("Both rings must cover the entire requested interval")
        errors.append(np.interp(x, tx, track.radius) - np.interp(x, rx, ref.R_over_R0))
    error = np.asarray(errors)
    return dict(
        x_interval=[xmin, xmax],
        samples_per_ring=samples,
        rms_radius_over_R0=float(np.sqrt(np.mean(error**2))),
        per_ring_rms_radius_over_R0=np.sqrt(np.mean(error**2, axis=1)).tolist(),
        max_radius_error_over_R0=float(np.max(np.abs(error))),
    )


def temporal_field_comparisons(reports):
    """Compare identical sampled points at equal times for paired dt runs."""
    groups = {}
    for report in reports:
        configuration = {k: v for k, v in report["settings"].items() if k != "dt"}
        groups.setdefault(json.dumps(configuration, sort_keys=True), []).append(report)
    comparisons = []
    for group in groups.values():
        ordered = sorted(group, key=lambda r: r["settings"]["dt"], reverse=True)
        for coarse, fine in zip(ordered, ordered[1:]):
            cfields = {round(f["time"], 10): f for f in coarse["sampler_fields"]}
            ffields = {round(f["time"], 10): f for f in fine["sampler_fields"]}
            for time in sorted(cfields.keys() & ffields.keys()):
                grids = [
                    pv.read(sample_directory(report["run"]) / fields[time]["file"])
                    for report, fields in ((coarse, cfields), (fine, ffields))
                ]
                if not np.array_equal(grids[0].points, grids[1].points):
                    raise ValueError("Temporal comparison needs identical sampler grids")
                row = dict(coarse=coarse["run"], fine=fine["run"], time=time)
                for field in ("velocity", "vorticity"):
                    a, b = [np.asarray(g.point_data[field], dtype=float) for g in grids]
                    denominator = np.linalg.norm(b)
                    row[field + "_relative_l2"] = (
                        float(np.linalg.norm(a - b) / denominator) if denominator else None
                    )
                    row[field + "_initial_bitwise_equal"] = (
                        bool(np.array_equal(a, b)) if time == 0 else None
                    )
                comparisons.append(row)
    return comparisons


def reported_self_diagnostics(run):
    """Summarize the solver's recorded diagnostics without reevaluating fields."""
    path = sample_directory(run) / "flow_integrals.csv"
    flow = pd.read_csv(path)
    columns = (
        "n_particles_total",
        "vorticity_divergence_error",
        "vortex_strength_misalignment_degrees",
        "lagrangian_cfl",
        "max_eddy_viscosity",
        "max_effective_viscosity",
        "n_stabilization_events",
        "n_regularization_events",
        "stretching_viscosity_feedback_coefficient",
    )
    return dict(
        file=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        final_sample_time=float(flow.time.iloc[-1]),
        final={key: float(flow[key].iloc[-1]) for key in columns},
        maximum={key: float(flow[key].max()) for key in columns},
        final_energy_ratio=float(
            flow.total_kinetic_energy.iloc[-1] / flow.total_kinetic_energy.iloc[0]
        ),
    )


def plot_diagnostics(reports, output):
    """Plot saved native histories; do not reconstruct solver diagnostics."""
    theme = _theme()
    fig, axes = plt.subplots(3, 2, figsize=(10, 11), sharex=True)
    quantities = (
        ("total_kinetic_energy", "Energy / initial", True),
        ("total_enstrophy", "Enstrophy / initial", True),
        ("vorticity_divergence_error", "Relative divergence", False),
        ("vortex_strength_misalignment_degrees", "Misalignment [degrees]", False),
        ("lagrangian_cfl", "Lagrangian CFL", False),
        ("n_particles_total", "Particle count", False),
    )
    for report in reports:
        flow = pd.read_csv(sample_directory(report["run"]) / "flow_integrals.csv")
        label = (
            report["settings"]["method"].replace("p_moments", "weak realignment").replace("_", " ")
        )
        for ax, (column, title, normalize) in zip(axes.flat, quantities, strict=True):
            values = flow[column] / flow[column].iloc[0] if normalize else flow[column]
            ax.plot(flow.time, values, label=label)
            ax.set_ylabel(title)
            ax.grid(alpha=0.15)
    for ax in axes[-1]:
        ax.set_xlabel("Physical time [s]")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "diagnostic_histories.png", dpi=theme.DEFAULT_DPI)
    plt.close(fig)


def write_report(report, output):
    """Write a readable result from the same sampler-derived score record."""
    lines = [
        "# Bounded VPM leapfrogging comparison",
        "",
        "These results use the VPM sampler outputs and recorded run statuses. "
        "They do not establish spatial convergence or matched periodic boundaries.",
        "",
        "| Run | Status | Last scalar time | Last field time | Wall minutes | Pair tracked until |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for result in report["runs"]:
        wall_minutes = (
            f"{result['wall_seconds'] / 60:.1f}"
            if np.isfinite(result["wall_seconds"])
            else "not recorded"
        )
        last_field_time = result["sampler_fields"][-1]["time"]
        lines.append(
            f"| {result['run']} | {result['status']} | "
            f"{result['self_diagnostics']['final_sample_time']:.4g} | {last_field_time:.4g} | "
            f"{wall_minutes} | {result['coherent_pair_end_time']:.4g} |"
        )
    lines += [
        "",
        "Pair tracking means two separated maxima on the recorded meridional "
        "plane. Its termination is a diagnostic cutoff, not proof of physical breakdown. "
        "A wall_time_limit status is a computational budget stop. "
        "A created status means the native metadata has not been finalized; "
        "it does not establish completion or explain termination. Saved sampler "
        "times remain usable independently of that lifecycle status. "
        "Completed and budget-limited runs provide lower bounds on numerical survival. "
        "A failed status needs its native vpm.log to distinguish numerical failure "
        "from output/resource failure; neither is automatically physical breakdown.",
        "",
        "## LBM radius discrepancies",
        "",
        "RMS errors below are percentages of R0 on fixed axial intervals. "
        "Compare methods on the same covered interval; unavailable intervals are not extrapolated.",
        "",
        "| Run | x/R0 interval | RMS radius error (% R0) |",
        "|---|---|---:|",
    ]
    for result in report["runs"]:
        for score in result["scores"]:
            error = (
                f"{100 * score['rms_radius_over_R0']:.3f}"
                if "rms_radius_over_R0" in score
                else "unavailable: " + score["unavailable"]
            )
            lines.append(f"| {result['run']} | {score['x_interval']} | {error} |")
    for result in report["runs"]:
        lines += [
            "",
            f"**{result['run']}**: {result['identity_termination']}. "
            f"Solver status: {result['status']}.",
            "Tracked axial coverage: "
            + ", ".join(
                f"core {ring}: x/R0={position:.3f}"
                for ring, position in result.get("tracked_x_extent_by_ring", {}).items()
            )
            + f". Both cores reached 7: {result.get('both_cores_reached_x7', 'not assessed')}.",
            "",
        ]
    lines += [
        "## Whole-solver timestep sensitivity",
        "",
        "Equal-time L2 differences on identical SurfaceSampler grids; "
        "these include time integration, diffusion and remapping effects.",
        "",
        "| Physical time | Velocity difference (%) | Vorticity difference (%) |",
        "|---:|---:|---:|",
    ]
    comparisons = report["temporal_field_comparisons"]
    for row in comparisons:
        if row["time"] > 0:
            lines.append(
                f"| {row['time']:.4g} | {100 * row['velocity_relative_l2']:.3f} | "
                f"{100 * row['vorticity_relative_l2']:.3f} |"
            )
    if not any(row["time"] > 0 for row in comparisons):
        lines += ["", "No comparable noninitial sampler times are available."]
    lines += [
        "",
        "The accompanying lbm_agreement.json records solver settings, field hashes, "
        "health diagnostics, cadence sensitivity and the reference hash. Longer survival "
        "alone is not evidence of better physics.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--interval", nargs=2, type=float, action="append")
    parser.add_argument("--bridge-limit", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=setup.TUTORIAL_DIR / "figures/study/les")
    args = parser.parse_args()
    if not 0 < args.bridge_limit < 1:
        parser.error("bridge-limit must be between zero and one")
    reference_path = Path(__file__).parent / "references/leapfrogging_lbm_trajectory.csv"
    reference = pd.read_csv(reference_path)
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    theme = _theme()
    theme.set_thesis_style()
    width, height = theme.figure_size("wide")
    fig, axes = plt.subplots(
        len(args.runs), 1, figsize=(width, height * len(args.runs)), squeeze=False
    )
    for ax, run in zip(axes.flat, args.runs, strict=True):
        folder = STUDY_DIR / run
        metadata = load_metadata(run) or load_study_metadata(folder)
        if not metadata:
            raise FileNotFoundError(folder / "solution/vpm_metadata.json")
        settings = metadata_settings(metadata)
        if settings["scenario"] != "leapfrog":
            raise ValueError("The LBM reference is for leapfrogging only")
        peaks, sources = sampler_history(run)
        peak_path = args.output / f"{run}_peaks.csv"
        peaks.to_csv(peak_path, index=False)
        tracks, reason = coherent_tracks(peaks, args.bridge_limit)
        if tracks.empty:
            raise ValueError(f"No separated initial cores for {run}")
        tracks.to_csv(args.output / f"{run}_tracks.csv", index=False)
        coarse_steps = np.unique(np.r_[peaks.step.unique()[::2], peaks.step.iloc[-1]])
        coarse_tracks, coarse_reason = coherent_tracks(
            peaks[peaks.step.isin(coarse_steps)], args.bridge_limit
        )
        scores = []
        for interval in args.interval or [
            [0.55, 1.5],
            [0.55, 2.5],
            [0.55, 3.5],
            [0.55, 5.5],
            [0.55, 7.0],
        ]:
            try:
                score = radius_score(tracks, reference, *interval)
            except ValueError as exc:
                scores.append(dict(x_interval=interval, unavailable=str(exc)))
                continue
            try:
                coarse = radius_score(coarse_tracks, reference, *interval)
                score["coarser_snapshot_rms_radius_over_R0"] = coarse["rms_radius_over_R0"]
            except ValueError as exc:
                score["coarser_snapshot_score_unavailable"] = str(exc)
            scores.append(score)
        reports.append(
            dict(
                run=run,
                status=metadata.get("lifecycle", {}).get("status", "unknown"),
                completed_steps=metadata.get("state", {}).get("step"),
                final_time=metadata.get("state", {}).get("time", float(peaks.time.max())),
                wall_seconds=float("nan"),
                settings=settings,
                coherent_pair_end_time=float(tracks.time.max()),
                tracked_x_extent_by_ring={
                    str(ring): float(group.x.max()) for ring, group in tracks.groupby("ring")
                },
                both_cores_reached_x7=bool((tracks.groupby("ring").x.max() >= 7.0).all()),
                identity_termination=reason,
                coarser_snapshot_identity_termination=coarse_reason,
                bridge_limit=args.bridge_limit,
                scores=scores,
                peaks_sha256=hashlib.sha256(peak_path.read_bytes()).hexdigest(),
                sampler_fields=sources,
                self_diagnostics=reported_self_diagnostics(run),
                core_definition="positive curl(u)_z maxima on the z=0, y>=0 SurfaceSampler plane",
                spatial_uncertainty="grid-resolved peak coordinates; repeat with finer sampler spacing",
            )
        )
        for ring, color in ((1, theme.PALETTE["purple"]), (2, theme.PALETTE["teal"])):
            ref = reference[reference.ring == ring]
            track = tracks[tracks.ring == ring]
            ax.plot(
                ref.x_over_R0 - 2.5,
                ref.R_over_R0,
                color=color,
                alpha=0.45,
                lw=1.5,
                label=f"LBM core {ring}",
            )
            ax.plot(track.x, track.radius, ".--", color=color, lw=1, ms=4, label=f"VPM core {ring}")
        sig = settings
        scheme = sig.get("diffusion", "CS")
        if scheme == "GBD":
            scheme += "/" + sig.get("gbd_remeshing", "M4_PRIME").replace("M4_PRIME", "M4'")
        ax.set(
            xlabel=r"$x/R_0$ (initial midpoint origin)",
            ylabel=r"Core radius, $R/R_0$",
            xlim=(-0.6, 7.4),
            ylim=(0.55, 1.48),
            title=(
                f"{scheme}, {sig.get('integrator', 'SSPRK3')}\n"
                f"{sig.get('method', 'baseline').replace('p_moments', 'Weak realignment').capitalize()}\n"
                rf"$h/R_0={sig['spacing']:g}$, $\Delta t={sig['dt']:g}$"
            ),
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)
        ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
    fig.text(
        0.15,
        0.015,
        r"$Re_\Gamma=3000$; Sampled plane-vorticity maxima."
        "\n"
        "Recorded samples; no phase or distance fitting.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(args.output / "core_trajectories.png", dpi=theme.DEFAULT_DPI)
    plt.close(fig)
    report = dict(
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference="Cheng, Lou & Lim (2015), Fig. 5(b), DOI:10.1063/1.4915890",
        reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        scoring="Equal-ring, uniform-x RMS radius discrepancy; linear interpolation; no extrapolation or fitting",
        limitation="A kinematic discrepancy, not a matched-boundary or converged benchmark error estimate",
        runs=reports,
        temporal_field_comparisons=temporal_field_comparisons(reports),
    )
    (args.output / "lbm_agreement.json").write_text(json.dumps(report, indent=2) + "\n")
    write_report(report, args.output)
    plot_diagnostics(reports, args.output)
    print(json.dumps([{k: r[k] for k in ("run", "status", "scores")} for r in reports], indent=2))


if __name__ == "__main__":
    main()
