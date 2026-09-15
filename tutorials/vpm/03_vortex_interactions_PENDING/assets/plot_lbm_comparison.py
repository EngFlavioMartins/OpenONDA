"""Score coherent field-core trajectories against the digitized LBM reference.

Identity is initialized by axial order in the first saved field and continued
by minimum total movement in the meridional plane. Stop at an unresolved pair
or a strong bridge; do not assign material identities to peaks after merger.
LBM scores require the initial field. No phase or distance fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
from .postprocess import (
    _theme,
    case_style,
    comparison_legend,
    figure_size,
    load_metadata,
    metadata_settings,
    plot_style_metadata,
    save_figure,
)
from .plot_core_sections import discover, read_plane


def sampled_peaks(x, r, omega, peak_merge_bridge=None):
    """Locate positive core maxima on a recorded SurfaceSampler grid.

    Coordinates are grid-resolved (no new field evaluation). A bridge is a
    linear interpolation of these same samples. Edge maxima invalidate tracking
    because the sampling domain may have clipped a core.
    Optional high-saddle grouping treats secondary lobes of a broad core as
    one peak. It does not smooth the field or bypass the separate pair cutoff.
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
    raw_count = len(indices)
    clusters = [[index] for index in indices]
    if peak_merge_bridge is not None:
        if not 0 < peak_merge_bridge < 1:
            raise ValueError("peak_merge_bridge must be between zero and one")
        interpolate = RegularGridInterpolator((x, r), omega)
        parent = list(range(raw_count))

        def root(index):
            while parent[index] != index:
                index = parent[index]
            return index

        # Attach each secondary maximum to at most one stronger maximum.
        # Unioning every high-saddle pair can join two distinct strong cores
        # through a weaker lobe even when their direct bridge is low.
        for child in range(1, raw_count):
            left = indices[child]
            child_value = float(omega[tuple(left)])
            best = None
            for stronger in range(child):
                right = indices[stronger]
                line = np.linspace([x[left[0]], r[left[1]]], [x[right[0]], r[right[1]]], 101)
                saddle = float(interpolate(line).min())
                if saddle < peak_merge_bridge * child_value:
                    continue
                distance = float(np.linalg.norm(line[-1] - line[0]))
                score = (saddle, -distance, -stronger)
                if best is None or score > best[0]:
                    best = (score, stronger)
            if best is not None:
                parent[child] = best[1]
        representatives = sorted({root(index) for index in range(raw_count)})
        clusters = [
            [indices[j] for j in range(raw_count) if root(j) == index] for index in representatives
        ]
        indices = [indices[index] for index in representatives]
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
            raw_n_peaks=raw_count,
            cluster_n_peaks=len(clusters[rank]),
            cluster_x_span=[
                float(min(x[a] for a, _ in clusters[rank])),
                float(max(x[a] for a, _ in clusters[rank])),
            ],
            cluster_radius_span=[
                float(min(r[b] for _, b in clusters[rank])),
                float(max(r[b] for _, b in clusters[rank])),
            ],
            strongest_peak_pair_bridge_ratio=bridge,
            peak_rank=rank,
        )
        for rank, (i, j) in enumerate(indices)
    ]


def sampler_history(run, peak_merge_bridge=None):
    records = discover(setup.TUTORIAL_DIR / "samples", [run])
    rows, sources = [], []
    for record in sorted(records, key=lambda entry: entry["time"]):
        path = record["path"]
        x, r, omega = read_plane(path)
        step = int(path.stem.rsplit("_", 1)[1])
        rows.extend(
            dict(run=run, time=record["time"], step=step, **peak)
            for peak in sampled_peaks(x, r, omega, peak_merge_bridge)
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
    """Return sampler output for one tutorial case."""
    return setup.TUTORIAL_DIR / "samples" / run


def coherent_tracks(peaks, bridge_limit=0.5, secondary_peak_limit=0.5):
    """Conservatively terminate identities when two distinct cores are lost.

    The bridge threshold is an explicit diagnostic cutoff, not a physical
    merger criterion. Weak secondary maxima may coexist with a dominant pair:
    the third maximum must be below ``secondary_peak_limit`` times the second.
    All maxima remain in the peaks table. Comparable competing cores, clipped
    fields and strong bridges still terminate identity. The caller must provide
    snapshots in a resolved cadence.
    """
    previous = None
    rows = []
    reason = "all supplied samples contain a separated pair"
    for step, group in peaks.sort_values("time").groupby("step", sort=False):
        if len(group) < 2 or (group.n_peaks < 2).any():
            reason = f"step {step}: fewer than two resolved peaks or clipped field"
            break
        if len(group) > 2:
            group = group.sort_values("vorticity", ascending=False)
            if group.iloc[2].vorticity >= secondary_peak_limit * group.iloc[1].vorticity:
                reason = (
                    f"step {step}: competing peak exceeds {secondary_peak_limit:g} of second core"
                )
                break
            group = group.iloc[:2]
        if group.strongest_peak_pair_bridge_ratio.max() >= bridge_limit:
            reason = f"step {step}: bridge ratio reached {bridge_limit:g}"
            break
        if previous is None:
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
        # Closely spaced outputs can locate a core in the same sampler cell.
        # Identical consecutive points add no information to R(x); retain
        # rejection of axial reversals and distinct radii at the same x.
        track = track.loc[~(track[["x", "radius"]].diff() == 0).all(axis=1)]
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


def leapfrog_events(tracks):
    """Bracket axial-order exchanges while two field cores remain identifiable.

    Interpolate x1-x2=0 in physical time; report the surrounding output times
    so the interpolation is not mistaken for time-resolved reference evidence.
    A zero plateau is one event only when the order actually reverses.
    """
    axial = tracks.pivot(index="time", columns="ring", values="x").sort_index()
    radial = tracks.pivot(index="time", columns="ring", values="radius").reindex(axial.index)
    if not {1, 2}.issubset(axial.columns) or axial.isna().any().any():
        raise ValueError("Passage timing needs both identified cores at common times")
    times = axial.index.to_numpy(dtype=float)
    difference = (axial[1] - axial[2]).to_numpy()
    nonzero = np.flatnonzero(difference != 0)
    events = []
    for left, right in zip(nonzero[:-1], nonzero[1:]):
        if difference[left] * difference[right] >= 0:
            continue
        fraction = difference[left] / (difference[left] - difference[right])
        time = times[left] + fraction * (times[right] - times[left])
        locations = [np.interp(time, times, axial[ring]) for ring in (1, 2)]
        radii = [np.interp(time, times, radial[ring]) for ring in (1, 2)]
        events.append(
            dict(
                time=float(time),
                time_bracket=[float(times[left]), float(times[right])],
                midpoint_x=float(np.mean(locations)),
                radial_separation=float(abs(radii[1] - radii[0])),
            )
        )
    return dict(
        passages=events,
        successive_passage_intervals=np.diff([e["time"] for e in events]).tolist(),
        full_cycle_periods=[
            events[i + 2]["time"] - events[i]["time"] for i in range(len(events) - 2)
        ],
        lbm_passage_times=None,
        lbm_temporal_phase_error=None,
        reference_limitation="The supplied LBM CSV has no time coordinate.",
    )


def plot_leapfrog_history(reports, output, formats=("png",)):
    """Show core positions and separation using the existing tracks."""
    if all(
        pd.read_csv(output / f"{report['run']}_tracks.csv").time.nunique() < 2 for report in reports
    ):
        for fmt in ("pdf", "png"):
            (output / f"leapfrogging_history.{fmt}").unlink(missing_ok=True)
        return
    theme = _theme()
    fig, axes = plt.subplots(3, 1, figsize=figure_size(10.5), sharex=True)
    for report in reports:
        tracks = pd.read_csv(output / f"{report['run']}_tracks.csv")
        style = case_style(report["run"])
        for ring, linestyle in ((1, "-"), (2, "--")):
            track = tracks[tracks.ring == ring]
            time = track.time * setup.RING_CIRCULATION / setup.RING_RADIUS**2
            line = dict(
                color=style["color"],
                marker=style["marker"],
                ms=3,
                markevery=max(1, len(track) // 12),
                linestyle=linestyle,
                lw=1,
            )
            axes[0].plot(time, track.x, label=style["label"] if ring == 1 else None, **line)
            axes[1].plot(time, track.radius, **line)
        axial = tracks.pivot(index="time", columns="ring", values="x")
        time = axial.index * setup.RING_CIRCULATION / setup.RING_RADIUS**2
        axes[2].plot(
            time,
            axial[1] - axial[2],
            color=style["color"],
            marker=style["marker"],
            ms=3,
            markevery=max(1, len(axial) // 12),
            lw=1,
        )
    for index, (ax, label) in enumerate(
        zip(axes, (r"$x/R_0$", r"$r/R_0$", r"$\Delta x/R_0$"), strict=True)
    ):
        ax.set_ylabel(label)
        ax.set_title(f"({chr(97 + index)})", pad=2)
    handles, labels = axes[0].get_legend_handles_labels()
    comparison_legend(fig, handles, labels)
    axes[2].set_xlabel(r"$t\Gamma_0/R_0^2$")
    theme.centered_subplots_adjust(
        fig, outer=0.19, bottom=0.135, top=0.86 if len(labels) <= 2 else 0.80, hspace=0.24
    )
    save_figure(fig, output / "leapfrogging_history", axes, formats)
    plt.close(fig)


def core_speeds(tracks):
    """Differentiate core positions with a centred five-frame quadratic fit.

    Fits use actual output times and omit two endpoint frames. A 0.02 R0
    sampler grid gives a coordinate quantization bound of 0.01 R0; propagating
    it through the fit bounds this part of the speed error only.
    """
    rows = []
    for ring, track in tracks.groupby("ring"):
        track = track.sort_values("time")
        for index in range(2, len(track) - 2):
            window = track.iloc[index - 2 : index + 3]
            time = float(track.iloc[index].time)
            offset = window.time.to_numpy() - time
            matrix = np.column_stack((np.ones(5), offset, offset**2))
            derivative = np.linalg.pinv(matrix)[1]
            speed = float(derivative @ window.x.to_numpy())
            rows.append(
                dict(
                    ring=int(ring),
                    time=time,
                    radius=float(track.iloc[index].radius),
                    axial_speed_R0_per_second=speed,
                    speed_grid_bound_R0_per_second=float(0.01 * np.abs(derivative).sum()),
                )
            )
    return pd.DataFrame(rows)


def plot_core_speeds(reports, output, formats=("png",)):
    """Show the contraction/acceleration part of leapfrogging explicitly."""
    theme = _theme()
    fig, ax = plt.subplots(figsize=figure_size(7.0))
    for report in reports:
        speed = core_speeds(pd.read_csv(output / f"{report['run']}_tracks.csv"))
        if speed.empty:
            continue
        speed.to_csv(output / f"{report['run']}_core_speeds.csv", index=False)
        style = case_style(report["run"])
        for ring, linestyle in ((1, "-"), (2, "--")):
            values = speed[speed.ring == ring]
            ax.plot(
                values.time * setup.RING_CIRCULATION / setup.RING_RADIUS**2,
                values.axial_speed_R0_per_second * setup.RING_RADIUS**2 / setup.RING_CIRCULATION,
                color=style["color"],
                marker=style["marker"],
                ms=3,
                markevery=max(1, len(values) // 12),
                lw=1,
                linestyle=linestyle,
                label=style["label"] if ring == 1 else None,
            )
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        comparison_legend(fig, handles, labels)
        ax.set(xlabel=r"$t\Gamma_0/R_0^2$", ylabel=r"$U_c R_0/\Gamma_0$")
        theme.centered_subplots_adjust(fig, outer=0.18, bottom=0.21, top=0.77)
        save_figure(fig, output / "core_speeds", ax, formats)
    else:
        for fmt in ("pdf", "png"):
            (output / f"core_speeds.{fmt}").unlink(missing_ok=True)
    plt.close(fig)


def temporal_field_comparisons(reports):
    """Compare identical sampled points at equal times for paired dt runs."""
    groups = {}
    for report in reports:
        configuration = {k: v for k, v in report["settings"].items() if k not in ("dt", "method")}
        groups.setdefault(json.dumps(configuration, sort_keys=True), []).append(report)
    comparisons = []
    for group in groups.values():
        ordered = sorted(group, key=lambda r: r["settings"]["dt"], reverse=True)
        for coarse, fine in zip(ordered, ordered[1:]):
            if coarse["settings"]["dt"] <= fine["settings"]["dt"]:
                continue
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


def plot_group_history(reports, output, formats=("png",)):
    """Plot native ancestry diagnostics independently of Eulerian peak identity."""
    sources = []
    valid = []
    for report in reports:
        run = report["run"]
        path = sample_directory(run) / "ring_diagnostics.csv"
        preserved = report["settings"]["regularization"].get(
            "regularization_preserve_groups", False
        )
        if not path.is_file() or not preserved:
            sources.append(
                dict(
                    run=run,
                    available=False,
                    reason="Native group history absent or remeshing used nearest-source labels",
                )
            )
            continue
        frame = pd.read_csv(path)
        sources.append(
            dict(
                run=run,
                available=True,
                file=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                end_time=float(frame.time.max()),
            )
        )
        valid.append((run, frame))
    if not valid:
        for fmt in ("pdf", "png"):
            (output / f"group_history.{fmt}").unlink(missing_ok=True)
        return sources
    theme = _theme()
    fig, axes = plt.subplots(2, 1, figsize=figure_size(8.3), sharex=True)
    for run, frame in valid:
        style = case_style(run)
        for group, linestyle in ((1, "-"), (0, "--")):
            track = frame[frame.group_id == group].sort_values("time")
            time = track.time * setup.RING_CIRCULATION / setup.RING_RADIUS**2
            line = dict(
                color=style["color"],
                marker=style["marker"],
                ms=3,
                markevery=max(1, len(track) // 12),
                linestyle=linestyle,
                lw=1,
            )
            axes[0].plot(
                time,
                track.vortex_centroid_x / setup.RING_RADIUS,
                label=style["label"] if group == 1 else None,
                **line,
            )
            axes[1].plot(time, track.major_radius / setup.RING_RADIUS, **line)
    axes[0].set_ylabel(r"$\bar{x}_g/R_0$")
    axes[1].set_ylabel(r"$R_g/R_0$")
    axes[1].set_xlabel(r"$t\Gamma_0/R_0^2$")
    handles, labels = axes[0].get_legend_handles_labels()
    comparison_legend(fig, handles, labels)
    theme.centered_subplots_adjust(
        fig, outer=0.19, bottom=0.17, top=0.89 if len(labels) <= 2 else 0.82, hspace=0.14
    )
    save_figure(fig, output / "group_history", axes, formats)
    plt.close(fig)
    return sources


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
        "selective_eddy_viscosity_feedback_coefficient",
    )
    return dict(
        file=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        final_sample_time=float(flow.time.iloc[-1]),
        final={key: float(flow[key].iloc[-1]) for key in columns},
        maximum={key: float(flow[key].max()) for key in columns},
        energy_measurements=(
            sorted(flow.energy_measurement.dropna().unique().tolist())
            if "energy_measurement" in flow
            else ["unknown"]
        ),
        final_energy_ratio=float(
            flow.total_kinetic_energy.iloc[-1] / flow.total_kinetic_energy.iloc[0]
        ),
    )


def plot_diagnostics(reports, output, formats=("png",)):
    """Plot saved native histories; do not reconstruct solver diagnostics."""
    theme = _theme()
    fig, axes = plt.subplots(3, 2, figsize=figure_size(12.0), sharex=True)
    quantities = (
        ("total_kinetic_energy", r"$E/E_0$", True),
        ("total_enstrophy", r"$Z/Z_0$", True),
        ("vorticity_divergence_error", "Divergence error", False),
        ("vortex_strength_misalignment_degrees", "Misalignment [deg]", False),
        ("lagrangian_cfl", "Lagrangian CFL", False),
        ("n_particles_total", "Particle count", False),
    )
    for report in reports:
        flow = pd.read_csv(sample_directory(report["run"]) / "flow_integrals.csv")
        style = case_style(report["run"])
        time = flow.time * setup.RING_CIRCULATION / setup.RING_RADIUS**2
        for index, (ax, (column, title, normalize)) in enumerate(
            zip(axes.flat, quantities, strict=True)
        ):
            values = flow[column] / flow[column].iloc[0] if normalize else flow[column]
            ax.plot(
                time,
                values,
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                ms=3,
                markevery=max(1, len(flow) // 12),
                lw=1,
            )
            ax.set_ylabel(title)
            ax.set_title(f"({chr(97 + index)})", pad=2)
    for ax in axes[-1]:
        ax.set_xlabel(r"$t\Gamma_0/R_0^2$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    comparison_legend(fig, handles, labels)
    theme.centered_subplots_adjust(fig, outer=0.18, bottom=0.12, top=0.83, hspace=0.24, wspace=0.70)
    save_figure(fig, output / "diagnostic_histories", axes.flat, formats)
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
    if report.get("unavailable_runs"):
        lines += [
            "",
            "Not plotted (no saved fields or metadata): "
            + ", ".join(report["unavailable_runs"])
            + ".",
        ]
    if any(result["identity_start_step"] != 0 for result in report["runs"]):
        lines += [
            "",
            "Runs without a step-zero field show kinematics from their first saved "
            "two-core field; their LBM identity scores are withheld.",
        ]
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
        f"Optional within-core saddle grouping: {report.get('peak_merge_bridge')}. "
        "The raw maximum counts and grouped-peak coordinate spans are retained in the peak tables.",
        "A third maximum below half the second core's amplitude does not invalidate the "
        "dominant pair. Comparable competing peaks and strong bridges still stop tracking.",
        "The trajectory figure ends each run when two coherent field tracks can no longer be followed. "
        "Track numbers come from initial axial order, not particle group_id ancestry. "
        "Later field maxima remain in the peak tables but do not extend the two-core scores.",
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
        "## VPM leapfrogging timing",
        "",
        "Axial-order reversals of two identified field cores, interpolated between "
        "saved planes. Brackets show the output cadence. These are not breakdown "
        "events. The LBM CSV has no timestamps, so LBM periods and temporal phase "
        "errors are unavailable.",
        "",
        "| Run | Passage times [s] | Successive intervals [s] | Two-passage cycle periods [s] |",
        "|---|---|---|---|",
    ]
    for result in report["runs"]:
        events = result["leapfrogging"]
        passages = (
            "; ".join(
                f"{event['time']:.4f} [{event['time_bracket'][0]:.3f}, {event['time_bracket'][1]:.3f}]"
                for event in events["passages"]
            )
            or "none resolved"
        )
        periods = (
            ", ".join(f"{period:.4f}" for period in events["full_cycle_periods"]) or "unavailable"
        )
        intervals = events["successive_passage_intervals"]
        interval_text = ", ".join(f"{value:.4f}" for value in intervals) or "unavailable"
        lines.append(f"| {result['run']} | {passages} | {interval_text} | {periods} |")
    lines += [
        "",
        "## Whole-solver timestep sensitivity",
        "",
        "Equal-time L2 differences on identical SurfaceSampler grids; "
        "these include time integration, diffusion and remapping effects.",
        "",
        "| Coarse run → fine run | Physical time | Velocity difference (%) | Vorticity difference (%) |",
        "|---|---:|---:|---:|",
    ]
    comparisons = report["temporal_field_comparisons"]
    for row in comparisons:
        if row["time"] > 0:
            lines.append(
                f"| {row['coarse']} → {row['fine']} | {row['time']:.4g} | "
                f"{100 * row['velocity_relative_l2']:.3f} | "
                f"{100 * row['vorticity_relative_l2']:.3f} |"
            )
    if not any(row["time"] > 0 for row in comparisons):
        lines += ["", "No comparable noninitial sampler times are available."]
    lines += [
        "",
        "## Tagged group histories",
        "",
        "The separate group_history figure uses native strength-weighted centroids and radii. "
        "These remain defined after merger when group contributions are preserved, but they "
        "are not maxima of the total vorticity field. Old nearest-source labels cannot recover "
        "ancestry retrospectively. Figure 4's material marker trajectories are a third, distinct observable.",
        "",
        *[
            f"- {row['run']}: "
            + (f"recorded through t={row['end_time']:.4g} s" if row["available"] else row["reason"])
            for row in report.get("group_diagnostics", [])
        ],
        "",
        "The accompanying lbm_agreement.json records solver settings, field hashes, "
        "health diagnostics, cadence sensitivity and the reference hash. Longer survival "
        "alone is not evidence of better physics.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))


def write_figure_manifest(output, reports, formats):
    """Keep the thesis-ready exports linked to their analysis and input runs."""
    analysis = output / "lbm_agreement.json"
    names = (
        "core_trajectories",
        "leapfrogging_history",
        "core_speeds",
        "diagnostic_histories",
        "group_history",
    )
    exports = []
    for name in names:
        for fmt in formats:
            path = output / f"{name}.{fmt}"
            if path.is_file():
                exports.append(
                    {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                )
    manifest = {
        "style": plot_style_metadata(),
        "generator": "assets/plot_lbm_comparison.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "analysis": "lbm_agreement.json",
        "analysis_sha256": hashlib.sha256(analysis.read_bytes()).hexdigest(),
        "runs": [
            {
                "name": report["run"],
                "status": report["status"],
                "metadata": f"solution/{report['recorded_run']}/vpm_metadata.json",
                "metadata_sha256": report["metadata_sha256"],
                "color": case_style(report["run"])["color"],
            }
            for report in reports
        ],
        "exports": exports,
    }
    (output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--interval", nargs=2, type=float, action="append")
    parser.add_argument("--bridge-limit", type=float, default=0.5)
    parser.add_argument(
        "--peak-merge-bridge",
        type=float,
        default=None,
        help="Group lobes joined above this fraction of the weaker peak; default keeps every maximum",
    )
    parser.add_argument(
        "--output", type=Path, default=setup.TUTORIAL_DIR / "figures/leapfrogging_study"
    )
    parser.add_argument("--format", choices=("png", "pdf", "both"), default="both")
    args = parser.parse_args()
    formats = ("pdf", "png") if args.format == "both" else (args.format,)
    if not 0 < args.bridge_limit < 1:
        parser.error("bridge-limit must be between zero and one")
    if args.peak_merge_bridge is not None and not args.bridge_limit < args.peak_merge_bridge < 1:
        parser.error("peak-merge-bridge must exceed bridge-limit and be below one")
    reference_path = Path(__file__).parent / "references/leapfrogging_lbm_trajectory.csv"
    reference = pd.read_csv(reference_path)
    saved = {record["run"] for record in discover(setup.TUTORIAL_DIR / "samples", args.runs)}
    runs = []
    for run in args.runs:
        if run not in saved:
            print(f"Skipping {run}: no saved core sections", flush=True)
        elif not load_metadata(run):
            print(f"Skipping {run}: no solver metadata", flush=True)
        else:
            runs.append(run)
    if not runs:
        print("No saved cases are ready to plot.", flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    theme = _theme()
    theme.set_thesis_style()
    fig, ax = plt.subplots(figsize=figure_size(7.0))
    for ring in (1, 2):
        ref = reference[reference.ring == ring]
        ax.plot(
            ref.x_over_R0 - 2.5,
            ref.R_over_R0,
            color=theme.COLORS["RefGray"],
            lw=1.0,
            linestyle="-" if ring == 1 else "--",
        )
    for run in runs:
        recorded_run = run
        metadata_path = setup.TUTORIAL_DIR / "solution" / recorded_run / "vpm_metadata.json"
        metadata_bytes = metadata_path.read_bytes()
        metadata = json.loads(metadata_bytes)
        settings = metadata_settings(metadata)
        numerics = metadata.get("configuration", {}).get("numerics", {})
        settings["filter_width"] = numerics.get("turbulence", {}).get("filter_width")
        # Closure and redistribution contrasts are not timestep refinements.
        settings["regularization"] = {
            key: value
            for key, value in settings["stabilization"].items()
            if key.startswith("regularization_")
        }
        if settings["scenario"] != "leapfrog":
            raise ValueError("The LBM reference is for leapfrogging only")
        peaks, sources = sampler_history(run, args.peak_merge_bridge)
        peak_path = args.output / f"{run}_peaks.csv"
        peaks.to_csv(peak_path, index=False)
        tracks, reason = coherent_tracks(peaks, args.bridge_limit)
        if tracks.empty:
            raise ValueError(f"No separated core pair in the saved fields for {run}")
        tracks.to_csv(args.output / f"{run}_tracks.csv", index=False)
        initial_step = int(peaks.step.min())
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
            if initial_step != 0:
                scores.append(
                    dict(
                        x_interval=interval,
                        unavailable="Initial field absent; core identities are not anchored to LBM",
                    )
                )
                continue
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
                recorded_run=recorded_run,
                metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
                status=metadata.get("lifecycle", {}).get("status", "unknown"),
                identity_start_step=initial_step,
                completed_steps=metadata.get("state", {}).get("step"),
                final_time=metadata.get("state", {}).get("time", float(peaks.time.max())),
                wall_seconds=float("nan"),
                settings=settings,
                reference_reynolds_number=3000,
                reference_seed_amplitude=0.0,
                reynolds_matches_reference=bool(np.isclose(settings["reynolds_number"], 3000)),
                seed_matches_reference=settings["amplitude"] == 0.0,
                coherent_pair_end_time=float(tracks.time.max()),
                tracked_x_extent_by_ring={
                    str(ring): float(group.x.max()) for ring, group in tracks.groupby("ring")
                },
                both_cores_reached_x7=bool((tracks.groupby("ring").x.max() >= 7.0).all()),
                identity_termination=reason,
                coarser_snapshot_identity_termination=coarse_reason,
                bridge_limit=args.bridge_limit,
                scores=scores,
                leapfrogging=leapfrog_events(tracks),
                peaks_sha256=hashlib.sha256(peak_path.read_bytes()).hexdigest(),
                sampler_fields=sources,
                self_diagnostics=reported_self_diagnostics(run),
                core_definition="positive curl(u)_z maxima on the z=0, y>=0 SurfaceSampler plane",
                spatial_uncertainty="grid-resolved peak coordinates; repeat with finer sampler spacing",
            )
        )
        style = case_style(run)
        for ring in (1, 2):
            track = tracks[tracks.ring == ring]
            ax.plot(
                track.x,
                track.radius,
                color=style["color"],
                marker=style["marker"],
                ms=3.5,
                markevery=max(1, len(track) // 12),
                lw=1.1,
                linestyle="-" if ring == 1 else "--",
            )
    ax.set(
        xlabel=r"$x/R_0$",
        ylabel=r"$R/R_0$",
        xlim=(-0.6, 7.4),
        ylim=(0.55, 1.48),
    )
    handles = [
        Line2D(
            [0],
            [0],
            color=case_style(run)["color"],
            marker=case_style(run)["marker"],
            ms=4,
            lw=1.1,
            label=case_style(run)["label"],
        )
        for run in runs
    ]
    handles.append(Line2D([0], [0], color=theme.COLORS["RefGray"], lw=1.0, label="LBM"))
    comparison_legend(fig, handles)
    theme.centered_subplots_adjust(
        fig, outer=0.18, bottom=0.19, top=0.75 if len(handles) <= 4 else 0.68
    )
    save_figure(fig, args.output / "core_trajectories", ax, formats)
    plt.close(fig)
    report = dict(
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference="Cheng, Lou & Lim (2015), Fig. 5(b), DOI:10.1063/1.4915890",
        reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        peak_merge_bridge=args.peak_merge_bridge,
        scoring="Equal-ring, uniform-x RMS radius discrepancy; linear interpolation; no extrapolation or fitting",
        limitation=(
            "Field-core kinematics before merger only; no group_id ancestry. "
            "Periodic boundaries and spatial convergence are not matched, so "
            "the discrepancy is not a benchmark error estimate."
        ),
        requested_runs=args.runs,
        unavailable_runs=[run for run in args.runs if run not in runs],
        runs=reports,
        temporal_field_comparisons=temporal_field_comparisons(reports),
    )
    report["group_diagnostics"] = plot_group_history(reports, args.output, formats)
    (args.output / "lbm_agreement.json").write_text(json.dumps(report, indent=2) + "\n")
    write_report(report, args.output)
    plot_diagnostics(reports, args.output, formats)
    plot_leapfrog_history(reports, args.output, formats)
    plot_core_speeds(reports, args.output, formats)
    write_figure_manifest(args.output, reports, formats)
    print(f"Saved LBM comparison for {len(reports)} run(s) to {args.output}", flush=True)


if __name__ == "__main__":
    main()
