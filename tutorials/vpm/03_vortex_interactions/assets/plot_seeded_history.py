#!/usr/bin/env python3
"""Plot native histories from completed seeded-breakdown VPM cases.

This utility reads only native flow/ring CSVs, HDF5-derived mode metrics and
serialized lifecycle metadata. It does not reconstruct fields or score the
seeded Re=3415 cases against the unperturbed Re=3000 LBM trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GBD_QUALIFICATION = "gbd_breakdown_fmm_cpu_root_200000_qualification"
GBD_TO_200 = "gbd_breakdown_fmm_cpu_root_200000_from000100_to000200"
RUN_SEGMENTS = {
    GBD_TO_200: (GBD_QUALIFICATION, GBD_TO_200),
    "cs_breakdown_filter_cs020_cpu_t6_step300_tail": (
        "cs_breakdown_filter_cs020_cpu_t6_step080",
        "cs_breakdown_filter_cs020_cpu_t6_step300_continuation",
        "cs_breakdown_filter_cs020_cpu_t6_step300_tail",
    ),
    "cs_breakdown_filter_cs000_cpu_t6_step300_continuation": (
        "cs_breakdown_filter_cs000_cpu_t6_step080",
        "cs_breakdown_filter_cs000_cpu_t6_step300_continuation",
    ),
}
# The root operator verified native step 100 before the resource-stopped
# qualification was restarted. Its raw sampler tail through 120 is retained
# on disk but is outside this declared continuation prefix.
PREFIX_STEPS = {GBD_QUALIFICATION: 100}
# ParticleFieldEvaluation recreates its Fourier lattice/energy history on load.
# These are observer values, not the primitive particle sums checked below.
RESTART_GRID_FIELDS = frozenset(
    {
        "total_kinetic_energy",
        "total_enstrophy",
        "test_filtered_enstrophy",
        "total_helicity",
        "viscous_kinetic_energy_rate",
    }
)
RESTART_HEALTH_FIELDS = frozenset(
    {
        "strain_increment_infinity",
        "strain_increment_spectral",
        "maximum_particle_strength",
        "maximum_particle_vorticity",
    }
)
RESTART_F32_FIELDS = frozenset(
    {
        "max_eddy_viscosity",
        "max_effective_viscosity",
        "vortex_strength_misalignment_degrees",
    }
)
RESTART_F32_RTOL = 32 * 2**-23
ENERGY_RATE_SOURCES = frozenset(
    {
        "direct_energy_backward_difference",
        "direct_transition_viscous_rate",
        "fourier_energy_backward_difference",
        "free_space_fft_energy_backward_difference",
        "periodic_fourier_energy_backward_difference",
        "fourier_transition_viscous_rate",
        "fourier_grid_transition_backward_difference",
    }
)
RESTART_RATE_SOURCES = ENERGY_RATE_SOURCES - {"fourier_grid_transition_backward_difference"}
FLOW_STATE_FIELDS = frozenset(
    {
        "n_particles_total",
        "vortex_strength_magnitude_sum",
        "max_vortex_strength_magnitude",
        "min_particle_core_radius",
        "mean_particle_core_radius",
        "max_particle_core_radius",
        *(
            f"{quantity}_{axis}"
            for quantity in (
                "net_vortex_strength",
                "linear_impulse",
                "angular_impulse",
            )
            for axis in "xyz"
        ),
    }
)


def _load_plotting() -> None:
    """Keep validation and CLI help usable without the plotting stack."""
    global plt, pd, Line2D, theme
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.lines import Line2D
    from openonda import plotting as theme


def _label(run: str) -> str:
    if run == "cs_breakdown_splitting_fmm_cpu_root_200000":
        return "CS + filament splitting"
    if run.startswith("gbd_breakdown_fmm_cpu_root_200000"):
        return "GBD"
    if run == "cs_breakdown_filter_cs020_cpu_t6_step080":
        return r"Control $C_s=.20$"
    if run == "cs_breakdown_filter_cs000_cpu_t6_step080":
        return r"Molecular $C_s=0$"
    if run == "cs_breakdown_filter_cs020_cpu_t6_step300_continuation":
        return r"Control $C_s=.20$"
    if run == "cs_breakdown_filter_cs000_cpu_t6_step300_continuation":
        return r"Molecular $C_s=0$"
    if run == "cs_breakdown_filter_cs020_cpu_t6_step300_tail":
        return r"Control $C_s=.20$"
    if run == "cs_breakdown_coverage_h05_fixed_sigma_cpu_t6_continuation":
        return r"Coverage $h=.05$, $\sigma_0=.06$"
    if run == "cs_breakdown_p_moments_cpu_t6_qualification":
        return "Moment-preserving realignment"
    return run.removeprefix("cs_breakdown_").replace("_", " ").title()


def _run_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", value):
        raise ValueError(f"Invalid run name: {value!r}")
    return value


def _chains(declarations: list[str], selected: list[str]) -> dict[str, tuple[str, ...]]:
    result = dict(RUN_SEGMENTS)
    declared = set()
    for declaration in declarations:
        alias, separator, names = declaration.partition("=")
        segments = tuple(_run_name(name) for name in names.split(","))
        if not separator or alias not in selected or alias in declared or len(segments) < 2:
            raise ValueError("Each --chain must be SELECTED_RUN=PREFIX,CONTINUATION[,TAIL]")
        if len(set(segments)) != len(segments) or segments[-1] != alias:
            raise ValueError("A chain must contain distinct segments and end with its selected run")
        declared.add(alias)
        result[alias] = segments
    return result


def _integer(value, context: str, *, minimum: int = 0) -> int:
    number = float(value)
    if not math.isfinite(number) or not number.is_integer() or number < minimum:
        raise ValueError(f"{context}: expected an integer >= {minimum}, got {value!r}")
    return int(number)


def _prefix_steps(declarations: list[str], segments: set[str]) -> dict[str, int]:
    result = dict(PREFIX_STEPS)
    declared = set()
    for declaration in declarations:
        run, separator, step = declaration.partition("=")
        if not separator or run not in segments or run in declared:
            raise ValueError("Each --prefix-through must be SELECTED_SEGMENT=STEP")
        result[run] = _integer(step, "prefix step")
        declared.add(run)
    return result


def _clock(metadata: dict) -> tuple[int, int, float, float]:
    state = metadata["state"]
    start = _integer(state.get("initial_step", 0), "initial_step")
    end = _integer(state["step"], "step")
    initial_time = float(state.get("initial_time", 0.0))
    end_time = float(state["time"])
    dt = float(metadata["configuration"]["numerics"]["time_step_size"])
    if end < start or dt <= 0 or not all(map(math.isfinite, (initial_time, end_time, dt))):
        raise ValueError("Invalid accepted metadata clock")
    if not math.isclose(end_time, initial_time + (end - start) * dt, rel_tol=0, abs_tol=1e-10):
        raise ValueError("Metadata endpoint does not match its accepted step/dt clock")
    return start, end, initial_time, dt


def _validate_segments(metadata: list[dict]) -> None:
    identity = None
    previous = None
    for item in metadata:
        configuration = item["configuration"]
        current = {
            key: configuration[key]
            for key in (
                "numerics",
                "initial_conditions",
                "initial_weak_particle_percent",
            )
        }
        if identity is not None and current != identity:
            raise ValueError("Restart segments disagree on numerical/physical configuration")
        start, end, initial_time, dt = _clock(item)
        if previous is not None:
            prior_end, prior_time = previous
            if start != prior_end or not math.isclose(
                initial_time, prior_time, rel_tol=0, abs_tol=1e-10
            ):
                raise ValueError("Restart segments do not meet at the same accepted step/time")
        identity = current
        previous = end, initial_time + (end - start) * dt


def _bounded_metadata(metadata: dict, through: int | None) -> dict:
    start, end, initial_time, dt = _clock(metadata)
    if through is None:
        return metadata
    if not start <= through <= end:
        raise ValueError("Declared prefix step lies outside the recorded accepted clock")
    return {
        **metadata,
        "plot_recorded_state": dict(metadata["state"]),
        "state": {
            **metadata["state"],
            "step": through,
            "time": initial_time + (through - start) * dt,
        },
    }


def _value(value: str):
    if value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return value


def _same_value(left, right) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        # Missing mode components (e.g. a field-only metric on a particle row)
        # are legitimate. A missing/finite mismatch is not.
        if math.isnan(left) or math.isnan(right):
            return math.isnan(left) and math.isnan(right)
        return math.isclose(left, right, rel_tol=5e-7, abs_tol=1e-12)
    return left == right


def _finite(value) -> bool:
    return isinstance(value, int | float) and math.isfinite(value)


def _missing(value) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _restart_observer_difference(column: str, prefix: dict, restart: dict) -> str | None:
    """Allow only source-defined observer resets at a declared flow boundary."""
    left, right = prefix.get(column, math.nan), restart.get(column, math.nan)
    if column in RESTART_HEALTH_FIELDS:
        # load_backup clears _accepted_health_snapshot. A finite replacement
        # is still a measurement of the same state and must agree.
        if _missing(right) and (_finite(left) or _missing(left)):
            return "accepted_health_unavailable"
        return None
    if column in RESTART_F32_FIELDS:
        if (
            _finite(left)
            and _finite(right)
            and math.isclose(left, right, rel_tol=RESTART_F32_RTOL, abs_tol=1e-12)
        ):
            return "recomputed_gradient_f32_roundoff"
        return None
    source = restart.get("kinetic_energy_rate_source")
    if column in RESTART_GRID_FIELDS:
        allowed = (
            _finite(left)
            and _finite(right)
            and prefix.get("energy_measurement")
            == restart.get("energy_measurement")
            == "periodic_fourier_energy"
            and source == "fourier_transition_viscous_rate"
        )
        return "periodic_fourier_grid_reinitialized" if allowed else None
    if column == "kinetic_energy_rate":
        if _finite(left) and _finite(right) and source in RESTART_RATE_SOURCES:
            return "energy_history_reinitialized"
    if column == "kinetic_energy_rate_source":
        if left in ENERGY_RATE_SOURCES and right in RESTART_RATE_SOURCES:
            return "energy_history_reinitialized"
    return None


def _stitch(
    paths: list[Path],
    metadata: list[dict],
    keys: tuple[str, ...],
    *,
    restarted_flow: bool = False,
    boundary_records: list[dict] | None = None,
) -> list[dict]:
    """Merge accepted rows only after checking clocks and overlapping values.

    CSV values allow their printed floating-point precision; time and integer
    identities are checked separately. A declared flow restart may reset only
    the source-defined observers above; each difference is recorded and the
    prefix owns the shared row. Other overlaps remain strict. Raw rows after a
    segment's accepted endpoint are excluded.
    """
    _validate_segments(metadata)
    if restarted_flow and (keys != ("step",) or boundary_records is None):
        raise ValueError("Restarted flow stitching requires a boundary audit record")
    merged = {}
    owners = {}
    schema = None
    previous_membership = {}
    for segment_index, (path, item) in enumerate(zip(paths, metadata, strict=True)):
        start, end, initial_time, dt = _clock(item)
        membership = {}
        series_steps = {}
        seen = set()
        boundary_found = False
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            columns = set(reader.fieldnames or ()) - {"run"}
            optional_health = RESTART_HEALTH_FIELDS if restarted_flow else set()
            if not {*keys, "time"}.issubset(columns) or (
                schema is not None and columns - optional_health != schema - optional_health
            ):
                raise ValueError(f"{path}: missing or inconsistent CSV schema")
            schema = columns if schema is None else schema | columns
            for raw in reader:
                step = _integer(raw["step"], f"{path}: step")
                if not start <= step <= end:
                    continue
                if None in raw or any(value is None for value in raw.values()):
                    raise ValueError(f"{path}: incomplete accepted CSV row at step {step}")
                row = {key: _value(value) for key, value in raw.items()}
                row["step"] = step
                for key in ("group_id", "n_particles", "n_particles_total"):
                    if key in row:
                        if (
                            key == "n_particles"
                            and row.get("source") == "cross_section_field"
                            and math.isnan(row[key])
                        ):
                            continue
                        # The mode exporter uses group -1 for whole-plane data.
                        minimum = (
                            -1
                            if key == "group_id" and row.get("source") == "cross_section_field"
                            else 0
                        )
                        row[key] = _integer(row[key], f"{path}: {key}", minimum=minimum)
                expected_time = initial_time + (step - start) * dt
                if not isinstance(row["time"], float) or not math.isclose(
                    row["time"], expected_time, rel_tol=0, abs_tol=1e-10
                ):
                    raise ValueError(f"{path}: time disagrees with accepted clock at step {step}")
                key = tuple(row[name] for name in keys)
                if key in seen:
                    raise ValueError(f"{path}: duplicate accepted row {key} within one segment")
                seen.add(key)
                series = key[1:]
                if step < series_steps.get(series, step):
                    raise ValueError(f"{path}: nonmonotone accepted series {series}")
                series_steps[series] = step
                membership.setdefault(step, set()).add(series)
                if key in merged:
                    boundary = restarted_flow and segment_index > 0 and step == start
                    if boundary:
                        if not FLOW_STATE_FIELDS.issubset(columns & merged[key].keys()):
                            raise ValueError(
                                f"{path}: restart boundary lacks primitive flow state fields"
                            )
                        if any(
                            not _finite(row[column]) or not _finite(merged[key][column])
                            for column in FLOW_STATE_FIELDS
                        ):
                            raise ValueError(
                                f"{path}: restart boundary contains nonfinite primitive state"
                            )
                    differences = {}
                    for column in schema:
                        left, right = merged[key].get(column, math.nan), row.get(column, math.nan)
                        reason = (
                            _restart_observer_difference(column, merged[key], row)
                            if boundary
                            else None
                        )
                        same = _same_value(left, right)
                        if not same and reason is None:
                            raise ValueError(f"{path}: conflicting overlap {key}, column {column}")
                        if (
                            reason is not None
                            and left != right
                            and not (_missing(left) and _missing(right))
                        ):
                            differences[column] = {
                                "prefix": None if _missing(left) else left,
                                "restart": None if _missing(right) else right,
                                "reason": reason,
                            }
                    if boundary:
                        boundary_found = True
                        boundary_records.append(
                            {
                                "step": step,
                                "time": row["time"],
                                "prefix_path": str(owners[key]),
                                "restart_path": str(path),
                                "owner": "prefix",
                                "observer_differences": differences,
                            }
                        )
                else:
                    merged[key] = row
                    owners[key] = path
                if restarted_flow and segment_index > 0 and step > start:
                    if any(
                        column in schema and not _finite(row.get(column))
                        for column in RESTART_HEALTH_FIELDS
                    ):
                        raise ValueError(
                            f"{path}: accepted step {step} lacks refreshed step-health fields"
                        )
        if not membership:
            raise ValueError(f"{path}: no rows within the accepted metadata clock")
        if restarted_flow and segment_index > 0 and not boundary_found:
            raise ValueError(f"{path}: missing shared flow row at restart step {start}")
        for step in membership.keys() & previous_membership.keys():
            if membership[step] != previous_membership[step]:
                raise ValueError(
                    f"{path}: overlapping group/source membership differs at step {step}"
                )
        previous_membership = membership
    return [merged[key] for key in sorted(merged)]


def _read(run: str, chains: dict[str, tuple[str, ...]], prefix_steps: dict[str, int] | None = None):
    segments = chains.get(run, (run,))
    bounds = PREFIX_STEPS if prefix_steps is None else prefix_steps
    metadata = [
        _bounded_metadata(
            json.loads((ROOT / "solution" / segment / "vpm_metadata.json").read_text()),
            bounds.get(segment),
        )
        for segment in segments
    ]
    _validate_segments(metadata)
    tables = []
    boundary_records = []
    for directory, filename, keys in (
        ("samples", "flow_integrals.csv", ("step",)),
        ("samples", "ring_diagnostics.csv", ("step", "group_id")),
        ("figures", "breakdown_metrics.csv", ("step", "source", "group_id")),
    ):
        if len(keys) > 1 and _is_gbd(metadata[-1]):
            # GBD reassigns labels during combined diffusion. Reading these
            # group-based files would invite treating labels as ring lineage.
            tables.append([])
            continue
        rows = _stitch(
            [ROOT / directory / segment / filename for segment in segments],
            metadata,
            keys,
            restarted_flow=filename == "flow_integrals.csv",
            boundary_records=boundary_records if filename == "flow_integrals.csv" else None,
        )
        tables.append(rows)
    final_metadata = {
        **metadata[-1],
        "plot_segments": segments,
        "plot_segment_ends": [item["state"]["step"] for item in metadata],
        "plot_restart_steps": [item["state"].get("initial_step", 0) for item in metadata[1:]],
        "plot_restart_boundaries": boundary_records,
    }
    return *tables, final_metadata


def _is_gbd(metadata: dict) -> bool:
    return metadata["configuration"]["numerics"]["viscous"]["scheme"] == "GBD"


def _group_label(label: str, group, metadata: dict) -> str:
    kind = "label" if _is_gbd(metadata) else "ring"
    return f"{label}, {kind} {int(group)}"


def _group_panel_scope(data) -> str:
    if any(_is_gbd(item[3]) for item in data.values()):
        return "\nGBD omitted: field-based ring metrics pending"
    return ""


def _has_group_histories(data) -> bool:
    return any(not _is_gbd(item[3]) for item in data.values())


def _remove_omitted_group_exports(output: Path, formats) -> None:
    """Remove this plotter's stale group pages on a GBD-only rerender."""
    for name in ("mode_histories", "morphology_histories"):
        for figure_format in formats:
            (output / f"{name}.{figure_format}").unlink(missing_ok=True)


def _case_title(data) -> str:
    values = set()
    for _, _, _, metadata in data.values():
        for ring in metadata["configuration"]["initial_conditions"]:
            nu = float(ring["kinematic_viscosity"])
            reynolds = abs(float(ring["circulation"])) / nu if nu else math.inf
            values.add(f"{reynolds:.5g}")
    return "Seeded native histories, $Re_\\Gamma=" + ", ".join(sorted(values)) + "$"


def _run_colors(data) -> dict:
    palette = theme.COLOR_CYCLE
    return {
        run: (
            theme.COLORS["VPMpurple"]
            if _is_gbd(item[3])
            else theme.COLORS["FVMorange"]
            if run == "cs_breakdown_splitting_fmm_cpu_root_200000"
            else palette[index % len(palette)]
        )
        for index, (run, item) in enumerate(data.items())
    }


def _save(fig, output: Path, name: str, figure_format: str, dpi: int | None) -> None:
    theme.save_fig(fig, output / name, figure_format=figure_format, dpi=dpi)


def _style(ax, ylabel: str, xlabel: str = "Physical time [s]") -> None:
    ax.set_ylabel(ylabel)
    # Every multirow layout shares its physical clock: label it once below
    # the final row, where Matplotlib also shows the shared tick numbers.
    last_row = ax.get_subplotspec().is_last_row()
    ax.set_xlabel(xlabel if last_row else "")
    ax.tick_params(axis="x", labelbottom=last_row)
    ax.ticklabel_format(axis="both", useOffset=False, style="plain")
    ax.grid(alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)


def _constant_core_limits(values) -> tuple[float, float] | None:
    finite = [float(value) for value in values if _finite(value)]
    if not finite:
        return None
    low, high = min(finite), max(finite)
    centre = (low + high) / 2
    if high - low > max(abs(centre) * 1e-8, 1e-12):
        return None
    padding = max(abs(centre) / 12, 1e-6)
    return max(0.0, centre - padding), centre + padding


def _broken_curve(steps, times, values, boundaries) -> tuple[list, list]:
    """Insert gaps after prefix-owned endpoints without dropping either side."""
    x, y = [], []
    previous_step = None
    for step, time, value in zip(steps, times, values, strict=True):
        if previous_step is not None and any(
            previous_step <= boundary < step for boundary in boundaries
        ):
            x.append(math.nan)
            y.append(math.nan)
        x.append(time)
        y.append(value)
        previous_step = step
    return x, y


def _plot_observer(ax, flow, values, metadata, **kwargs) -> None:
    x, y = _broken_curve(flow.step, flow.time, values, metadata.get("plot_restart_steps", ()))
    ax.plot(x, y, **kwargs)


def _plot_diagnostics(data, output: Path, figure_format: str, dpi: int | None) -> None:
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(theme.MAX_FIGURE_WIDTH_CM * theme.CM, 26 * theme.CM),
        sharex="col",
        layout="constrained",
    )
    colors = _run_colors(data)
    health_ax = axes[1, 0].twinx()
    energy_measurements = {
        run: (
            "unbounded"
            if flow.kinetic_energy_rate_source.astype(str).str.contains("free_space").any()
            else "periodic"
        )
        for run, (flow, _, _, _) in data.items()
    }
    mixed_energy_measurements = len(set(energy_measurements.values())) > 1
    for run, (flow, _, _, metadata) in data.items():
        label = _label(run)
        color = colors[run]
        t = flow.time
        energy = (
            flow.total_kinetic_energy
            if mixed_energy_measurements
            else flow.total_kinetic_energy / flow.total_kinetic_energy.iloc[0]
        )
        energy_label = (
            f"{label} ({energy_measurements[run]})" if mixed_energy_measurements else label
        )
        _plot_observer(axes[0, 0], flow, energy, metadata, color=color, label=energy_label)
        _plot_observer(
            axes[0, 1],
            flow,
            flow.total_enstrophy / flow.total_enstrophy.iloc[0],
            metadata,
            color=color,
            label=label,
        )
        axes[1, 0].plot(
            t, flow.vorticity_divergence_error, color=color, label=f"{label} divergence"
        )
        health_ax.plot(
            t,
            flow.vortex_strength_misalignment_degrees,
            "--",
            color=color,
            label=f"{label} misalignment",
        )
        axes[1, 1].plot(t, flow.lagrangian_cfl, color=color, label=label)
        axes[2, 0].plot(t, flow.mean_core_radius, color=color, label=label)
        axes[2, 1].plot(t, flow.n_particles_total, color=color, label=label)
    axes[0, 0].set_title(
        "Kinetic energy\n(estimators differ)" if mixed_energy_measurements else "Kinetic energy"
    )
    axes[0, 1].set_title("Enstrophy")
    axes[1, 0].set_title("Divergence / alignment")
    axes[1, 1].set_title("Lagrangian CFL")
    axes[2, 0].set_title("Mean core radius")
    axes[2, 1].set_title("Particle count")
    energy_ylabel = r"Native $E$" if mixed_energy_measurements else r"$E/E_0$"
    for ax, ylabel in zip(
        axes.flat,
        (energy_ylabel, r"$Z/Z_0$", "Divergence error", "CFL", r"$\sigma$ [m]", "$N$"),
        strict=True,
    ):
        _style(ax, ylabel)
    health_ax.set_ylabel("misalignment [degrees]")
    health_ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    health_ax.spines["top"].set_visible(False)
    health_ax.grid(False)
    axes[1, 0].axhline(0.12, color="black", ls=":", lw=0.8, label="divergence limit")
    health_ax.axhline(25.0, color="black", ls=":", lw=0.8, label="misalignment limit")
    axes[1, 1].axhline(1.0, color="black", ls=":", lw=0.8, label="CFL limit")
    core_limits = _constant_core_limits(
        value for flow, _, _, _ in data.values() for value in flow.mean_core_radius
    )
    if core_limits is not None:
        axes[2, 0].set_ylim(*core_limits)
    handles = [
        Line2D(
            [0],
            [0],
            color=colors[run],
            label=f"{_label(run)} ({energy_measurements[run]})"
            if mixed_energy_measurements
            else _label(run),
        )
        for run in data
    ]
    handles.extend(
        [
            Line2D([0], [0], color=theme.COLORS["reference"], ls="--", label="Misalignment"),
            Line2D([0], [0], color=theme.COLORS["reference"], ls=":", label="Health limits"),
        ]
    )
    fig.legend(
        handles=handles, loc="outside upper center", ncol=2, frameon=False, title=_case_title(data)
    )
    _save(fig, output, "diagnostic_histories", figure_format, dpi)


def _plot_energy_rates(data, output: Path, figure_format: str, dpi: int | None) -> None:
    fig, axes = plt.subplots(
        2, 1, figsize=theme.figure_size("stacked"), sharex=True, layout="constrained"
    )
    colors = _run_colors(data)
    for run, (flow, _, _, metadata) in data.items():
        for ax, column in zip(
            axes, ("kinetic_energy_rate", "viscous_kinetic_energy_rate"), strict=True
        ):
            _plot_observer(ax, flow, flow[column], metadata, color=colors[run], label=_label(run))
    for ax, title in zip(axes, ("Native energy rate", "Viscous energy rate"), strict=True):
        ax.set_title(title)
        _style(ax, r"Rate [m$^5$/s$^3$]")
    axes[0].legend(frameon=False)
    fig.suptitle(_case_title(data) + "\nNo curve joins across restart observers")
    _save(fig, output, "energy_rate_histories", figure_format, dpi)


def _plot_modes(data, output: Path, figure_format: str, dpi: int | None) -> None:
    if not _has_group_histories(data):
        return
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(theme.MAX_FIGURE_WIDTH_CM * theme.CM, 18 * theme.CM),
        sharex="col",
        layout="constrained",
    )
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    series_colors = {}
    for run, (_, _, modes, metadata) in data.items():
        if _is_gbd(metadata):
            continue
        label = _label(run)
        particles = modes[modes.source == "particle_backup"]
        for group, group_rows in particles.groupby("group_id"):
            suffix = _group_label(label, group, metadata)
            series_colors.setdefault((run, int(group)), palette[len(series_colors) % len(palette)])
            color = series_colors[(run, int(group))]
            axes[0, 0].plot(
                group_rows.time, group_rows.axial_mode8_amplitude, "o-", color=color, label=suffix
            )
            axes[0, 1].plot(
                group_rows.time, group_rows.radial_mode8_amplitude, "o-", color=color, label=suffix
            )
            axes[1, 0].plot(
                group_rows.time, group_rows.axial_mode8_real, "o-", color=color, label=suffix
            )
            axes[1, 0].plot(
                group_rows.time,
                group_rows.axial_mode8_imag,
                "--",
                color=color,
                label=f"{suffix} Im",
            )
            axes[1, 1].plot(
                group_rows.time, group_rows.radial_mode8_real, "o-", color=color, label=suffix
            )
            axes[1, 1].plot(
                group_rows.time,
                group_rows.radial_mode8_imag,
                "--",
                color=color,
                label=f"{suffix} Im",
            )
    for ax, title, ylabel in zip(
        axes.flat,
        ("Axial mode 8", "Radial mode 8", "Axial Re/Im", "Radial Re/Im"),
        (r"$|a_8|/R_0$", r"$|r_8|/R_0$", r"Coefficient / $R_0$", r"Coefficient / $R_0$"),
        strict=True,
    ):
        ax.set_title(title)
        _style(ax, ylabel)
    axes[0, 0].axhline(0.05, color="black", ls=":", lw=0.8, label="initial axial seed")
    axes[0, 0].legend(frameon=False)
    axes[1, 0].legend(
        handles=[
            Line2D([0], [0], color="black", marker="o", linestyle="-", label="Re"),
            Line2D([0], [0], color="black", linestyle="--", label="Im"),
        ],
        frameon=False,
        loc="lower left",
    )
    fig.suptitle(_case_title(data) + _group_panel_scope(data))
    _save(fig, output, "mode_histories", figure_format, dpi)


def _plot_morphology(data, output: Path, figure_format: str, dpi: int | None) -> None:
    if not _has_group_histories(data):
        return
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(theme.MAX_FIGURE_WIDTH_CM * theme.CM, 18 * theme.CM),
        sharex="col",
        layout="constrained",
    )
    for run, (_, rings, _, metadata) in data.items():
        if _is_gbd(metadata):
            continue
        label = _label(run)
        for group, rows in rings.groupby("group_id"):
            rows = rows.sort_values("time")
            suffix = _group_label(label, group, metadata)
            axes[0, 0].plot(rows.time, rows.vortex_centroid_x, label=suffix)
            axes[0, 1].plot(rows.time, rows.major_radius, label=suffix)
            axes[1, 0].plot(rows.time, rows.impulse_radius, label=suffix)
            axes[1, 1].plot(rows.time, rows.tube_circulation, label=suffix)
    for ax, title, ylabel in zip(
        axes.flat,
        ("Group centroid", "Major-radius proxy", "Impulse-radius proxy", "Circulation proxy"),
        ("$x$ [m]", "$R$ [m]", "$R$ [m]", r"$\sum |\Gamma|/(2\pi R)$"),
        strict=True,
    ):
        ax.set_title(title)
        _style(ax, ylabel)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(_case_title(data) + _group_panel_scope(data))
    _save(fig, output, "morphology_histories", figure_format, dpi)


def _plot_impulses(data, output: Path, figure_format: str, dpi: int | None) -> None:
    fig, axes = plt.subplots(
        2, 1, figsize=theme.figure_size("stacked"), sharex=True, layout="constrained"
    )
    styles = ("-", "--", ":", "-.")
    for index, (run, (flow, _, _, _)) in enumerate(data.items()):
        label = _label(run)
        for component, color in zip("xyz", theme.COLOR_CYCLE[:3], strict=True):
            axes[0].plot(
                flow.time,
                flow[f"linear_impulse_{component}"],
                color=color,
                ls=styles[index % len(styles)],
                label=f"{label} {component}",
            )
            axes[1].plot(
                flow.time,
                flow[f"angular_impulse_{component}"],
                color=color,
                ls=styles[index % len(styles)],
                label=f"{label} {component}",
            )
    axes[0].set_title("Linear impulse")
    axes[1].set_title("Angular impulse")
    for ax, ylabel in zip(axes, (r"Component [m$^4$/s]", r"Component [m$^5$/s]"), strict=True):
        _style(ax, ylabel)
        ax.legend(frameon=False, ncol=2)
    fig.suptitle(_case_title(data))
    _save(fig, output, "impulse_histories", figure_format, dpi)


def _write_summary(data, output: Path) -> None:
    rows = [
        "# Seeded native history summary",
        "",
        "These plots use native sampler CSVs and, for CS, the corrected particle-mode CSVs. They are not a matched LBM comparison.",
        "",
        "Only rows within each segment's metadata clock and declared prefix boundary are included. The resource-stopped root GBD qualification is capped at its operator-verified native step 100; its raw CSV tail is excluded. Overlapping clocks, group/source membership and values are checked before merging; run duration and output locations may differ, but numerical and initial physical configurations must match. This CSV/metadata check does not independently verify native restart hashes.",
        "",
        "At a declared flow restart, primitive particle sums/count/core/impulse fields must agree. The prefix owns the shared row. Source-defined observer resets and narrowly bounded gradient/LES roundoff are recorded in restart_boundaries.json below; no other differences are accepted. Energy, enstrophy and energy-rate curves break after each restart boundary because the diagnostic lattice/history is reinitialized. Their segments do not establish continuous dissipation across that change.",
        "",
        "GBD group IDs are dominant-label/nearest-filled-node assignments after combined diffusion, not material lineage. Group centroids, radius proxies and mode amplitudes can change through relabeling even before reconnection. GBD per-group CSVs are therefore neither read nor plotted; its ring morphology/mode histories are unavailable pending field-based metrics. Retained GBD plots show global health, energy, particle count and impulses only. Assess overtaking/breakdown separately from native fields and plane-peak tracking.",
        "",
        "| run | lifecycle | accepted step | accepted time (s) | last flow step | last flow time (s) | last N | last divergence | last misalignment (deg) | last CFL |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run, (flow, _, _, metadata) in data.items():
        last = flow.iloc[-1]
        rows.append(
            f"| {run} | {metadata.get('lifecycle', {}).get('status', 'unrecorded')} | {metadata['state']['step']} | "
            f"{metadata['state']['time']:.4g} | {int(last.step)} | {last.time:.4g} | {int(last.n_particles_total)} | "
            f"{last.vorticity_divergence_error:.5g} | {last.vortex_strength_misalignment_degrees:.5g} | "
            f"{last.lagrangian_cfl:.5g} |"
        )
    rows.extend(["", "Selected segment order:", ""])
    if not _has_group_histories(data):
        rows.extend(
            [
                "GBD-only selection: mode_histories and morphology_histories exports are omitted; field-based ring histories are produced separately.",
                "",
            ]
        )
    for run, (_, _, _, metadata) in data.items():
        rows.append(
            f"- `{run}`: "
            + " → ".join(
                f"`{segment}` (through step {end})"
                for segment, end in zip(
                    metadata["plot_segments"], metadata["plot_segment_ends"], strict=True
                )
            )
        )
    records = {
        run: metadata["plot_restart_boundaries"] for run, (_, _, _, metadata) in data.items()
    }
    rows.extend(
        [
            "",
            "Restart observer differences (prefix owner retained):",
            "",
            "| run | step | field | prefix | restart | reason |",
            "|---|---:|---|---:|---:|---|",
        ]
    )
    for run, boundaries in records.items():
        for boundary in boundaries:
            for field, difference in sorted(boundary["observer_differences"].items()):
                rows.append(
                    f"| {run} | {boundary['step']} | {field} | {difference['prefix']} | {difference['restart']} | {difference['reason']} |"
                )
    (output / "summary.md").write_text("\n".join(rows) + "\n")
    (output / "restart_boundaries.json").write_text(
        json.dumps(records, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument(
        "--chain",
        action="append",
        default=[],
        metavar="RUN=PREFIX,CONTINUATION",
        help="Explicit chronological chain ending in a selected run; repeat for multiple chains",
    )
    parser.add_argument(
        "--prefix-through",
        action="append",
        default=[],
        metavar="SEGMENT=STEP",
        help="Declare a verified native prefix endpoint; root GBD qualification defaults to step 100",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "figures/cs_breakdown")
    parser.add_argument("--formats", nargs="+", help="Shared export formats (default: PNG and PDF)")
    parser.add_argument("--dpi", type=int, help="Override the shared raster resolution")
    parser.add_argument(
        "--thesis", action="store_true", help="Use the shared LaTeX thesis font setup"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate selected metadata/CSVs without plotting imports or output writes",
    )
    args = parser.parse_args()
    if args.dpi is not None and args.dpi <= 0:
        parser.error("--dpi must be positive")
    try:
        runs = [_run_name(run) for run in args.runs]
        chains = _chains(args.chain, runs)
        segments = {segment for run in runs for segment in chains.get(run, (run,))}
        bounds = _prefix_steps(args.prefix_through, segments)
        rows = {run: _read(run, chains, bounds) for run in runs}
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.error(str(error))
    if args.check_only:
        for run, (flow, rings, modes, metadata) in rows.items():
            print(
                f"{run}: accepted through step {metadata['state']['step']}; {len(flow)} flow, {len(rings)} ring, {len(modes)} mode rows"
            )
            for boundary in metadata["plot_restart_boundaries"]:
                print(json.dumps({"run": run, **boundary}, sort_keys=True, allow_nan=False))
        return
    _load_plotting()
    formats = args.formats or theme.EXPORT_FORMATS
    if any(value not in theme.EXPORT_FORMATS for value in formats):
        parser.error(f"--formats must be chosen from {theme.EXPORT_FORMATS}")
    (theme.set_thesis_style if args.thesis else theme.set_style)()
    data = {
        run: (pd.DataFrame(flow), pd.DataFrame(rings), pd.DataFrame(modes), metadata)
        for run, (flow, rings, modes, metadata) in rows.items()
    }
    args.output.mkdir(parents=True, exist_ok=True)
    if not _has_group_histories(data):
        _remove_omitted_group_exports(args.output, theme.EXPORT_FORMATS)
    for figure_format in formats:
        _plot_diagnostics(data, args.output, figure_format, args.dpi)
        _plot_energy_rates(data, args.output, figure_format, args.dpi)
        _plot_modes(data, args.output, figure_format, args.dpi)
        _plot_morphology(data, args.output, figure_format, args.dpi)
        _plot_impulses(data, args.output, figure_format, args.dpi)
    _write_summary(data, args.output)
    print(f"wrote native seeded plots to {args.output}")


if __name__ == "__main__":
    main()
