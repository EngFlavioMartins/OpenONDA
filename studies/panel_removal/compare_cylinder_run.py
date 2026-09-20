"""Compare cylinder runs at identical physical times without phase adjustment.

A short startup record never passes the mature-flow gate. Default averaging is
80 s onward, matching the isolated reference campaign; legacy reference files
ending at 100 s can supply provisional diagnostics but not enough shedding cycles.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/reference_flow"
spec = importlib.util.spec_from_file_location(
    "cylinder_reference_statistics", REFERENCE / "postprocess_grid_study.py"
)
statistics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(statistics)


def final_segment(path):
    """Read the last chronological CSV segment with finite times in seconds.

    Returns a DataFrame after the last backwards time jump, preserving
    same-time profile rows. Saved times are rounded to eight decimals. Missing
    or nonnumeric/nonfinite time columns raise ValueError; source data are
    never edited and no missing sample is filled."""
    data = pd.read_csv(path)
    if "time" not in data:
        raise ValueError(f"{path} has no time column")
    try:
        data["time"] = pd.to_numeric(data["time"], errors="raise").astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} has nonnumeric sample times") from error
    if not np.isfinite(data.time.to_numpy()).all():
        raise ValueError(f"{path} has nonfinite sample times")
    reset = np.flatnonzero(np.diff(data.time.to_numpy()) < -1.0e-10)
    if len(reset):
        data = data.iloc[reset[-1] + 1 :].copy()
    data["time"] = data.time.round(8)
    return data


def cycle_frequency(time, lift):
    """Estimate shedding frequency from complete cycles, without a phase fit.

    FFT bins identify a spectral band only. The independent cycle estimate uses
    interpolated upward zero crossings of the time-mean-centered lift signal.
    Reported uncertainty screens sampling, cycle variability and drift; it is
    not a substitute for extending a nonstationary record.
    """
    time, lift = np.asarray(time, dtype=float), np.asarray(lift, dtype=float)
    result = {
        "qualified": False,
        "frequency": None,
        "complete_periods": 0,
        "reasons": [],
        "method": "time-mean-centered upward zero crossings",
    }
    if (
        len(time) < 32
        or time.shape != lift.shape
        or not np.isfinite(time).all()
        or not np.isfinite(lift).all()
        or np.any(np.diff(time) <= 0)
    ):
        result["reasons"].append("insufficient finite increasing samples")
        return result
    centered = lift - statistics._mean(time, lift)
    rms = np.sqrt(statistics._mean(time, centered**2))
    if rms < 1.0e-7:
        result["reasons"].append("lift oscillation is below the signal floor")
        return result
    uniform = np.linspace(time[0], time[-1], len(time))
    interpolated = np.interp(uniform, time, centered)
    power = np.abs(np.fft.rfft(interpolated * np.hanning(len(time)))) ** 2
    power[0] = 0.0
    frequencies = np.fft.rfftfreq(len(time), uniform[1] - uniform[0])
    peak_index = 1 + int(np.argmax(power[1:]))
    peak = float(frequencies[peak_index])
    resolution = float(frequencies[1] - frequencies[0])
    result.update(
        fft_peak_frequency=peak, fft_bin_width=resolution, fft_relative_bin_width=resolution / peak
    )
    indices = np.flatnonzero((centered[:-1] <= 0.0) & (centered[1:] > 0.0))
    crossings = (
        time[indices] - centered[indices] * np.diff(time)[indices] / np.diff(centered)[indices]
    )
    periods = np.diff(crossings)
    result["complete_periods"] = len(periods)
    if len(periods) < 1 or np.any(periods <= 0):
        result["reasons"].append("no complete positive shedding period")
        return result
    mean_period = float(np.mean(periods))
    frequency = 1.0 / mean_period
    cv = float(np.std(periods, ddof=1) / mean_period) if len(periods) > 1 else None
    midpoints = 0.5 * (crossings[:-1] + crossings[1:])
    halves = [
        periods[midpoints < 0.5 * (time[0] + time[-1])],
        periods[midpoints >= 0.5 * (time[0] + time[-1])],
    ]
    half_frequencies = [float(1.0 / np.mean(values)) if len(values) else None for values in halves]
    drift = (
        None
        if any(value is None for value in half_frequencies)
        else abs(half_frequencies[1] - half_frequencies[0]) / frequency
    )
    dt_max = float(np.diff(time).max())
    samples_per_cycle = mean_period / dt_max
    spectral_consistent = abs(frequency - peak) <= resolution
    result.update(
        frequency=frequency,
        mean_period=mean_period,
        period_standard_deviation=float(np.std(periods, ddof=1)) if len(periods) > 1 else None,
        period_cv=cv,
        period_range=[float(periods.min()), float(periods.max())],
        period_5_95=np.quantile(periods, [0.05, 0.95]).tolist(),
        half_window_frequencies=half_frequencies,
        half_window_period_counts=[len(values) for values in halves],
        half_window_relative_drift=drift,
        minimum_samples_per_cycle=samples_per_cycle,
        spectral_band_consistent=bool(spectral_consistent),
        complete_cycle_window=[float(crossings[0]), float(crossings[-1])],
    )
    if len(periods) < 8:
        result["reasons"].append("fewer than eight complete periods")
    if min(len(values) for values in halves) < 3:
        result["reasons"].append("fewer than three complete periods in a half window")
    if cv is None or cv > 0.03:
        result["reasons"].append("cycle period spread exceeds the 3% frequency target")
    if drift is None or drift > 0.03:
        result["reasons"].append("half-window frequency drift exceeds 3%")
    if samples_per_cycle < 20:
        result["reasons"].append("fewer than twenty samples per cycle")
    if not spectral_consistent:
        result["reasons"].append("cycle frequency disagrees with the dominant FFT band")
    if len(periods) > 1:
        from scipy.stats import t as student_t

        period_half_width = float(
            student_t.ppf(0.975, len(periods) - 1) * np.std(periods, ddof=1) / np.sqrt(len(periods))
        )
        if period_half_width < mean_period:
            cycle_half_width = max(
                1.0 / (mean_period - period_half_width) - frequency,
                frequency - 1.0 / (mean_period + period_half_width),
            )
            # If c0 lies in [a0,b0] and cN in [aN,bN], their true separation
            # lies in [aN-b0,bN-a0]. Invert those bounds for N / separation.
            # This requires no assumed interpolation-error order, and narrows
            # over many complete periods instead of imposing a dt/period floor.
            first_bracket = time[indices[0] : indices[0] + 2]
            last_bracket = time[indices[-1] : indices[-1] + 2]
            duration_lower = last_bracket[0] - first_bracket[1]
            duration_upper = last_bracket[1] - first_bracket[0]
            if duration_lower <= 0:
                result["reasons"].append("endpoint crossing brackets overlap")
                return result
            sampling_frequency_interval = [
                len(periods) / duration_upper,
                len(periods) / duration_lower,
            ]
            timing_half_width = max(
                frequency - sampling_frequency_interval[0],
                sampling_frequency_interval[1] - frequency,
            )
            result.update(
                endpoint_crossing_brackets=[first_bracket.tolist(), last_bracket.tolist()],
                crossing_duration_interval=[float(duration_lower), float(duration_upper)],
                sampling_frequency_interval=[float(value) for value in sampling_frequency_interval],
                sampling_frequency_half_width=float(timing_half_width),
            )
            drift_half_width = 0.0 if drift is None else 0.5 * drift * frequency
            half_width = float(max(cycle_half_width, timing_half_width, drift_half_width))
            result.update(
                frequency_half_width=half_width,
                relative_frequency_half_width=half_width / frequency,
                uncertainty_note="Maximum of cycle-mean Student-t half-width, exact endpoint-bracket frequency bound and half-window drift; correlated or nonstationary cycles require longer records.",
            )
            if half_width / frequency > 0.03:
                result["reasons"].append("cycle frequency uncertainty exceeds 3%")
        else:
            result["reasons"].append("cycle period uncertainty is unbounded")
    else:
        result["reasons"].append("cycle period uncertainty is unavailable")
    result["qualified"] = not result["reasons"]
    return result


def force_comparison(trial, reference, start, end):
    """Compare force statistics at exact common saved physical times.

    Parameters
    ----------
    trial, reference : pathlib.Path
        Sample directories containing forces_history.csv.
    start, end : float
        Inclusive comparison interval in seconds.

    Returns
    -------
    dict
        Dimensionless force means/fluctuation RMS, cycle-frequency evidence,
        fractional relative changes and actual common-time coverage. Moments
        use trapezoidal time weights. Frequencies are Hz; for the unit-speed,
        unit-diameter cylinder their numerical value is the Strouhal number.
        Frequency differences remain unqualified if complete-cycle evidence
        is insufficient; equal FFT bins alone cannot qualify agreement.

    Raises
    ------
    ValueError
        If fewer than 32 finite common samples exist or required columns are missing."""
    left = (
        final_segment(trial / "forces_history.csv")
        .drop_duplicates("time", keep="last")
        .set_index("time")
    )
    right = (
        final_segment(reference / "forces_history.csv")
        .drop_duplicates("time", keep="last")
        .set_index("time")
    )
    times = left.index.intersection(right.index)
    times = times[(times >= start) & (times <= end)].to_numpy()
    if len(times) < 32:
        raise ValueError("Need at least32 common force samples")
    values = {}
    signals = {}
    for name, table in (("trial", left), ("reference", right)):
        required = ("drag_coefficient", "lift_coefficient")
        missing = [column for column in required if column not in table]
        if missing:
            raise ValueError(f"{name} force history is missing {missing}")
        # Real ForceSampler records also contain a textual `patch` field and
        # provenance columns. Only the coefficients consumed below are numeric
        # inputs to this comparison; metadata must never enter np.isfinite.
        columns = (
            *required,
            *(("side_force_coefficient",) if "side_force_coefficient" in table else ()),
        )
        numeric = {"time": times}
        for column in columns:
            try:
                numeric[column] = pd.to_numeric(table.loc[times, column], errors="raise").to_numpy(
                    dtype=float
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} force history has nonnumeric {column}") from error
        if not all(np.isfinite(array).all() for array in numeric.values()):
            raise ValueError("Nonfinite force sample")
        signals[name] = numeric
        values[name] = statistics._force_statistics(numeric, float(times[0]), float(times[-1]))
        values[name]["cycle_frequency_lift"] = cycle_frequency(times, numeric["lift_coefficient"])
    metrics = ("mean_drag", "rms_drag", "rms_lift", "strouhal_lift")
    differences = {
        metric: statistics._difference(values["trial"][metric], values["reference"][metric])
        for metric in metrics
    }
    differences["strouhal_lift_fft_bin"] = differences["strouhal_lift"]
    trial_frequency = values["trial"]["cycle_frequency_lift"]
    reference_frequency = values["reference"]["cycle_frequency_lift"]
    qualified = trial_frequency["qualified"] and reference_frequency["qualified"]
    differences["strouhal_lift"] = (
        statistics._difference(trial_frequency["frequency"], reference_frequency["frequency"])
        if qualified
        else None
    )
    combined_uncertainty = (
        (
            (trial_frequency["frequency_half_width"] + reference_frequency["frequency_half_width"])
            / reference_frequency["frequency"]
        )
        if qualified
        else None
    )
    frequency_comparison = {
        "qualified": bool(qualified),
        "cycle_relative_change": differences["strouhal_lift"],
        "combined_relative_half_width": combined_uncertainty,
        "conservative_relative_difference_bound": differences["strouhal_lift"]
        + combined_uncertainty
        if qualified
        else None,
        "fft_note": "Matching raw FFT bins does not resolve a 3% frequency comparison with only 8–16 cycles.",
    }
    delta = signals["trial"]["drag_coefficient"] - signals["reference"]["drag_coefficient"]
    return {
        "window": [float(times[0]), float(times[-1])],
        "common_samples": len(times),
        **values,
        "relative_changes": differences,
        "frequency_comparison": frequency_comparison,
        "instantaneous_cd_difference_rms": float(np.sqrt(statistics._mean(times, delta**2))),
    }


def time_mean_profile(table, axis, components, times):
    """Return trapezoidal time means at every fixed spatial probe.

    ``table`` contains the named ``axis`` coordinate and velocity ``components``;
    ``times`` is the exact shared physical-time array in seconds. The returned
    DataFrame retains coordinates and velocity units. Every probe must cover
    all requested times or ValueError is raised; time interpolation is absent."""
    selected = table.loc[table.time.isin(times)].drop_duplicates(["time", axis], keep="last")
    means = []
    positions = []
    for position, rows in selected.groupby(axis, sort=True):
        rows = rows.sort_values("time")
        if len(rows) != len(times) or not np.array_equal(rows.time.to_numpy(), times):
            raise ValueError("Profile probe is missing a common physical sample time")
        try:
            values = rows[components].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("Nonnumeric profile velocity") from error
        means.append(np.trapezoid(values, times, axis=0) / (times[-1] - times[0]))
        positions.append(position)
    return pd.DataFrame(means, index=np.asarray(positions), columns=components)


def profile_comparison(trial, reference, name, start, end, prefix="fvm_", minimum_coordinate=None):
    """Compare time-mean vector profiles only on common fluid support.

    Parameters
    ----------
    trial, reference : pathlib.Path
        Directories containing the selected saved line samples.
    name : str
        Centreline or transverse sampler name.
    start, end : float
        Inclusive averaging window in seconds.
    prefix : str
        Coupled sampler prefix (fvm_ or vpm_).
    minimum_coordinate : float or None
        Optional exclusive downstream x/D cutoff.

    Returns
    -------
    dict
        Trapezoidal time-mean profiles are compared using a spatial integral
        of squared three-component velocity error. Spatial interpolation stays
        inside each reference fluid segment. Relative L2 is dimensionless;
        RMS_Uinf and covered_length_D assume unit freestream speed and diameter.
        Actual times, extents, excluded points and roundoff snapping are recorded.

    Raises
    ------
    ValueError
        If common times, finite velocity or shared fluid support are insufficient."""
    left = final_segment(trial / f"{prefix}{name}.csv")
    right = final_segment(reference / f"{name}.csv")
    times = np.intersect1d(left.time.unique(), right.time.unique())
    times = times[(times >= start) & (times <= end)]
    if len(times) < 2:
        raise ValueError("Insufficient common profile times")
    axis = "position_x" if name == "centreline" else "position_y"
    components = ["velocity_x", "velocity_y", "velocity_z"]
    for label, table in (("trial", left), ("reference", right)):
        for coordinate in ("position_x", "position_y", "position_z"):
            if coordinate not in table:
                raise ValueError(f"{label} profile has no {coordinate}")
            try:
                table[coordinate] = pd.to_numeric(table[coordinate], errors="raise").astype(float)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{label} profile has nonnumeric {coordinate}") from error
            if not np.isfinite(table[coordinate].to_numpy()).all():
                raise ValueError(f"{label} profile has nonfinite {coordinate}")
    a = time_mean_profile(left, axis, components, times)
    b = time_mean_profile(right, axis, components, times)
    if minimum_coordinate is not None:
        a = a.loc[a.index > minimum_coordinate]
    if name == "centreline":
        fluid = np.abs(a.index.to_numpy()) > 0.5 + 1.0e-8
        segments = [
            ("upstream", a.index < -0.5, b.index < -0.5),
            ("downstream", a.index > 0.5, b.index > 0.5),
        ]
    else:
        fluid = np.ones(len(a), dtype=bool)
        segments = [("transverse", fluid, np.ones(len(b), dtype=bool))]
    numerator = denominator = length = 0.0
    coverage = []
    compared_points = 0
    for label, target_mask, source_mask in segments:
        target = a.loc[target_mask & fluid]
        if len(target) == 0:
            continue
        source = b.loc[source_mask]
        if len(source) < 2:
            raise ValueError(f"No fluid reference support for {label} profile segment")
        source_x = source.index.to_numpy()
        # Clip each fluid segment independently. A global min/max check would
        # otherwise silently extend the nearest wall-adjacent source value.
        endpoint_roundoff = 64 * np.finfo(float).eps * np.maximum(1.0, np.abs(source_x[[0, -1]]))
        supported = (target.index >= source_x[0] - endpoint_roundoff[0]) & (
            target.index <= source_x[-1] + endpoint_roundoff[1]
        )
        retained = target.loc[supported]
        if len(retained) < 2:
            raise ValueError(f"Insufficient shared fluid support in {label} segment")
        x = retained.index.to_numpy()
        u = retained.to_numpy()
        source_values = source.to_numpy()
        if not np.isfinite(u).all() or not np.isfinite(source_values).all():
            raise ValueError("Nonfinite fluid profile velocity")
        # CSVs may spell the identical endpoint as .6 versus .6000000000000001.
        # Snap only within double-precision roundoff, then interpolate strictly
        # inside source support. Physically distinct near-wall probes stay out.
        query_x = np.clip(x, source_x[0], source_x[-1])
        v = np.column_stack(
            [np.interp(query_x, source_x, source_values[:, j]) for j in range(len(components))]
        )
        numerator += np.trapezoid(np.sum((u - v) ** 2, axis=1), x)
        denominator += np.trapezoid(np.sum(v**2, axis=1), x)
        segment_length = float(x[-1] - x[0])
        length += segment_length
        compared_points += len(x)
        coverage.append(
            {
                "segment": label,
                "reference_fluid_extent_D": [float(source_x[0]), float(source_x[-1])],
                "trial_fluid_extent_D": [float(target.index.min()), float(target.index.max())],
                "compared_extent_D": [float(x[0]), float(x[-1])],
                "covered_length_D": segment_length,
                "trial_points": len(target),
                "compared_points": len(x),
                "points_excluded_outside_reference_support": int((~supported).sum()),
                "roundoff_snapped_points": int(np.count_nonzero(query_x != x)),
            }
        )
    if not np.isfinite(numerator + denominator) or length <= 0:
        raise ValueError("Invalid profile comparison")
    return {
        "common_times": len(times),
        "time_window": [float(times[0]), float(times[-1])],
        "time_average": "trapezoidal over common physical times",
        "minimum_coordinate_exclusive": minimum_coordinate,
        "covered_length_D": float(length),
        "trial_points": len(a),
        "compared_points": compared_points,
        "points_excluded": len(a) - compared_points,
        "points_excluded_solid": int((~fluid).sum()),
        "points_excluded_outside_reference_support": sum(
            item["points_excluded_outside_reference_support"] for item in coverage
        ),
        "coverage": coverage,
        "roundoff_snapped_points": sum(item["roundoff_snapped_points"] for item in coverage),
        "mean_velocity_relative_l2": float(np.sqrt(numerator / max(denominator, 1.0e-14))),
        "mean_velocity_rms_Uinf": float(np.sqrt(numerator / length)),
    }


def cadence_coverage(times, start, end, cadence):
    """Report every missing expected sample on a physical-time cadence.

    ``times``, inclusive ``start``/``end``, and positive ``cadence`` are seconds.
    The returned mapping requires at least two expected samples and records
    missing counts/times. Invalid cadence returns an explicit incomplete result."""
    if cadence is None or not np.isfinite(cadence) or cadence <= 0:
        return {"complete": False, "reason": "valid experiment coupling_dt is required"}
    first = int(np.ceil((start - 1.0e-8) / cadence))
    last = int(np.floor((end + 1.0e-8) / cadence))
    expected = np.round(np.arange(first, last + 1) * cadence, 8)
    actual = np.unique(np.round(np.asarray(times, dtype=float), 8))
    actual = actual[(actual >= start - 1.0e-8) & (actual <= end + 1.0e-8)]
    missing = np.setdiff1d(expected, actual)
    return {
        "complete": len(expected) >= 2 and len(missing) == 0,
        "cadence": float(cadence),
        "expected_samples": len(expected),
        "present_expected_samples": len(expected) - len(missing),
        "missing_samples": len(missing),
        "first_missing_times": missing[:10].tolist(),
        "expected_window": [float(expected[0]), float(expected[-1])] if len(expected) else None,
    }


def vpm_profile_positions(name):
    """Return the named launcher line positions in metres using float32 storage.

    The unit-diameter case uses nominal 0.08 D spacing; the endpoint-inclusive
    point count is computed exactly as in the VPM line sampler."""
    start, end = (
        ([0.6, 0.0, 0.0], [12.0, 0.0, 0.0])
        if name == "centreline"
        else ([float(name[-1]), -3.0, 0.0], [float(name[-1]), 3.0, 0.0])
    )
    start, end = np.asarray(start, dtype=np.float32), np.asarray(end, dtype=np.float32)
    count = max(2, int(np.ceil(np.linalg.norm(end - start) / 0.08)) + 1)
    parameter = np.linspace(0.0, 1.0, count, dtype=np.float32)
    return start[None, :] + parameter[:, None] * (end - start)[None, :]


def vpm_profile_evidence(trial, reference, start, end, cadence, transfer_downstream):
    """Audit whole-wake and exterior profiles over the requested time window.

    ``trial``/``reference`` are sample directories; ``start``, ``end`` and
    ``cadence`` are seconds. ``transfer_downstream`` is the exclusive exterior
    x/D threshold for this unit-diameter case. Returns profile errors plus
    sample/probe coverage; missing evidence remains explicitly unqualified."""
    profiles, evidence = {}, {}
    for name in ("centreline", "transverse_x1", "transverse_x2", "transverse_x4"):
        path = trial / f"vpm_{name}.csv"
        reference_path = reference / f"{name}.csv"
        expected = vpm_profile_positions(name)
        if not path.is_file() or not reference_path.is_file():
            profiles[name] = {
                "available": False,
                "reason": "VPM or reference profile file is absent",
            }
            evidence[name] = {
                **cadence_coverage([], start, end, cadence),
                "spatial_complete": False,
            }
            continue
        try:
            frame = final_segment(path)
            ref = final_segment(reference_path)
            common = np.intersect1d(frame.time.unique(), ref.time.unique())
            evidence[name] = cadence_coverage(common, start, end, cadence)
            axis = "position_x" if name == "centreline" else "position_y"
            invalid_frames = []
            in_window = frame[(frame.time >= start - 1.0e-8) & (frame.time <= end + 1.0e-8)]
            for time, group in in_window.groupby("time"):
                group = group.sort_values(axis)
                columns = [
                    "position_x",
                    "position_y",
                    "position_z",
                    "velocity_x",
                    "velocity_y",
                    "velocity_z",
                ]
                values = group[columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
                tolerance = 4 * np.finfo(np.float32).eps * np.maximum(1.0, np.abs(expected))
                valid = (
                    len(group) == len(expected)
                    and np.isfinite(values).all()
                    and np.all(np.abs(values[:, :3] - expected) <= tolerance)
                )
                if not valid:
                    invalid_frames.append(float(time))
            evidence[name].update(
                expected_points_per_frame=len(expected),
                expected_start=expected[0].astype(float).tolist(),
                expected_end=expected[-1].astype(float).tolist(),
                invalid_spatial_frames=len(invalid_frames),
                first_invalid_frame_times=invalid_frames[:10],
                spatial_complete=bool(len(in_window)) and not invalid_frames,
            )
            comparison = profile_comparison(trial, reference, name, start, end, prefix="vpm_")
            profiles[name] = {"available": True, **comparison}
            evidence[name]["reference_support_complete"] = (
                comparison["compared_points"] == len(expected)
                and comparison["points_excluded_outside_reference_support"] == 0
            )
        except (ValueError, KeyError, TypeError) as error:
            profiles[name] = {"available": False, "reason": str(error)}
            evidence.setdefault(name, cadence_coverage([], start, end, cadence))
            evidence[name]["spatial_complete"] = False
    try:
        if (
            transfer_downstream is None
            or not np.isfinite(transfer_downstream)
            or not 0.6 < transfer_downstream < 12.0
        ):
            raise ValueError("A valid declared transfer downstream boundary is required")
        exterior = profile_comparison(
            trial,
            reference,
            "centreline",
            start,
            end,
            prefix="vpm_",
            minimum_coordinate=transfer_downstream,
        )
        profiles["centreline_exterior"] = {"available": True, **exterior}
        expected_exterior = int(
            np.count_nonzero(vpm_profile_positions("centreline")[:, 0] > transfer_downstream)
        )
        evidence["centreline_exterior"] = {
            "complete": evidence.get("centreline", {}).get("complete", False),
            "spatial_complete": evidence.get("centreline", {}).get("spatial_complete", False),
            "expected_points_per_frame": expected_exterior,
            "reference_support_complete": exterior["compared_points"] == expected_exterior
            and exterior["points_excluded_outside_reference_support"] == 0,
            "transfer_downstream": transfer_downstream,
        }
    except (OSError, ValueError, KeyError, TypeError) as error:
        profiles["centreline_exterior"] = {"available": False, "reason": str(error)}
        evidence["centreline_exterior"] = {
            "complete": False,
            "spatial_complete": False,
            "reference_support_complete": False,
        }
    return profiles, evidence


def full_donor_planarity(rows, start, end, cadence):
    """Audit recorded whole-FVM span variation at every expected exchange.

    The returned evidence uses the saved velocity scale and a 1e-3 threshold,
    not only sparse plotting probes. Missing or nonfinite donor diagnostics
    prevent qualification."""
    times = []
    invalid = []
    normalized = []
    for row in rows:
        try:
            metrics = row["planar_spanwise_consistency"]
            velocity = float(metrics["span_velocity_max"])
            variation = float(metrics["span_variation_max"])
            scale = float(metrics["velocity_scale"])
            if (
                not np.isfinite([velocity, variation, scale]).all()
                or min(velocity, variation) < 0.0
                or scale <= 0.0
            ):
                raise ValueError("invalid full-donor metric")
            ratios = (velocity / scale, variation / scale)
            if not np.isfinite(ratios).all():
                raise ValueError("invalid normalized full-donor metric")
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            invalid.append(float(row["time"]))
            continue
        times.append(row["time"])
        normalized.append(ratios)
    result = cadence_coverage(times, start, end, cadence)
    result["complete"] = result["complete"] and not invalid
    result["invalid_records"] = len(invalid)
    result["first_invalid_times"] = invalid[:10]
    result["max_normalized_span_velocity"] = max((v[0] for v in normalized), default=None)
    result["max_normalized_span_variation"] = max((v[1] for v in normalized), default=None)
    result["normalized_threshold"] = 1.0e-3
    result["within_tolerance"] = bool(normalized) and all(max(v) <= 1.0e-3 for v in normalized)
    return result


def experiment_identity(run, reference):
    """Audit saved panel-free planar configuration and reference provenance.

    Returns the declared unit-speed/unit-diameter physical settings, coupling
    cadence and identity evidence; absent or incompatible settings are reported
    as unqualified rather than inferred from filenames."""
    path = run / "experiment.json"
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError):
        return {
            "verified": False,
            "reason": "experiment.json is absent or invalid",
            "path": str(path),
        }
    try:
        coupling_dt = float(manifest["coupling_dt"])
        if not np.isfinite(coupling_dt) or coupling_dt <= 0:
            coupling_dt = None
    except (KeyError, TypeError, ValueError):
        coupling_dt = None
    try:
        transfer_downstream = float(manifest["transfer_box"][1])
        if not np.isfinite(transfer_downstream):
            transfer_downstream = None
    except (KeyError, IndexError, TypeError, ValueError):
        transfer_downstream = None
    verified = (
        coupling_dt is not None
        and manifest.get("panel") is False
        and manifest.get("geometry") == "span-invariant cylinder Re150"
        and manifest.get("particle_strength_measure") == "omega_z*h^2*span"
        and manifest.get("span") == 1.0
        and (run / "samples").resolve() != reference.resolve()
    )
    return {
        "verified": bool(verified),
        "path": str(path),
        "reason": None
        if verified
        else "metadata does not identify a distinct panel-free planar experiment",
        "declared_geometry": manifest.get("geometry"),
        "declared_panel": manifest.get("panel"),
        "declared_strength_measure": manifest.get("particle_strength_measure"),
        "coupling_dt": coupling_dt,
        "transfer_downstream": transfer_downstream,
    }


def report(run, reference, start=80.0, end=None):
    """Build mature-cylinder qualification without changing the run or samples.

    Parameters
    ----------
    run : pathlib.Path
        Isolated coupled cylinder directory.
    reference : pathlib.Path
        Fully meshed reference sample directory.
    start : float
        Requested averaging start in seconds, default 80.
    end : float or None
        Requested end in seconds; None uses the common available endpoint.

    Returns
    -------
    dict
        Coverage, identity, force/profile agreement, complete-cycle frequency
        evidence, drift, and spanwise consistency. Short startup can expose
        provisional diagnostics but cannot pass the mature shedding gate.
        No phase fitting, filtering or temporal profile interpolation is used."""
    trial = run / "samples"
    if not (trial / "forces_history.csv").is_file():
        return {"status": "waiting_for_samples", "mature_gate_passed": False}
    left = final_segment(trial / "forces_history.csv")
    right = final_segment(reference / "forces_history.csv")
    common_end = min(float(left.time.max()), float(right.time.max()))
    if end is not None:
        common_end = min(common_end, end)
    common_start = max(start, float(left.time.min()), float(right.time.min()))
    if common_end - common_start < 1.0:
        return {
            "status": "before_mature_window",
            "latest_time": float(left.time.max()),
            "reference_latest_time": float(right.time.max()),
            "mature_gate_passed": False,
        }
    identity = experiment_identity(run, reference)
    dt = identity.get("coupling_dt")
    profile_dt = None if dt is None else 5.0 * dt
    evidence = {
        "forces": cadence_coverage(
            np.intersect1d(left.time, right.time), common_start, common_end, dt
        ),
        "profiles": {},
    }
    requested_window_covered = common_start <= start + 1.0e-8 and (
        end is None or common_end >= end - 1.0e-8
    )
    forces = force_comparison(trial, reference, common_start, common_end)
    profiles = {
        name: profile_comparison(trial, reference, name, common_start, common_end)
        for name in ("centreline", "transverse_x1", "transverse_x2", "transverse_x4")
    }
    for name in profiles:
        trial_times = final_segment(trial / f"fvm_{name}.csv").time.unique()
        reference_times = final_segment(reference / f"{name}.csv").time.unique()
        evidence["profiles"][name] = cadence_coverage(
            np.intersect1d(trial_times, reference_times), common_start, common_end, profile_dt
        )
    vpm_profiles, evidence["vpm_profiles"] = vpm_profile_evidence(
        trial, reference, common_start, common_end, profile_dt, identity.get("transfer_downstream")
    )
    cycles = min(
        forces[name]["cycle_frequency_lift"]["complete_periods"] for name in ("trial", "reference")
    )
    frequency_comparison = forces["frequency_comparison"]
    criteria = {
        "requested_statistics_window_covered": requested_window_covered,
        "complete_force_cadence": evidence["forces"]["complete"],
        "complete_profile_cadence": all(item["complete"] for item in evidence["profiles"].values()),
        "complete_vpm_profile_evidence": all(
            item.get("complete", False)
            and item.get("spatial_complete", False)
            and item.get("reference_support_complete", False)
            for item in evidence["vpm_profiles"].values()
        ),
        "vpm_mean_profiles_within_3pct_Uinf": all(
            item["available"] and item["mean_velocity_rms_Uinf"] <= 0.03
            for name, item in vpm_profiles.items()
            if name != "centreline_exterior"
        ),
        "vpm_exterior_centreline_within_3pct_Uinf": vpm_profiles["centreline_exterior"]["available"]
        and vpm_profiles["centreline_exterior"]["mean_velocity_rms_Uinf"] <= 0.03,
        "verified_panel_free_planar_experiment": identity["verified"],
        "at_least_eight_shedding_cycles": cycles is not None and cycles >= 8.0,
        "mean_drag_within_2pct": forces["relative_changes"]["mean_drag"] <= 0.02,
        "rms_lift_within_5pct": forces["relative_changes"]["rms_lift"] <= 0.05,
        "cycle_frequency_qualified": frequency_comparison["qualified"],
        "strouhal_within_3pct": frequency_comparison["qualified"]
        and frequency_comparison["conservative_relative_difference_bound"] <= 0.03,
        "mean_profiles_within_3pct_Uinf": all(
            p["mean_velocity_rms_Uinf"] <= 0.03 for p in profiles.values()
        ),
        "drag_half_window_drift_below_1pct": all(
            forces[name]["drag_drift_relative"] <= 0.01 for name in ("trial", "reference")
        ),
    }
    diagnostics = run / "solution/coupler_diagnostics.jsonl"
    rows = []
    if diagnostics.is_file():
        for line in diagnostics.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A concurrent writer may not yet have committed its final line.
            if rows and row["time"] < rows[-1]["time"] - 1.0e-8:
                rows = []
            rows.append(row)
    # Retain only the final restart segment and latest record at each time.
    rows = list(
        {
            round(float(row["time"]), 8): row
            for row in rows
            if common_start - 1.0e-8 <= row["time"] <= common_end + 1.0e-8
        }.values()
    )
    evidence["interfaces"] = cadence_coverage(
        [row["time"] for row in rows], common_start, common_end, dt
    )
    criteria["complete_interface_cadence"] = evidence["interfaces"]["complete"]
    criteria["all_interfaces_converged"] = bool(rows) and all(
        row["interface_iteration"]["converged"] for row in rows
    )
    evidence["full_donor_span"] = full_donor_planarity(rows, common_start, common_end, dt)
    criteria["complete_full_donor_span_evidence"] = evidence["full_donor_span"]["complete"]
    criteria["full_donor_spanwise_invariance_below_1e_3_scaled"] = evidence["full_donor_span"][
        "within_tolerance"
    ]
    span_path = trial / "span_probe.csv"
    variation = None
    span_times = []
    invalid_span_times = []
    if span_path.is_file():
        span = final_segment(span_path)
        span = span[(span.time >= common_start - 1.0e-8) & (span.time <= common_end + 1.0e-8)]
        span_times = span.time.unique()
        variations = []
        expected_z = np.linspace(-0.45, 0.45, 9)
        for time, group in span.groupby("time"):
            group = group.sort_values("position_z")
            values = group[
                ["position_x", "position_y", "position_z", "velocity_x", "velocity_y", "velocity_z"]
            ].to_numpy()
            valid = (
                len(group) == len(expected_z)
                and np.isfinite(values).all()
                and np.allclose(group.position_z, expected_z, atol=1.0e-8, rtol=0.0)
                and np.allclose(group.position_x, 1.5, atol=1.0e-8, rtol=0.0)
                and np.allclose(group.position_y, 0.0, atol=1.0e-8, rtol=0.0)
            )
            if not valid:
                invalid_span_times.append(float(time))
            else:
                variations.append(float(np.ptp(values[:, 3:], axis=0).max()))
        if variations:
            variation = max(variations)
    evidence["span"] = cadence_coverage(span_times, common_start, common_end, profile_dt)
    evidence["span"]["invalid_spatial_frames"] = len(invalid_span_times)
    evidence["span"]["first_invalid_frame_times"] = invalid_span_times[:10]
    criteria["complete_span_probe_evidence"] = (
        evidence["span"]["complete"] and not invalid_span_times
    )
    criteria["span_velocity_variation_below_1e_3"] = variation is not None and variation <= 1.0e-3
    return {
        "status": "mature_comparison"
        if cycles is not None and cycles >= 8
        else "provisional_insufficient_cycles",
        "mature_gate_passed": bool(all(criteria.values())),
        "criteria": {k: bool(v) for k, v in criteria.items()},
        "forces": forces,
        "profiles": profiles,
        "vpm_profiles": vpm_profiles,
        "shedding_cycles": cycles,
        "max_span_velocity_variation": variation,
        "full_donor_spanwise_consistency": evidence["full_donor_span"],
        "reference_samples": str(reference),
        "experiment_identity": identity,
        "evidence_coverage": evidence,
        "phase_adjustment": False,
        "limitation": "Engineering admission only; independent grid, VPM spacing, coupling-step and box-size sensitivity remain required. Short startup cannot establish shedding accuracy.",
    }


def plot_comparison(
    run: Path,
    reference: Path,
    destination: Path,
    *,
    snapshot_time: float | None = None,
    profile_prefix: str = "fvm_",
    figure_format: str = "png",
    history_end: float | None = None,
) -> dict:
    """Export raw histories and exact saved snapshots on fixed thesis canvases.

    Parameters
    ----------
    run, reference : pathlib.Path
        Coupled run directory and fully meshed reference samples directory.
    destination : pathlib.Path
        Force figure path; centreline and transverse companions share its stem.
    snapshot_time : float or None
        Exact saved time in seconds; None selects the latest common snapshot.
    profile_prefix : {"fvm_", "vpm_"}
        Coupled velocity component to compare with the FVM reference.
    figure_format : {"png", "pdf"}, default="png"
        Figure format with identical data and physical dimensions.
    history_end : float or None
        Optional end of the visible force interval in seconds.

    Returns
    -------
    dict
        Figure paths, visible coverage and available saved-profile metadata.
        Missing snapshots are labelled; no qualification values are changed.

    Raises
    ------
    ValueError, RuntimeError
        If inputs, the requested format, thesis fonts, or layout are invalid.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    from openonda.plotting import (
        CM,
        COLORS,
        DEFAULT_DPI,
        centered_subplots_adjust,
        figure_path,
        fit_thesis_y_label_margins,
        set_thesis_style,
        validate_thesis_figure,
    )

    set_thesis_style()
    destination = figure_path(destination, figure_format)
    destination.parent.mkdir(parents=True, exist_ok=True)
    trial = run / "samples"
    component = "VPM" if profile_prefix == "vpm_" else "FVM"
    force_paths = {
        "Trial": trial / "forces_history.csv",
        "Reference": reference / "forces_history.csv",
    }
    forces = {
        label: final_segment(path).drop_duplicates("time", keep="last")
        for label, path in force_paths.items()
        if path.is_file()
    }
    trial_time = forces["Trial"].time if "Trial" in forces else pd.Series(dtype=float)
    latest = float(trial_time.max()) if len(trial_time) else 0.0
    first = float(trial_time.min()) if len(trial_time) else 0.0
    if history_end is not None:
        latest = min(latest, float(history_end))
    startup = latest < 80.0
    names = ("centreline", "transverse_x1", "transverse_x2", "transverse_x4")
    profiles = {}
    shared_times = None
    for name in names:
        # The unprefixed fallback permits read-only plotting of native reference
        # samples without renaming their files or changing numerical schema.
        trial_path = trial / f"{profile_prefix}{name}.csv"
        if not trial_path.is_file() and profile_prefix == "fvm_":
            trial_path = trial / f"{name}.csv"
        reference_path = reference / f"{name}.csv"
        if not trial_path.is_file() or not reference_path.is_file():
            continue
        pair = (final_segment(trial_path), final_segment(reference_path))
        profiles[name] = pair
        times = np.intersect1d(pair[0].time.unique(), pair[1].time.unique())
        times = times[(times >= first) & (times <= latest)]
        shared_times = times if shared_times is None else np.intersect1d(shared_times, times)
    requested = None if snapshot_time is None else round(float(snapshot_time), 8)
    chosen = (
        (float(shared_times[-1]) if shared_times is not None and len(shared_times) else None)
        if requested is None
        else requested
    )
    paired_snapshot = chosen is not None and shared_times is not None and chosen in shared_times
    style = {
        "Trial": {"color": COLORS["VPMpurple"], "linestyle": "-"},
        "Reference": {"color": COLORS["RefGray"], "linestyle": "--"},
    }
    paths = []

    def save(figure, axes, suffix):
        """Measure final margins and save without recropping the physical canvas."""
        handles, labels = axes[-1].get_legend_handles_labels()
        if handles:
            figure.legend(
                handles,
                ["Coupled" if label == "Trial" else label for label in labels],
                loc="upper center",
                ncol=2,
                frameon=False,
            )
        for axis in axes:
            axis.grid(True)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        centered_subplots_adjust(
            figure,
            outer=0.18,
            top=1.0 - 1.8 * CM / figure.get_figheight(),
            bottom=1.35 * CM / figure.get_figheight(),
            hspace=0.8,
        )
        fit_thesis_y_label_margins(figure, axes)
        validate_thesis_figure(figure, axes)
        path = destination.with_stem(destination.stem + suffix)
        figure.savefig(path, dpi=DEFAULT_DPI, bbox_inches=None)
        plt.close(figure)
        paths.append(str(path.resolve()))

    figure, axes = plt.subplots(2, 1, figsize=(12.5 * CM, 9.4 * CM))
    for axis, column, symbol in zip(
        axes, ("drag_coefficient", "lift_coefficient"), (r"$C_D$", r"$C_L$"), strict=True
    ):
        for label, table in forces.items():
            visible = table.loc[(table.time >= first) & (table.time <= latest)]
            if len(visible) and column in visible:
                values = pd.to_numeric(visible[column], errors="raise").to_numpy(dtype=float)
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite {column} in plotted {label} history")
                axis.plot(visible.time, values, label=label, **style[label])
        axis.set(ylabel=symbol)
        if latest > first:
            axis.set_xlim(first, latest)
        if not axis.lines:
            axis.text(0.5, 0.5, "No force samples", ha="center", transform=axis.transAxes)
    axes[-1].set_xlabel(r"$t$ [s]")
    save(figure, axes, "")

    available = {}
    for suffix, selected_names, height in (
        ("_centreline", names[:1], 7.0),
        ("_transverse", names[1:], 13.5),
    ):
        figure, axes = plt.subplots(
            len(selected_names), 1, figsize=(12.5 * CM, height * CM), squeeze=False
        )
        axes = axes.ravel()
        for axis, name in zip(axes, selected_names, strict=True):
            coordinate = "position_x" if name == "centreline" else "position_y"
            title = r"$y/D=0$" if name == "centreline" else rf"$x/D={name[-1]}$"
            available[name] = bool(name in profiles and paired_snapshot)
            if not available[name]:
                axis.text(
                    0.5,
                    0.5,
                    "No paired saved snapshot",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
            else:
                for label, table in zip(("Trial", "Reference"), profiles[name], strict=True):
                    frame = (
                        table.loc[table.time == chosen]
                        .drop_duplicates(coordinate, keep="last")
                        .sort_values(coordinate)
                    )
                    x = pd.to_numeric(frame[coordinate], errors="raise").to_numpy(dtype=float)
                    u = pd.to_numeric(frame.velocity_x, errors="raise").to_numpy(dtype=float)
                    segments = (
                        (x < -0.5 - 1e-8, x > 0.5 + 1e-8)
                        if name == "centreline"
                        else (np.ones(len(x), dtype=bool),)
                    )
                    labelled = False
                    for selected in segments:
                        if not np.any(selected):
                            continue
                        if not np.isfinite(x[selected]).all() or not np.isfinite(u[selected]).all():
                            raise ValueError(f"Nonfinite fluid velocity in plotted {label} {name}")
                        axis.plot(
                            x[selected],
                            u[selected],
                            label=label if not labelled else None,
                            **style[label],
                        )
                        labelled = True
                title += rf", $t={chosen:g}$ s"
            if name == "centreline":
                axis.axvspan(-0.5, 0.5, color=COLORS["MaskGray"], zorder=0)
            axis.set(title=title, ylabel=r"$u_x/U_\infty$")
        axes[-1].set_xlabel(r"$x/D$" if suffix == "_centreline" else r"$y/D$")
        save(figure, axes, suffix)
    stage = (
        "Startup only; mature-flow qualification is incomplete."
        if startup
        else ("Available-data comparison; qualification is reported separately.")
    )
    caption = (
        f"Cylinder comparison. {stage} Raw, unshifted force histories cover "
        f"{first:g} to {latest:g} s. The centreline and transverse companions compare "
        f"coupled {component} velocity with the fully meshed reference at exact saved "
        f"time {chosen:g} s. "
        if paired_snapshot
        else f"Cylinder comparison. {stage} No paired saved velocity snapshot is available. "
    )
    caption += (
        "Velocities are normalized by freestream speed and coordinates by cylinder "
        "diameter. Solid centreline points are masked. No averaging, time interpolation, "
        "filtering, or phase alignment is applied. Missing force/profile series are "
        "not replaced. Figure styling does not change qualification.\n"
    )
    destination.with_suffix(".md").write_text(caption)
    return {
        "path": str(destination.resolve()),
        "companion_paths": paths[1:],
        "startup": bool(startup),
        "component": component,
        "visible_force_window": [first, latest],
        "snapshot_time": chosen if paired_snapshot else None,
        "requested_snapshot_time": requested,
        "available_profiles": available,
        "time_interpolation": False,
        "phase_adjustment": False,
        "qualification_changed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--reference", type=Path, default=REFERENCE / "samples/dense")
    parser.add_argument("--start", type=float, default=80.0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--plot",
        type=Path,
        metavar="FIGURE",
        help="Optional raw-history and same-time profile figure",
    )
    parser.add_argument(
        "--snapshot-time",
        type=float,
        help="Plot this exact saved profile time; default latest common time",
    )
    parser.add_argument(
        "--plot-vpm", type=Path, metavar="FIGURE", help="Optional VPM wake companion figure"
    )
    parser.add_argument("--format", choices=("png", "pdf"), default="png")
    args = parser.parse_args()
    result = report(args.run, args.reference, args.start, args.end)
    if args.plot:
        result["plot"] = plot_comparison(
            args.run,
            args.reference,
            args.plot,
            snapshot_time=args.snapshot_time,
            figure_format=args.format,
            history_end=args.end,
        )
    if args.plot_vpm:
        result["vpm_plot"] = plot_comparison(
            args.run,
            args.reference,
            args.plot_vpm,
            snapshot_time=args.snapshot_time,
            profile_prefix="vpm_",
            figure_format=args.format,
            history_end=args.end,
        )
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered)
