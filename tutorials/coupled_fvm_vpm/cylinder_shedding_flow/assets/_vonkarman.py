"""Von Karman instability analysis for the Re=150 cylinder shedding experiment.

Pure analysis library over the probe and force time series; no solver, no
matplotlib.  The central quantities are:

``normalized_transverse_velocity(t) = Uy(1.5D, 0, 0)/Uinf``  the primary instability observable, sampled
every 0.05 s by both the hybrid and the fully meshed reference.

``sigma``      linear growth rate of the envelope (ln A = ln initial_amplitude + sigma t)
``initial_amplitude``         initial antisymmetric amplitude extrapolated to t = 0
``St``         saturated Strouhal number (Welch/FFT peak, band-limited)
``A*``         fixed envelope threshold marking nonlinear onset
``t*``         first time the envelope reaches A*
``predicted_onset_time_shift``    predicted onset shift (1/sigma) ln(initial_amplitude,hyb / initial_amplitude,ref)
``measured_onset_time_shift``    measured onset shift t*_ref - t*_hyb

The hypothesis is supported only if growth rate and saturated frequency agree
between the cases while the hybrid carries a larger initial amplitude, so that
it reaches the same A* earlier by exactly predicted_onset_time_shift.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks, periodogram

# Shedding band at Re = 150 (St ~ 0.18 for a cylinder at this Re).
SHEDDING_BAND = (0.08, 0.30)
# Envelope level considered "well into nonlinear saturation".
ONSET_AMPLITUDE_THRESHOLD = 0.25
# Envelope level considered above the numerical start-up noise floor.
NOISE_FLOOR = 1e-3
# Shedding period tolerance for onset agreement (in periods).
ONSET_PERIODS = 0.25
# How many oscillation peaks the growth fit must span (one peak per period).
MIN_FIT_POINTS = 5


def resample_uniform(
    t: np.ndarray, normalized_transverse_velocity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return a uniformly sampled copy of (t, normalized_transverse_velocity) on the median delta-t grid."""
    time_step_size = float(np.median(np.diff(t)))
    if not np.isfinite(time_step_size) or time_step_size <= 0.0:
        return np.asarray(t, dtype=float), np.asarray(normalized_transverse_velocity, dtype=float)
    t_u = np.arange(t[0], t[-1] + 0.5 * time_step_size, time_step_size)
    return t_u, np.interp(t_u, t, normalized_transverse_velocity)


def peak_envelope(
    t: np.ndarray, normalized_transverse_velocity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Local maxima of ``|normalized_transverse_velocity|`` — one per shedding period.

    For an exponentially growing carrier the oscillation peaks sit exactly on
    the sine maxima, so their values grow as ``initial_amplitude exp(sigma t)``.  Picking
    peaks (rather than filtering and Hilbert-transforming) is immune to the
    record's enormous dynamic range, which corrupts a filtered envelope for a
    growing signal.
    """
    t = np.asarray(t, dtype=float)
    normalized_transverse_velocity = np.asarray(normalized_transverse_velocity, dtype=float)
    if t.size < 8:
        return t, np.abs(normalized_transverse_velocity)
    time_step_size = float(np.median(np.diff(t)))
    if not np.isfinite(time_step_size) or time_step_size <= 0.0:
        return t, np.abs(normalized_transverse_velocity)
    # Use |normalized_transverse_velocity| directly: the observable u_y at the y=0 midspan oscillates about
    # a zero baseline, so the oscillation peaks sit exactly on |normalized_transverse_velocity| and no mean
    # subtraction is needed (a record-wide mean is a spurious artifact of the
    # growing envelope and must not be removed here).
    amplitude = np.abs(normalized_transverse_velocity)
    # |normalized_transverse_velocity| peaks once per half period (at the sine extrema), so the minimum
    # separation is ~0.3 shedding periods.  Peaks below the noise floor are
    # oscillation troughs and spurious noise bumps, not envelope samples.
    period = 1.0 / (0.5 * (SHEDDING_BAND[0] + SHEDDING_BAND[1]))
    distance = max(int(round(0.3 * period / time_step_size)), 2)
    idx, _ = find_peaks(amplitude, distance=distance, height=NOISE_FLOOR)
    if idx.size < 4:
        return t, amplitude
    return t[idx], amplitude[idx]


def interpolated_envelope(
    t: np.ndarray, peak_time: np.ndarray, peak_amplitude: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth peak envelope on the sample grid (linear interpolation)."""
    t = np.asarray(t, dtype=float)
    peak_time = np.asarray(peak_time, dtype=float)
    peak_amplitude = np.asarray(peak_amplitude, dtype=float)
    if peak_time.size == 0:
        return t, np.zeros_like(t)
    if peak_time.size == 1:
        return t, np.full_like(t, float(peak_amplitude[0]))
    return t, np.interp(t, peak_time, peak_amplitude)


def envelope(
    t: np.ndarray, normalized_transverse_velocity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Full envelope pipeline: peaks, then interpolation onto the sample grid."""
    peak_time, peak_amplitude = peak_envelope(t, normalized_transverse_velocity)
    return interpolated_envelope(np.asarray(t, dtype=float), peak_time, peak_amplitude)


def growth_fit(
    peak_time: np.ndarray, peak_amplitude: np.ndarray, fit_start_time: float, fit_end_time: float
) -> dict:
    """Linear fit of ``ln(amplitude) = ln initial_amplitude + sigma t`` over [fit_start_time, fit_end_time]."""
    mask = (peak_time >= fit_start_time) & (peak_time <= fit_end_time)
    if np.count_nonzero(mask) < MIN_FIT_POINTS:
        return {
            "growth_rate": np.nan,
            "initial_amplitude": np.nan,
            "coefficient_of_determination": np.nan,
            "n_points": int(np.count_nonzero(mask)),
            "fit_start_time": float(fit_start_time),
            "fit_end_time": float(fit_end_time),
        }
    t = peak_time[mask]
    y = np.log(np.maximum(peak_amplitude[mask], 1e-12))
    coef, res, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(t), t]), y, rcond=None)
    log_initial_amplitude, sigma = coef
    y_pred = log_initial_amplitude + sigma * t
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    coefficient_of_determination = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else np.nan
    return {
        "growth_rate": float(sigma),
        "initial_amplitude": float(np.exp(log_initial_amplitude)),
        "coefficient_of_determination": float(coefficient_of_determination),
        "n_points": int(np.count_nonzero(mask)),
        "fit_start_time": float(fit_start_time),
        "fit_end_time": float(fit_end_time),
    }


def first_time(t: np.ndarray, values: np.ndarray, threshold: float) -> float:
    """First ``t`` with ``values >= threshold`` (NaN if never reached)."""
    idx = np.flatnonzero(np.asarray(values) >= threshold)
    return float(t[idx[0]]) if idx.size else float("nan")


def dominant_frequency(
    t: np.ndarray,
    normalized_transverse_velocity: np.ndarray,
    t_min: float,
    t_max: float,
    band=SHEDDING_BAND,
) -> tuple[float, float]:
    """Dominant frequency and band power of ``normalized_transverse_velocity`` in [t_min, t_max].

    Periodogram over the whole segment plus a parabolic interpolation of the
    log-spectrum around the peak, so the estimate is not quantised to FFT bins.
    """
    mask = (t >= t_min) & (t <= t_max)
    if np.count_nonzero(mask) < 32:
        return float("nan"), float("nan")
    seg = t[mask]
    if seg.size < 32 or not np.all(np.diff(seg) > 0):
        return float("nan"), float("nan")
    t_u, uniform_normalized_transverse_velocity = resample_uniform(
        seg, normalized_transverse_velocity[mask]
    )
    if t_u.size < 32:
        return float("nan"), float("nan")
    fs = 1.0 / float(np.median(np.diff(t_u)))
    qd = uniform_normalized_transverse_velocity - np.mean(uniform_normalized_transverse_velocity)
    nfft = max(int(2 ** np.ceil(np.log2(qd.size))) * 4, 64)
    freq, power = periodogram(qd, fs=fs, nfft=nfft)
    band_mask = (freq >= band[0]) & (freq <= band[1])
    if not np.any(band_mask):
        return float("nan"), float("nan")
    idx = np.flatnonzero(band_mask)
    i = idx[np.argmax(power[band_mask])]
    f0 = float(freq[i])
    if 1 <= i < freq.size - 1:
        y = np.log(np.maximum(power[i - 1 : i + 2], 1e-30))
        denom = y[0] - 2.0 * y[1] + y[2]
        if abs(denom) > 1e-30:
            delta = 0.5 * (y[0] - y[2]) / denom
            f0 = float(freq[i] + delta * (freq[i + 1] - freq[i]))
    shedding_band_power = float(np.sum(power[band_mask]))
    return f0, shedding_band_power


def saturated_stats(
    normalized_transverse_velocity_time: np.ndarray,
    normalized_transverse_velocity: np.ndarray,
    drag_coefficient_time: np.ndarray,
    drag_coefficient: np.ndarray,
    lift_coefficient_time: np.ndarray,
    lift_coefficient: np.ndarray,
    onset_time: float,
    margin: float = 5.0,
) -> dict:
    """Saturated (post-onset) statistics of normalized_transverse_velocity and force coefficients."""
    lo = onset_time + margin
    if not np.isfinite(lo):
        return {
            "mean_drag_coefficient": np.nan,
            "rms_lift_coefficient": np.nan,
            "rms_normalized_transverse_velocity": np.nan,
            "start_time": lo,
        }
    window_normalized_transverse_velocity = normalized_transverse_velocity[
        (normalized_transverse_velocity_time >= lo)
    ]
    drag_coefficient_window = drag_coefficient[drag_coefficient_time >= lo]
    lift_coefficient_window = lift_coefficient[lift_coefficient_time >= lo]
    return {
        "mean_drag_coefficient": (
            float(np.mean(drag_coefficient_window)) if drag_coefficient_window.size else np.nan
        ),
        "rms_lift_coefficient": (
            float(np.sqrt(np.mean(lift_coefficient_window**2)))
            if lift_coefficient_window.size
            else np.nan
        ),
        "rms_normalized_transverse_velocity": float(
            np.sqrt(np.mean(window_normalized_transverse_velocity**2))
        )
        if window_normalized_transverse_velocity.size
        else np.nan,
        "start_time": float(lo),
    }


def cross_correlation(
    t_ref: np.ndarray, q_ref: np.ndarray, t_hyb: np.ndarray, q_hyb: np.ndarray, t_win_min: float
) -> dict:
    """Correlation of the two probes in the common saturated window.

    A shift grid around the predicted onset offset is searched so phase
    alignment is not hostage to the exact onset estimate.
    """
    mask_ref = t_ref >= t_win_min
    mask_hyb = t_hyb >= t_win_min
    if np.count_nonzero(mask_ref) < 16 or np.count_nonzero(mask_hyb) < 16:
        return {"shift": np.nan, "correlation": np.nan}
    t_u, uniform_normalized_transverse_velocity = resample_uniform(t_ref[mask_ref], q_ref[mask_ref])
    t_v, q_v = resample_uniform(t_hyb[mask_hyb], q_hyb[mask_hyb])
    if t_u.size < 16 or t_v.size < 16:
        return {"shift": np.nan, "correlation": np.nan}
    valid = (t_u >= t_v[0]) & (t_u <= t_v[-1])
    if np.count_nonzero(valid) < 16:
        return {"shift": np.nan, "correlation": np.nan}
    t_common = t_u[valid]
    a = uniform_normalized_transverse_velocity[valid]
    b = np.interp(t_common, t_v, q_v)
    best = {"shift": 0.0, "correlation": float(np.corrcoef(a, b)[0, 1])}
    time_step_size = float(np.median(np.diff(t_u)))
    for shift in np.arange(-1.0, 1.0 + 1e-9, time_step_size):
        b_shift = np.interp(t_common + shift, t_v, q_v)
        if not np.all(np.isfinite(b_shift)):
            continue
        c = float(np.corrcoef(a, b_shift)[0, 1])
        if c > best["correlation"]:
            best = {"shift": shift, "correlation": c}
    return best


def envelope_modulation(amplitude: np.ndarray) -> float:
    """Coefficient of variation of the envelope: 0 is a clean monotone growth."""
    return float(np.std(amplitude) / np.mean(amplitude)) if np.mean(amplitude) > 0 else float("nan")


@dataclass
class Series:
    """One case's time series, pre-sorted."""

    label: str
    normalized_transverse_velocity_time: np.ndarray
    normalized_transverse_velocity: np.ndarray
    drag_coefficient_time: np.ndarray
    drag_coefficient: np.ndarray
    lift_coefficient_time: np.ndarray
    lift_coefficient: np.ndarray


@dataclass
class CaseResult:
    """All extracted quantities for one case."""

    label: str
    normalized_transverse_velocity_time: np.ndarray = field(default_factory=lambda: np.empty(0))
    normalized_transverse_velocity: np.ndarray = field(default_factory=lambda: np.empty(0))
    envelope_time: np.ndarray = field(default_factory=lambda: np.empty(0))
    amplitude: np.ndarray = field(default_factory=lambda: np.empty(0))
    onset_time: float = float("nan")
    growth: dict = field(default_factory=dict)
    growth_frequency: float = float("nan")
    strouhal_number: float = float("nan")
    shedding_band_power: float = float("nan")
    saturated: dict = field(default_factory=dict)
    modulation: float = float("nan")
    peak_time: np.ndarray = field(default_factory=lambda: np.empty(0))
    peak_amplitude: np.ndarray = field(default_factory=lambda: np.empty(0))


def analyse_series(
    series: Series, *, onset_amplitude_threshold: float = ONSET_AMPLITUDE_THRESHOLD
) -> CaseResult:
    """Full single-case extraction from probe + force histories."""
    peak_time, peak_amplitude = peak_envelope(
        series.normalized_transverse_velocity_time, series.normalized_transverse_velocity
    )
    envelope_time, amplitude = interpolated_envelope(
        series.normalized_transverse_velocity_time, peak_time, peak_amplitude
    )
    onset_time = first_time(envelope_time, amplitude, onset_amplitude_threshold)

    # Growth fit: from above the noise floor to onset.  The noise floor is a
    # fixed absolute level, not a per-case fitted value, so both cases use the
    # same criterion.
    fit_start_time = first_time(envelope_time, amplitude, NOISE_FLOOR)
    if not np.isfinite(fit_start_time):
        fit_start_time = float(envelope_time[0]) if envelope_time.size else 0.0
    fit_end_time = (
        onset_time
        if np.isfinite(onset_time)
        else float(envelope_time[-1])
        if envelope_time.size
        else 0.0
    )
    growth = growth_fit(peak_time, peak_amplitude, fit_start_time, fit_end_time)

    t_end = float(series.normalized_transverse_velocity_time[-1])
    # Saturated window: at least 20 s ending at the record end (always inside
    # the saturated regime, with identical treatment for both cases).
    sat_lo = max(onset_time + 5.0, t_end - 20.0) if np.isfinite(onset_time) else t_end - 20.0
    growth_frequency, _ = dominant_frequency(
        series.normalized_transverse_velocity_time,
        series.normalized_transverse_velocity,
        fit_start_time,
        fit_end_time,
    )
    strouhal_number, shedding_band_power = dominant_frequency(
        series.normalized_transverse_velocity_time,
        series.normalized_transverse_velocity,
        sat_lo,
        t_end,
    )
    saturated = saturated_stats(
        series.normalized_transverse_velocity_time,
        series.normalized_transverse_velocity,
        series.drag_coefficient_time,
        series.drag_coefficient,
        series.lift_coefficient_time,
        series.lift_coefficient,
        onset_time,
    )
    return CaseResult(
        label=series.label,
        normalized_transverse_velocity_time=series.normalized_transverse_velocity_time,
        normalized_transverse_velocity=series.normalized_transverse_velocity,
        envelope_time=envelope_time,
        amplitude=amplitude,
        onset_time=onset_time,
        growth=growth,
        growth_frequency=growth_frequency,
        strouhal_number=strouhal_number,
        shedding_band_power=shedding_band_power,
        saturated=saturated,
        modulation=envelope_modulation(amplitude),
        peak_time=peak_time,
        peak_amplitude=peak_amplitude,
    )


@dataclass
class Comparison:
    """Cross-case comparison and hypothesis verdict."""

    reference_growth_rate: float
    hybrid_growth_rate: float
    reference_strouhal_number: float
    hybrid_strouhal_number: float
    reference_initial_amplitude: float
    hybrid_initial_amplitude: float
    reference_onset_time: float
    hybrid_onset_time: float
    measured_onset_time_shift: float
    predicted_onset_time_shift: float
    shedding_period: float
    correlation: dict
    metrics: dict
    verdict: str
    flags: list = field(default_factory=list)


def compare(reference: CaseResult, hybrid: CaseResult, *, seed: bool = False) -> Comparison:
    """Compare the two cases and render the hypothesis verdict.

    ``seed=False`` tests the unseeded hypothesis: shared sigma/St with a larger
    hybrid initial_amplitude that pushes onset earlier by exactly ``predicted_onset_time_shift``.  ``seed=True``
    tests that equal controlled seeds collapse the onset offset (``measured_onset_time_shift ~ 0``)
    while sigma and St still match.
    """
    reference_growth_rate = reference.growth["growth_rate"]
    hybrid_growth_rate = hybrid.growth["growth_rate"]
    reference_strouhal_number = reference.strouhal_number
    hybrid_strouhal_number = hybrid.strouhal_number
    reference_initial_amplitude = reference.growth["initial_amplitude"]
    hybrid_initial_amplitude = hybrid.growth["initial_amplitude"]

    mean_growth_rate = 0.5 * (reference_growth_rate + hybrid_growth_rate)
    mean_strouhal_number = (
        0.5 * (reference_strouhal_number + hybrid_strouhal_number)
        if np.isfinite(reference_strouhal_number) and np.isfinite(hybrid_strouhal_number)
        else reference_strouhal_number
    )
    shedding_period = (
        1.0 / mean_strouhal_number
        if np.isfinite(mean_strouhal_number) and mean_strouhal_number > 0
        else np.nan
    )

    measured_onset_time_shift = (
        reference.onset_time - hybrid.onset_time
        if (np.isfinite(hybrid.onset_time) and np.isfinite(reference.onset_time))
        else np.nan
    )
    # Sign convention: predicted_onset_time_shift = t*_ref - t*_hyb and measured_onset_time_shift = t*_ref - t*_hyb,
    # both positive when the hybrid saturates first.
    predicted_onset_time_shift = (
        (1.0 / mean_growth_rate) * np.log(hybrid_initial_amplitude / reference_initial_amplitude)
        if (
            np.isfinite(mean_growth_rate)
            and mean_growth_rate > 0
            and reference_initial_amplitude > 0
            and hybrid_initial_amplitude > 0
        )
        else np.nan
    )

    corr = cross_correlation(
        reference.normalized_transverse_velocity_time,
        reference.normalized_transverse_velocity,
        hybrid.normalized_transverse_velocity_time,
        hybrid.normalized_transverse_velocity,
        min(reference.saturated["start_time"], hybrid.saturated["start_time"]),
    )

    tol = 0.25 * shedding_period if np.isfinite(shedding_period) else np.nan
    metrics = {
        "relative_growth_rate_difference": abs(hybrid_growth_rate - reference_growth_rate)
        / abs(reference_growth_rate)
        if reference_growth_rate
        else np.nan,
        "relative_strouhal_number_difference": abs(
            hybrid_strouhal_number - reference_strouhal_number
        )
        / abs(reference_strouhal_number)
        if reference_strouhal_number
        else np.nan,
        "initial_amplitude_ratio": hybrid_initial_amplitude / reference_initial_amplitude
        if reference_initial_amplitude > 0
        else np.nan,
        "onset_error_periods": abs(measured_onset_time_shift - predicted_onset_time_shift)
        / shedding_period
        if np.isfinite(shedding_period)
        else np.nan,
    }

    initial_amplitude_ratio = metrics["initial_amplitude_ratio"]
    essential = all(
        np.isfinite(metrics[name])
        for name in (
            "relative_growth_rate_difference",
            "relative_strouhal_number_difference",
            "onset_error_periods",
        )
    ) and np.isfinite(initial_amplitude_ratio)
    flags = []
    if metrics["relative_growth_rate_difference"] > 0.15:
        flags.append("growth rate differs materially")
    if metrics["relative_strouhal_number_difference"] > 0.05:
        flags.append("saturated frequency differs materially")
    if not seed and initial_amplitude_ratio <= 1.0:
        flags.append("hybrid initial amplitude is not above the reference")
    if not seed and metrics["onset_error_periods"] > ONSET_PERIODS:
        flags.append("measured onset shift does not match the amplitude prediction")
    if seed and np.isfinite(measured_onset_time_shift) and abs(measured_onset_time_shift) > tol:
        # With equal controlled seeds both cases should share the same initial_amplitude and
        # saturate simultaneously.  A residual onset gap falsifies the seed
        # collapsing the coupling offset.
        flags.append("equal seeds did not collapse the onset offset")

    if not essential:
        verdict = "incomplete"
    elif flags:
        verdict = "falsified"
    else:
        verdict = "supported"

    return Comparison(
        reference_growth_rate=reference_growth_rate,
        hybrid_growth_rate=hybrid_growth_rate,
        reference_strouhal_number=reference_strouhal_number,
        hybrid_strouhal_number=hybrid_strouhal_number,
        reference_initial_amplitude=reference_initial_amplitude,
        hybrid_initial_amplitude=hybrid_initial_amplitude,
        reference_onset_time=reference.onset_time,
        hybrid_onset_time=hybrid.onset_time,
        measured_onset_time_shift=measured_onset_time_shift,
        predicted_onset_time_shift=predicted_onset_time_shift,
        shedding_period=shedding_period,
        correlation=corr,
        metrics=metrics,
        verdict=verdict,
        flags=flags,
    )
