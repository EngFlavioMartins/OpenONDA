"""Von Karman instability analysis for the Re=150 cylinder shedding experiment.

Pure analysis library over the probe and force time series; no solver, no
matplotlib.  The central quantities are:

``q(t) = Uy(1.5D, 0, 0)/Uinf``  the primary instability observable, sampled
every 0.05 s by both the hybrid and the fully meshed reference.

``sigma``      linear growth rate of the envelope (ln A = ln A0 + sigma t)
``A0``         initial antisymmetric amplitude extrapolated to t = 0
``St``         saturated Strouhal number (Welch/FFT peak, band-limited)
``A*``         fixed envelope threshold marking nonlinear onset
``t*``         first time the envelope reaches A*
``dt_pred``    predicted onset shift (1/sigma) ln(A0,hyb / A0,ref)
``dt_meas``    measured onset shift t*_ref - t*_hyb

The hypothesis is supported only if growth rate and saturated frequency agree
between the cases while the hybrid carries a larger initial amplitude, so that
it reaches the same A* earlier by exactly dt_pred.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks, periodogram

# Shedding band at Re = 150 (St ~ 0.18 for a cylinder at this Re).
SHEDDING_BAND = (0.08, 0.30)
# Envelope level considered "well into nonlinear saturation".
A_STAR = 0.25
# Envelope level considered above the numerical/IBM noise floor.
NOISE_FLOOR = 1e-3
# Shedding period tolerance for onset agreement (in periods).
ONSET_PERIODS = 0.25
# How many oscillation peaks the growth fit must span (one peak per period).
MIN_FIT_POINTS = 5


def resample_uniform(t: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a uniformly sampled copy of (t, q) on the median delta-t grid."""
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0.0:
        return np.asarray(t, dtype=float), np.asarray(q, dtype=float)
    t_u = np.arange(t[0], t[-1] + 0.5 * dt, dt)
    return t_u, np.interp(t_u, t, q)


def peak_envelope(t: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Local maxima of ``|q|`` — one per shedding period.

    For an exponentially growing carrier the oscillation peaks sit exactly on
    the sine maxima, so their values grow as ``A0 exp(sigma t)``.  Picking
    peaks (rather than filtering and Hilbert-transforming) is immune to the
    record's enormous dynamic range, which corrupts a filtered envelope for a
    growing signal.
    """
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    if t.size < 8:
        return t, np.abs(q)
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0.0:
        return t, np.abs(q)
    # Use |q| directly: the observable u_y at the y=0 midspan oscillates about
    # a zero baseline, so the oscillation peaks sit exactly on |q| and no mean
    # subtraction is needed (a record-wide mean is a spurious artifact of the
    # growing envelope and must not be removed here).
    amp = np.abs(q)
    # |q| peaks once per half period (at the sine extrema), so the minimum
    # separation is ~0.3 shedding periods.  Peaks below the noise floor are
    # oscillation troughs and spurious noise bumps, not envelope samples.
    period = 1.0 / (0.5 * (SHEDDING_BAND[0] + SHEDDING_BAND[1]))
    distance = max(int(round(0.3 * period / dt)), 2)
    idx, _ = find_peaks(amp, distance=distance, height=NOISE_FLOOR)
    if idx.size < 4:
        return t, amp
    return t[idx], amp[idx]


def interpolated_envelope(
    t: np.ndarray, t_peaks: np.ndarray, amp_peaks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth peak envelope on the sample grid (linear interpolation)."""
    t = np.asarray(t, dtype=float)
    t_peaks = np.asarray(t_peaks, dtype=float)
    amp_peaks = np.asarray(amp_peaks, dtype=float)
    if t_peaks.size == 0:
        return t, np.zeros_like(t)
    if t_peaks.size == 1:
        return t, np.full_like(t, float(amp_peaks[0]))
    return t, np.interp(t, t_peaks, amp_peaks)


def envelope(t: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full envelope pipeline: peaks, then interpolation onto the sample grid."""
    t_peaks, amp_peaks = peak_envelope(t, q)
    return interpolated_envelope(np.asarray(t, dtype=float), t_peaks, amp_peaks)


def growth_fit(
    t_peaks: np.ndarray, amp_peaks: np.ndarray, t_fit_start: float, t_fit_end: float
) -> dict:
    """Linear fit of ``ln(amp) = ln A0 + sigma t`` over [t_fit_start, t_fit_end]."""
    mask = (t_peaks >= t_fit_start) & (t_peaks <= t_fit_end)
    if np.count_nonzero(mask) < MIN_FIT_POINTS:
        return {
            "sigma": np.nan,
            "A0": np.nan,
            "r2": np.nan,
            "n_points": int(np.count_nonzero(mask)),
            "t_fit_start": float(t_fit_start),
            "t_fit_end": float(t_fit_end),
        }
    t = t_peaks[mask]
    y = np.log(np.maximum(amp_peaks[mask], 1e-12))
    coef, res, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(t), t]), y, rcond=None)
    ln_A0, sigma = coef
    y_pred = ln_A0 + sigma * t
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else np.nan
    return {
        "sigma": float(sigma),
        "A0": float(np.exp(ln_A0)),
        "r2": float(r2),
        "n_points": int(np.count_nonzero(mask)),
        "t_fit_start": float(t_fit_start),
        "t_fit_end": float(t_fit_end),
    }


def first_time(t: np.ndarray, values: np.ndarray, threshold: float) -> float:
    """First ``t`` with ``values >= threshold`` (NaN if never reached)."""
    idx = np.flatnonzero(np.asarray(values) >= threshold)
    return float(t[idx[0]]) if idx.size else float("nan")


def dominant_frequency(
    t: np.ndarray, q: np.ndarray, t_min: float, t_max: float, band=SHEDDING_BAND
) -> tuple[float, float]:
    """Dominant frequency and band power of ``q`` in [t_min, t_max].

    Periodogram over the whole segment plus a parabolic interpolation of the
    log-spectrum around the peak, so the estimate is not quantised to FFT bins.
    """
    mask = (t >= t_min) & (t <= t_max)
    if np.count_nonzero(mask) < 32:
        return float("nan"), float("nan")
    seg = t[mask]
    if seg.size < 32 or not np.all(np.diff(seg) > 0):
        return float("nan"), float("nan")
    t_u, q_u = resample_uniform(seg, q[mask])
    if t_u.size < 32:
        return float("nan"), float("nan")
    fs = 1.0 / float(np.median(np.diff(t_u)))
    qd = q_u - np.mean(q_u)
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
    power_band = float(np.sum(power[band_mask]))
    return f0, power_band


def saturated_stats(
    t_q: np.ndarray,
    q: np.ndarray,
    t_cd: np.ndarray,
    cd: np.ndarray,
    t_cl: np.ndarray,
    cl: np.ndarray,
    t_onset: float,
    margin: float = 5.0,
) -> dict:
    """Saturated (post-onset) statistics of q, Cd and Cl."""
    lo = t_onset + margin
    if not np.isfinite(lo):
        return {"mean_Cd": np.nan, "rms_Cl": np.nan, "rms_q": np.nan, "t_start": lo}
    q_win = q[(t_q >= lo)]
    cd_win = cd[(t_cd >= lo)]
    cl_win = cl[(t_cl >= lo)]
    return {
        "mean_Cd": float(np.mean(cd_win)) if cd_win.size else np.nan,
        "rms_Cl": float(np.sqrt(np.mean(cl_win**2))) if cl_win.size else np.nan,
        "rms_q": float(np.sqrt(np.mean(q_win**2))) if q_win.size else np.nan,
        "t_start": float(lo),
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
    t_u, q_u = resample_uniform(t_ref[mask_ref], q_ref[mask_ref])
    t_v, q_v = resample_uniform(t_hyb[mask_hyb], q_hyb[mask_hyb])
    if t_u.size < 16 or t_v.size < 16:
        return {"shift": np.nan, "correlation": np.nan}
    valid = (t_u >= t_v[0]) & (t_u <= t_v[-1])
    if np.count_nonzero(valid) < 16:
        return {"shift": np.nan, "correlation": np.nan}
    t_common = t_u[valid]
    a = q_u[valid]
    b = np.interp(t_common, t_v, q_v)
    best = {"shift": 0.0, "correlation": float(np.corrcoef(a, b)[0, 1])}
    dt = float(np.median(np.diff(t_u)))
    for shift in np.arange(-1.0, 1.0 + 1e-9, dt):
        b_shift = np.interp(t_common + shift, t_v, q_v)
        if not np.all(np.isfinite(b_shift)):
            continue
        c = float(np.corrcoef(a, b_shift)[0, 1])
        if c > best["correlation"]:
            best = {"shift": shift, "correlation": c}
    return best


def envelope_modulation(amp: np.ndarray) -> float:
    """Coefficient of variation of the envelope: 0 is a clean monotone growth."""
    return float(np.std(amp) / np.mean(amp)) if np.mean(amp) > 0 else float("nan")


@dataclass
class Series:
    """One case's time series, pre-sorted."""

    label: str
    t_q: np.ndarray
    q: np.ndarray
    t_cd: np.ndarray
    cd: np.ndarray
    t_cl: np.ndarray
    cl: np.ndarray


@dataclass
class CaseResult:
    """All extracted quantities for one case."""

    label: str
    t_q: np.ndarray = field(default_factory=lambda: np.empty(0))
    q: np.ndarray = field(default_factory=lambda: np.empty(0))
    t_env: np.ndarray = field(default_factory=lambda: np.empty(0))
    amp: np.ndarray = field(default_factory=lambda: np.empty(0))
    t_onset: float = float("nan")
    growth: dict = field(default_factory=dict)
    f_growth: float = float("nan")
    st: float = float("nan")
    power_band: float = float("nan")
    saturated: dict = field(default_factory=dict)
    modulation: float = float("nan")
    t_peaks: np.ndarray = field(default_factory=lambda: np.empty(0))
    amp_peaks: np.ndarray = field(default_factory=lambda: np.empty(0))


def analyse_series(series: Series, *, a_star: float = A_STAR) -> CaseResult:
    """Full single-case extraction from probe + force histories."""
    t_peaks, amp_peaks = peak_envelope(series.t_q, series.q)
    t_env, amp = interpolated_envelope(series.t_q, t_peaks, amp_peaks)
    t_onset = first_time(t_env, amp, a_star)

    # Growth fit: from above the noise floor to onset.  The noise floor is a
    # fixed absolute level, not a per-case fitted value, so both cases use the
    # same criterion.
    t_fit_start = first_time(t_env, amp, NOISE_FLOOR)
    if not np.isfinite(t_fit_start):
        t_fit_start = float(t_env[0]) if t_env.size else 0.0
    t_fit_end = t_onset if np.isfinite(t_onset) else float(t_env[-1]) if t_env.size else 0.0
    growth = growth_fit(t_peaks, amp_peaks, t_fit_start, t_fit_end)

    t_end = float(series.t_q[-1])
    # Saturated window: at least 20 s ending at the record end (always inside
    # the saturated regime, with identical treatment for both cases).
    sat_lo = max(t_onset + 5.0, t_end - 20.0) if np.isfinite(t_onset) else t_end - 20.0
    f_growth, _ = dominant_frequency(series.t_q, series.q, t_fit_start, t_fit_end)
    st, power_band = dominant_frequency(series.t_q, series.q, sat_lo, t_end)
    saturated = saturated_stats(
        series.t_q,
        series.q,
        series.t_cd,
        series.cd,
        series.t_cl,
        series.cl,
        t_onset,
    )
    return CaseResult(
        label=series.label,
        t_q=series.t_q,
        q=series.q,
        t_env=t_env,
        amp=amp,
        t_onset=t_onset,
        growth=growth,
        f_growth=f_growth,
        st=st,
        power_band=power_band,
        saturated=saturated,
        modulation=envelope_modulation(amp),
        t_peaks=t_peaks,
        amp_peaks=amp_peaks,
    )


@dataclass
class Comparison:
    """Cross-case comparison and hypothesis verdict."""

    sigma_ref: float
    sigma_hyb: float
    st_ref: float
    st_hyb: float
    a0_ref: float
    a0_hyb: float
    t_onset_ref: float
    t_onset_hyb: float
    dt_meas: float
    dt_pred: float
    shedding_period: float
    correlation: dict
    metrics: dict
    verdict: str
    flags: list = field(default_factory=list)


def compare(reference: CaseResult, hybrid: CaseResult, *, seed: bool = False) -> Comparison:
    """Compare the two cases and render the hypothesis verdict.

    ``seed=False`` tests the unseeded hypothesis: shared sigma/St with a larger
    hybrid A0 that pushes onset earlier by exactly ``dt_pred``.  ``seed=True``
    tests that equal controlled seeds collapse the onset offset (``dt_meas ~ 0``)
    while sigma and St still match.
    """
    sigma_ref = reference.growth["sigma"]
    sigma_hyb = hybrid.growth["sigma"]
    st_ref = reference.st
    st_hyb = hybrid.st
    a0_ref = reference.growth["A0"]
    a0_hyb = hybrid.growth["A0"]

    sigma_bar = 0.5 * (sigma_ref + sigma_hyb)
    st_bar = 0.5 * (st_ref + st_hyb) if np.isfinite(st_ref) and np.isfinite(st_hyb) else st_ref
    shedding_period = 1.0 / st_bar if np.isfinite(st_bar) and st_bar > 0 else np.nan

    dt_meas = (
        reference.t_onset - hybrid.t_onset
        if (np.isfinite(hybrid.t_onset) and np.isfinite(reference.t_onset))
        else np.nan
    )
    # Sign convention: dt_pred = t*_ref - t*_hyb and dt_meas = t*_ref - t*_hyb,
    # both positive when the hybrid saturates first.
    dt_pred = (
        (1.0 / sigma_bar) * np.log(a0_hyb / a0_ref)
        if (np.isfinite(sigma_bar) and sigma_bar > 0 and a0_ref > 0 and a0_hyb > 0)
        else np.nan
    )

    corr = cross_correlation(
        reference.t_q,
        reference.q,
        hybrid.t_q,
        hybrid.q,
        min(reference.saturated["t_start"], hybrid.saturated["t_start"]),
    )

    tol = 0.25 * shedding_period if np.isfinite(shedding_period) else np.nan
    metrics = {
        "rel_sigma": abs(sigma_hyb - sigma_ref) / abs(sigma_ref) if sigma_ref else np.nan,
        "rel_st": abs(st_hyb - st_ref) / abs(st_ref) if st_ref else np.nan,
        "a0_ratio": a0_hyb / a0_ref if a0_ref > 0 else np.nan,
        "onset_error_periods": abs(dt_meas - dt_pred) / shedding_period
        if np.isfinite(shedding_period)
        else np.nan,
    }

    a0_ratio = metrics["a0_ratio"]
    essential = all(
        np.isfinite(metrics[name]) for name in ("rel_sigma", "rel_st", "onset_error_periods")
    ) and np.isfinite(a0_ratio)
    flags = []
    if metrics["rel_sigma"] > 0.15:
        flags.append("growth rate differs materially")
    if metrics["rel_st"] > 0.05:
        flags.append("saturated frequency differs materially")
    if not seed and a0_ratio <= 1.0:
        flags.append("hybrid initial amplitude is not above the reference")
    if not seed and metrics["onset_error_periods"] > ONSET_PERIODS:
        flags.append("measured onset shift does not match the amplitude prediction")
    if seed and np.isfinite(dt_meas) and abs(dt_meas) > tol:
        # With equal controlled seeds both cases should share the same A0 and
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
        sigma_ref=sigma_ref,
        sigma_hyb=sigma_hyb,
        st_ref=st_ref,
        st_hyb=st_hyb,
        a0_ref=a0_ref,
        a0_hyb=a0_hyb,
        t_onset_ref=reference.t_onset,
        t_onset_hyb=hybrid.t_onset,
        dt_meas=dt_meas,
        dt_pred=dt_pred,
        shedding_period=shedding_period,
        correlation=corr,
        metrics=metrics,
        verdict=verdict,
        flags=flags,
    )
