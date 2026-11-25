"""Time-series statistics of the Part II runs: cycles, phase averages, harmonics, spectra, errors.

Every statistic is taken over a window of whole cycles. A cycle is one period of
the prescribed motion, or the interval between two zero up-crossings of a
reference signal (the cross-flow displacement of a free cylinder, the lift of a
fixed one). Standard errors come from the cycle-to-cycle scatter: the standard
deviation of the per-cycle values over the square root of the number of cycles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PHASE_BINS = 64
"""Bins of the phase average over one cycle."""
DIAGNOSTIC_PHASE_BINS = 16
"""Bins of the coupling diagnostics over the motion phase."""
HARMONICS = 5
"""Harmonics of the motion frequency projected on the window."""


class NoWholeCycle(ValueError):
    """The span holds no whole cycle, as in a run that stopped before its first."""


@dataclass(frozen=True)
class Window:
    """A statistics window of whole cycles.

    Attributes
    ----------
    edges : numpy.ndarray
        Cycle boundaries in s, ``n_cycles + 1`` increasing times.
    """

    edges: np.ndarray

    @property
    def start(self) -> float:
        return float(self.edges[0])

    @property
    def end(self) -> float:
        return float(self.edges[-1])

    @property
    def n_cycles(self) -> int:
        return len(self.edges) - 1

    def describe(self) -> dict[str, float | int]:
        periods = np.diff(self.edges)
        return {
            "start_s": self.start,
            "end_s": self.end,
            "n_cycles": self.n_cycles,
            "mean_period_s": float(periods.mean()),
            "period_standard_deviation_s": float(periods.std(ddof=1)) if len(periods) > 1 else 0.0,
        }


def motion_window(period: float, start_after: float, end: float, *, phase: float = 0.0) -> Window:
    """Whole periods of a prescribed motion ``sin(2 pi t / period + phase)`` within ``[start_after, end]``.

    The cycles start at the zero up-crossings of the motion.
    """
    first = np.ceil((start_after * 1.0 + phase * period / (2 * np.pi)) / period - 1.0e-9)
    origin = first * period - phase * period / (2 * np.pi)
    count = int(np.floor((end - origin) / period + 1.0e-9))
    if count < 1:
        raise NoWholeCycle(f"no whole period of {period:g} s lies in [{start_after:g}, {end:g}] s")
    return Window(origin + period * np.arange(count + 1))


def upcrossing_window(
    time: np.ndarray, signal: np.ndarray, start_after: float, end: float = np.inf
) -> Window:
    """Whole cycles between zero up-crossings of ``signal`` (mean removed) in ``[start_after, end]``.

    The mean is that of the samples in that span; crossing times are
    interpolated linearly between samples.
    """
    time = np.asarray(time, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)
    keep = (time >= start_after) & (time <= end)
    t, x = time[keep], signal[keep] - signal[keep].mean()
    rising = np.flatnonzero((x[:-1] < 0.0) & (x[1:] >= 0.0))
    crossings = t[rising] - x[rising] * (t[rising + 1] - t[rising]) / (x[rising + 1] - x[rising])
    if len(crossings) < 2:
        raise NoWholeCycle(f"fewer than two zero up-crossings after {start_after:g} s")
    return Window(crossings)


def cycle_index(time: np.ndarray, window: Window) -> np.ndarray:
    """Cycle number of every sample, ``-1`` outside the window."""
    index = np.searchsorted(window.edges, time, side="right") - 1
    index[(time < window.start) | (time >= window.end)] = -1
    return index


def phase_of(time: np.ndarray, window: Window) -> np.ndarray:
    """Phase in [0, 1) of every sample inside its cycle; NaN outside the window."""
    index = cycle_index(time, window)
    phase = np.full(len(time), np.nan)
    inside = index >= 0
    begin = window.edges[index[inside]]
    phase[inside] = (time[inside] - begin) / (window.edges[index[inside] + 1] - begin)
    return phase


def _resample_cycle(time, signal, begin, end, samples=PHASE_BINS * 8):
    """Signal at ``samples`` equal phases of one cycle, by linear interpolation."""
    phases = (np.arange(samples) + 0.5) / samples
    return np.interp(begin + phases * (end - begin), time, signal)


def cycle_statistics(time: np.ndarray, signal: np.ndarray, window: Window) -> dict[str, np.ndarray]:
    """Per-cycle mean, standard deviation, minimum and maximum of ``signal``.

    The cycle values come from the signal interpolated at equal phases, so a
    cycle does not weigh its samples by the local sampling density.
    """
    rows = {"mean": [], "std": [], "min": [], "max": []}
    for begin, end in zip(window.edges[:-1], window.edges[1:], strict=True):
        inside = (time >= begin) & (time <= end)
        values = _resample_cycle(time, signal, begin, end)
        rows["mean"].append(values.mean())
        rows["std"].append(values.std())
        # Extremes from the samples themselves, not the resampled curve.
        sampled = signal[inside] if np.any(inside) else values
        rows["min"].append(min(sampled.min(), values.min()))
        rows["max"].append(max(sampled.max(), values.max()))
    return {name: np.asarray(values) for name, values in rows.items()}


def standard_error(values: np.ndarray) -> float:
    """Standard error of the mean of per-cycle values: their standard deviation over sqrt(n)."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return float("nan")
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def cycle_frequency(window: Window) -> tuple[float, float]:
    """Frequency 1/mean period of the window's cycles and its standard error from the period scatter.

    The standard error of the mean period, divided by the mean period squared,
    is the first-order standard error of its inverse.
    """
    periods = np.diff(window.edges)
    mean = float(periods.mean())
    return 1.0 / mean, standard_error(periods) / mean**2


def window_statistics(time, signal, window: Window) -> dict[str, float]:
    """Mean, standard deviation and mean per-cycle extremes over the window, with standard errors.

    The standard error of the standard deviation is that of the per-cycle standard deviations.
    """
    cycles = cycle_statistics(time, signal, window)
    inside = (time >= window.start) & (time <= window.end)
    values = np.concatenate(
        [
            _resample_cycle(time, signal, begin, end)
            for begin, end in zip(window.edges[:-1], window.edges[1:], strict=True)
        ]
    )
    return {
        "mean": float(values.mean()),
        "mean_standard_error": standard_error(cycles["mean"]),
        "std": float(values.std()),
        "std_standard_error": standard_error(cycles["std"]),
        "cycle_max_mean": float(cycles["max"].mean()),
        "cycle_max_standard_error": standard_error(cycles["max"]),
        "cycle_min_mean": float(cycles["min"].mean()),
        "cycle_min_standard_error": standard_error(cycles["min"]),
        "max": float(signal[inside].max()),
        "min": float(signal[inside].min()),
    }


def phase_average(time, signal, window: Window, bins: int = PHASE_BINS) -> dict[str, np.ndarray]:
    """Phase average of ``signal`` in ``bins`` equal bins of the cycle, with the cycle-to-cycle std.

    Every cycle contributes the mean of its samples in each bin; the phase
    average is the mean over cycles and its scatter the standard deviation over
    cycles.
    """
    per_cycle = np.full((window.n_cycles, bins), np.nan)
    for cycle, (begin, end) in enumerate(zip(window.edges[:-1], window.edges[1:], strict=True)):
        values = _resample_cycle(time, signal, begin, end, samples=bins * 8)
        per_cycle[cycle] = values.reshape(bins, 8).mean(axis=1)
    return {
        "phase_centre": (np.arange(bins) + 0.5) / bins,
        "mean": per_cycle.mean(axis=0),
        "cycle_std": per_cycle.std(axis=0, ddof=1) if window.n_cycles > 1 else np.zeros(bins),
    }


def work_per_cycle(time, force, velocity, window: Window) -> np.ndarray:
    """Work in J of ``force`` on a body moving with ``velocity`` over each cycle, ∫ F·V dt.

    ``force`` and ``velocity`` have shape ``(n,)`` or ``(n, k)``; components are summed.
    """
    power = np.asarray(force, dtype=np.float64) * np.asarray(velocity, dtype=np.float64)
    if power.ndim == 2:
        power = power.sum(axis=1)
    work = []
    for begin, end in zip(window.edges[:-1], window.edges[1:], strict=True):
        grid = np.linspace(begin, end, PHASE_BINS * 16 + 1)
        work.append(np.trapezoid(np.interp(grid, time, power), grid))
    return np.asarray(work)


def harmonics(time, signal, window: Window, frequency: float, count: int = HARMONICS):
    """Amplitudes and phases of harmonics 1 to ``count`` of ``frequency`` by least squares on the window.

    The fit is ``a0 + sum_k A_k sin(2 pi k f t + phi_k)``: the phase is relative
    to the motion ``sin(2 pi f t)``. Returns ``(mean, amplitudes, phases_degrees,
    residual_rms)``.
    """
    inside = (time >= window.start) & (time <= window.end)
    t, x = time[inside], signal[inside]
    columns = [np.ones_like(t)]
    for k in range(1, count + 1):
        omega = 2.0 * np.pi * k * frequency
        columns += [np.sin(omega * t), np.cos(omega * t)]
    matrix = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(matrix, x, rcond=None)
    sine, cosine = coefficients[1::2], coefficients[2::2]
    amplitudes = np.hypot(sine, cosine)
    phases = np.degrees(np.arctan2(cosine, sine))
    residual = x - matrix @ coefficients
    return float(coefficients[0]), amplitudes, phases, float(np.sqrt(np.mean(residual**2)))


def uniform_samples(time, signal, window: Window):
    """The signal on a uniform grid of the median sample step over the window."""
    inside = (time >= window.start) & (time <= window.end)
    step = float(np.median(np.diff(time[inside])))
    grid = np.arange(window.start, window.end + 0.5 * step, step)
    return grid, np.interp(grid, time, signal), step


def welch_spectrum(time, signal, window: Window):
    """Power spectral density by Welch's method over the window.

    Hann window, segments of one quarter of the window length, 50 % overlap,
    the mean of each segment removed. Returns ``(frequency, psd, resolution)``
    with the resolution 1/segment duration in Hz.
    """
    from scipy.signal import welch

    grid, values, step = uniform_samples(time, signal, window)
    segment = max(8, len(values) // 4)
    frequency, psd = welch(
        values,
        fs=1.0 / step,
        window="hann",
        nperseg=segment,
        noverlap=segment // 2,
        detrend="constant",
        scaling="density",
    )
    return frequency, psd, 1.0 / (segment * step)


def cross_spectrum(time, first, second, window: Window):
    """Welch cross-spectral density of two signals, with the settings of :func:`welch_spectrum`."""
    from scipy.signal import csd

    grid, a, step = uniform_samples(time, first, window)
    _, b, _ = uniform_samples(time, second, window)
    segment = max(8, len(a) // 4)
    frequency, values = csd(
        a,
        b,
        fs=1.0 / step,
        window="hann",
        nperseg=segment,
        noverlap=segment // 2,
        detrend="constant",
        scaling="density",
    )
    return frequency, values


def peak_frequency(frequency: np.ndarray, psd: np.ndarray, *, minimum: float = 0.0) -> float:
    """Frequency of the largest PSD bin above ``minimum`` Hz, refined by three-point interpolation.

    The parabola through the logarithms of the peak bin and its two neighbours
    places the peak between bins.
    """
    usable = np.flatnonzero(frequency > minimum)
    index = int(usable[np.argmax(psd[usable])])
    if index == 0 or index == len(psd) - 1:
        return float(frequency[index])
    left, centre, right = np.log(psd[index - 1 : index + 2] + 1e-300)
    denominator = left - 2.0 * centre + right
    offset = 0.0 if denominator == 0.0 else 0.5 * (left - right) / denominator
    return float(frequency[index] + offset * (frequency[1] - frequency[0]))


def peak_amplitudes(time, signal, window: Window) -> dict[str, tuple[float, float]]:
    """Amplitude of an oscillation three ways: highest 20 % of the peaks, sqrt(2) std, maximum.

    Peaks are the per-cycle maxima of the signal less its window mean. Each
    amplitude comes with its standard error from the cycle scatter: that of the
    mean of the chosen peaks, sqrt(2) times that of the per-cycle standard
    deviations, and none (NaN) for the maximum, a single extreme.
    """
    cycles = cycle_statistics(time, signal, window)
    stats = window_statistics(time, signal, window)
    peaks = np.sort(cycles["max"] - stats["mean"])[::-1]
    top = peaks[: max(1, int(round(0.2 * len(peaks))))]
    inside = (time >= window.start) & (time <= window.end)
    return {
        "top_20_percent_peak_mean": (float(top.mean()), standard_error(top)),
        "sqrt2_std": (
            float(np.sqrt(2.0) * stats["std"]),
            float(np.sqrt(2.0) * stats["std_standard_error"]),
        ),
        "maximum": (float(np.max(np.abs(signal[inside] - stats["mean"]))), float("nan")),
    }


def block_bootstrap_interval(
    values: np.ndarray, *, block: int = 2, resamples: int = 4000, seed: int = 7
) -> tuple[float, float]:
    """95 % interval of the mean of per-cycle ``values`` by a moving-block bootstrap over cycles."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n < 2:
        return float("nan"), float("nan")
    block = min(block, n)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    # Every resample joins n_blocks blocks of consecutive cycles, cut to n cycles
    chosen = rng.integers(0, n - block + 1, size=(resamples, n_blocks))
    index = (chosen[:, :, None] + np.arange(block)).reshape(resamples, -1)[:, :n]
    means = values[index].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def percentiles(values: np.ndarray) -> dict[str, float]:
    """Median, 95th percentile and maximum of finite values."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"median": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(values.max()),
    }
