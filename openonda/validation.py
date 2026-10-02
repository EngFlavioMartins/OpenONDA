"""Source-preserving numerical comparisons for solver validation workflows."""

import numpy as np
from scipy.integrate import trapezoid


def time_mean(times, values, start, end):
    """Integrate a fully bracketed history, interpolating only window endpoints.

    Supports scalar or vector records. Missing records, non-finite values and
    unbracketed windows are errors; a shorter run is never silently averaged.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if (
        times.ndim != 1
        or len(times) < 2
        or values.ndim < 1
        or values.shape[0] != len(times)
        or not np.isfinite(times).all()
        or not np.isfinite(values).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("time mean requires finite values on a strictly increasing clock")
    if not np.isfinite([start, end]).all() or end <= start:
        raise ValueError("time mean requires a finite ordered window")
    if times[0] > start or times[-1] < end:
        raise ValueError("history does not bracket the complete averaging window")
    clock = np.r_[start, times[(times > start) & (times < end)], end]
    flat = values.reshape(len(times), -1)
    sampled = np.column_stack([np.interp(clock, times, v) for v in flat.T])
    return (trapezoid(sampled, clock, axis=0) / (end - start)).reshape(values.shape[1:])


def profile_error(actual, reference, floor):
    """Scaled RMS over every finite bin; the floor scales weak signals, not masks them."""
    actual, reference = np.asarray(actual, float), np.asarray(reference, float)
    if actual.shape != reference.shape or not np.isfinite(floor) or floor <= 0:
        raise ValueError("profile comparison requires matching shapes and a positive floor")
    valid = np.isfinite(actual) & np.isfinite(reference)
    if not np.any(valid):
        return float("nan")
    return float(
        np.sqrt(
            np.mean(
                ((actual[valid] - reference[valid]) / np.maximum(np.abs(reference[valid]), floor))
                ** 2
            )
        )
    )
