"""Roundoff tolerance of the solver clocks."""


def time_tolerance(time: float) -> float:
    """Return the tolerance in s within which two physical times are the same time.

    Times are sums of step sizes, so their roundoff grows with the time itself.
    """
    return max(1.0e-14, abs(float(time)) * 1.0e-12)
