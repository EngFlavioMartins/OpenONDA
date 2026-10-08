"""Coefficients of the variable-step backward (BDF2) transient term.

The momentum assembly applies the implicit Euler and BDF2 transient terms;
this module supplies their nonuniform-step coefficients.
"""

import numpy as np


def backward_coefficients(
    time_step_size: float,
    previous_time_step_size: float,
) -> tuple[float, float, float]:
    r"""Return dimensionless variable-step BDF2 coefficients.

    For the current step ``dt`` and preceding accepted step ``dt0``, the
    derivative at the new time is

    ``(a0*u_new - a1*u_old + a2*u_older) / dt``

    with the nonuniform-grid backward coefficients used by OpenFOAM.  Equal
    steps reduce exactly to ``(3/2, 2, 1/2)``.
    """
    time_step_size = float(time_step_size)
    previous_time_step_size = float(previous_time_step_size)
    if not np.isfinite(time_step_size) or time_step_size <= 0.0:
        raise ValueError("time_step_size must be finite and positive")
    if not np.isfinite(previous_time_step_size) or previous_time_step_size <= 0.0:
        raise ValueError("previous_time_step_size must be finite and positive")
    total = time_step_size + previous_time_step_size
    coefficient_new = 1.0 + time_step_size / total
    coefficient_old = 1.0 + time_step_size / previous_time_step_size
    coefficient_older = time_step_size * time_step_size / (previous_time_step_size * total)
    return coefficient_new, coefficient_old, coefficient_older
