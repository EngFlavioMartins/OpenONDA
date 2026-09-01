"""Independent NumPy/f64 discrete-VPM oracle API."""
from .core import (gfun, gradient, invariants, pair_rate, q, reference_particle,
                   target_fields, velocity, zeta)

__all__ = ["gfun", "gradient", "invariants", "pair_rate", "q",
           "reference_particle", "target_fields", "velocity", "zeta"]
