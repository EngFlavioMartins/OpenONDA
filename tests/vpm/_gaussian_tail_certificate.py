"""UNWIRED whole-tail and optional-completion mathematical certificates.

The whole-tail bound can qualify the exact same finite image set without
adding a completion field. It is a proposed explicit replacement criterion
for the existing heuristic block test, never an implicit bypass. Finite-image
approximation and arithmetic accuracy remain a separate obligation.
"""

from dataclasses import dataclass
import math

import numpy as np

from tests.vpm._gaussian_tail_arithmetic import add, div_positive, leading_tail, pairwise_sum, point
from tests.vpm._gaussian_tail_remainder_interval import tail_remainder


@dataclass(frozen=True)
class TailCertificate:
    whole_velocity: np.ndarray
    whole_gradient: np.ndarray
    completed_velocity: np.ndarray
    completed_gradient: np.ndarray
    diagnostics: dict


def _l1_upper(interval):
    upper = np.maximum(np.abs(interval.lower), np.abs(interval.upper))
    if not len(upper):
        return np.zeros(0)
    return pairwise_sum(point(upper.reshape(len(upper), -1).T)).upper


def tail_certificate(position, strength, radius, targets, *, z_min, z_max, shells,
                     prefix_terms=1024, max_sources=400_000, max_targets=400_000):
    """No field is published or modified; bounds cover infinite omitted images."""
    options = {"z_min": z_min, "z_max": z_max, "shells": shells,
               "max_sources": max_sources, "max_targets": max_targets}
    leading = leading_tail(position, strength, targets, prefix_terms=prefix_terms, **options)
    remainder = tail_remainder(position, strength, radius, targets, **options)
    whole_u = add(point(_l1_upper(leading.velocity_interval)), point(remainder.velocity)).upper
    whole_j = add(point(_l1_upper(leading.gradient_interval)), point(remainder.gradient)).upper
    completed_u = add(point(leading.velocity_error_bound), point(remainder.velocity)).upper
    completed_j = add(point(leading.gradient_error_bound), point(remainder.gradient)).upper
    diagnostics = {"shells": int(shells), "target_count": len(whole_u),
                   "norm_enclosure": "L1 vector/tensor norms conservatively bound Euclidean/Frobenius norms",
                   "whole_tail_scope": "exact unchanged finite +/-K image sum; no analytic completion added",
                   "completed_tail_scope": "finite image sum plus stored affine leading-tail completion",
                   "finite_image_accuracy_certified": False, "production_admissible": False,
                   "leading_arithmetic_velocity_max": float(leading.velocity_error_bound.max(initial=0)),
                   "leading_arithmetic_gradient_max": float(leading.gradient_error_bound.max(initial=0)),
                   "remainder": remainder.diagnostics}
    return TailCertificate(whole_u, whole_j, completed_u, completed_j, diagnostics)


def normalized_bound(velocity, gradient, *, velocity_scale, gradient_scale):
    """Outward normalization, without changing either user reference scale."""
    if not all(math.isfinite(v) and v > 0 for v in (velocity_scale, gradient_scale)):
        raise ValueError("positive finite normalization scales required")
    return np.maximum(div_positive(point(velocity), point(velocity_scale)).upper,
                      div_positive(point(gradient), point(gradient_scale)).upper)
