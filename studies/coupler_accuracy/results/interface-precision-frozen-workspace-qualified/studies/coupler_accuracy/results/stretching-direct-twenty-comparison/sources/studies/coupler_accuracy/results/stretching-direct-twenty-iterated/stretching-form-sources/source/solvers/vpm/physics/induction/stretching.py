"""Stretching contractions shared by accelerated induction backends."""

import taichi as ti


@ti.func
def stretching_rate(gradient, strength, mode):
    """Apply direct (0), transposed (1), or mixed (2) stretching."""
    rate = gradient @ strength
    if mode == 1:
        rate = gradient.transpose() @ strength
    elif mode == 2:
        rate = 0.5 * (gradient + gradient.transpose()) @ strength
    return rate
