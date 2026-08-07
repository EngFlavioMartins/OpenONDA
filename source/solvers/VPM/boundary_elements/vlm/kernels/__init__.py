"""
VLM kernel subpackage: Biot-Savart induced-velocity and collision kernels.

Wake shedding lives in ``vlm/solver/kernels.py``
(:func:`shed_wake_particles_kernel`), which is the single implementation: it
sheds the spanwise *difference* of cumulative circulation at each trailing-edge
edge, so the shed streamwise circulation telescopes to zero (Kelvin's theorem).

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from .biot_savart import (
    bound_vortex_velocity,
    horseshoe_semi_infinite_velocity,
    horseshoe_velocity,
    semi_infinite_vortex_velocity,
    vortex_ring_velocity,
)
from .collision import (
    detect_surface_collisions_kernel,
    is_point_in_quad,
)

__all__ = [
    "bound_vortex_velocity",
    "semi_infinite_vortex_velocity",
    "horseshoe_velocity",
    "horseshoe_semi_infinite_velocity",
    "vortex_ring_velocity",
    "detect_surface_collisions_kernel",
    "is_point_in_quad",
]
