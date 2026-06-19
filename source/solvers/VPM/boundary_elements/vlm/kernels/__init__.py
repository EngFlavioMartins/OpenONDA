"""
Initialization module for kernels.
==================
Initialization module for kernels. module.

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
from .wake_shedding import (
    compute_wake_particles_kernel,
    compute_wake_simple_kernel,
)

__all__ = [
    "bound_vortex_velocity",
    "semi_infinite_vortex_velocity",
    "horseshoe_velocity",
    "horseshoe_semi_infinite_velocity",
    "vortex_ring_velocity",
    "compute_wake_particles_kernel",
    "compute_wake_simple_kernel",
    "detect_surface_collisions_kernel",
    "is_point_in_quad",
]
