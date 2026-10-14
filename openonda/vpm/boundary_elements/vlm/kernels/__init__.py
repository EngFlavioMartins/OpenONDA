"""
VLM kernel subpackage: Biot-Savart induced-velocity kernels.

Wake shedding lives in ``vlm/solver/kernels.py``
(:func:`shed_wake_particles_kernel`), which is the single implementation: it
sheds the spanwise *difference* of cumulative circulation at each trailing-edge
edge, so the shed streamwise circulation telescopes to zero (Kelvin's theorem).
"""

from .biot_savart import (
    bound_vortex_velocity,
    horseshoe_velocity,
)

__all__ = [
    "bound_vortex_velocity",
    "horseshoe_velocity",
]
