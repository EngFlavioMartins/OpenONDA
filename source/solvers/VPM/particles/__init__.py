"""
Initialization module for particles.
==================
Initialization module for particles. module.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from .container import Particles
from .distribution import ParticleDistributor


def create_physics(**kwargs):
    """
    Factory function to create the physics module.

    Args:
        **kwargs: Arguments passed to the physics constructor
                  - particles_kernel: Kernel type ('GAUSSIAN', etc.)
                  - precision: 'f32' or 'f64'

    Returns:
        PhysicsEngine instance (direct method / unbounded domain)
    """
    from ..physics.engine import PhysicsEngine

    return PhysicsEngine(**kwargs)


__all__ = ["Particles", "ParticleDistributor", "create_physics"]
