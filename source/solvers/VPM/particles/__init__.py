"""
Particle subpackage: the Particles container and ParticleDistributor seeding.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from .container import Particles
from .distribution import ParticleDistributor

__all__ = ["Particles", "ParticleDistributor"]
