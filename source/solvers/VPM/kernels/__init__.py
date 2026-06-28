"""
Regularization-kernel subpackage: Gaussian, high-order Gaussian, super-Gaussian,
and Winckelmans vortex-blob kernels.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from . import gaussian, high_order_gaussian, super_gaussian, winckelmans

__all__ = [
    "gaussian",
    "high_order_gaussian",
    "winckelmans",
    "super_gaussian",
]
