"""
Regularization-kernel subpackage: Gaussian, high-order Gaussian, super-Gaussian,
and Winckelmans vortex-blob kernels.

Each kernel module defines its governing equations once, in a
``KernelDefinition``; ``registry`` lists the kernels the solver can run.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from . import gaussian, high_order_gaussian, super_gaussian, winckelmans
from .base import RadialVortexKernel, make_device_vortex_kernels, make_vortex_kernel
from .definition import KernelDefinition
from .registry import KERNELS, kernel_definition, kernel_names

__all__ = [
    "gaussian",
    "high_order_gaussian",
    "winckelmans",
    "super_gaussian",
    "KERNELS",
    "KernelDefinition",
    "RadialVortexKernel",
    "kernel_definition",
    "kernel_names",
    "make_device_vortex_kernels",
    "make_vortex_kernel",
]
