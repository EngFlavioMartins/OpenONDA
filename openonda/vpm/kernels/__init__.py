"""
Regularization-kernel subpackage: Gaussian, high-order Gaussian, super-Gaussian,
and Winckelmans vortex-blob kernels.

Each kernel module defines its governing equations once, in a
``KernelDefinition``; ``registry`` lists the kernels the solver can run.
"""

from . import gaussian, high_order_gaussian, super_gaussian, winckelmans
from .base import RadialVortexKernel, make_device_vortex_kernels, make_vortex_kernel
from .definition import KernelDefinition
from .registry import KERNELS, kernel_definition, kernel_names

__all__ = [
    "KERNELS",
    "KernelDefinition",
    "RadialVortexKernel",
    "gaussian",
    "high_order_gaussian",
    "kernel_definition",
    "kernel_names",
    "make_device_vortex_kernels",
    "make_vortex_kernel",
    "super_gaussian",
    "winckelmans",
]
