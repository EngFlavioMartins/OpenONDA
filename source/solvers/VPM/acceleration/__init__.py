"""
Acceleration module — fast multipole and spatial-index algorithms.
==================================================================
Provides O(N log N) velocity evaluation via Barnes-Hut treecodes (CPU and GPU)
and spatial hash-grid neighbour search.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from .treecode import BarnesHutTreecode, benchmark_treecode, compute_velocities_treecode
from .treecode_gpu import TaichiTreecode, compute_velocities_treecode_gpu

__all__ = [
    # CPU Barnes-Hut
    "BarnesHutTreecode",
    "compute_velocities_treecode",
    "benchmark_treecode",
    # GPU Taichi Barnes-Hut
    "TaichiTreecode",
    "compute_velocities_treecode_gpu",
]
