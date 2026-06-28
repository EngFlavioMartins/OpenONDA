"""
Configuration subpackage: solver config dataclasses, constants, and backend setup.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

# CRITICAL: Initialize Taichi backend ONLY when Solver or VLMSolver are instantiated.
# This ensures user choices for precision (f32/f64) and backend (CPU/GPU) are respected.
