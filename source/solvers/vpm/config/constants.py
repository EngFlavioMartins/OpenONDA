"""
Physical and numerical constants for the VPM solver (e.g. MAX_N_PARTICLES,
MAX_SOURCES), with helpers to list and print them.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import numpy as np

# =========================================================
# NUMERICAL CONSTANTS
# =========================================================

# Numerical epsilon for stability and convergence checks
EPSILON = 1e-12

# Small value to prevent division by zero
SMALL_VALUE = 1e-15

# Mathematical constants
PI = np.pi
TWO_PI = 2.0 * PI
FOUR_PI = 4.0 * PI
SQRT_PI = PI**0.5

ONE_OVER_SQRT2 = 1.0 / (2.0**0.5)
ONE_OVER_FOUR_PI = 1.0 / (4.0 * PI)
THREE_OVER_FOUR_PI = 3.0 / (4.0 * PI)
SQRT_2_OVER_PI = (2.0 / PI) ** 0.5

ONE_OVER_TWO_PI_POW_1p15 = 1.0 / ((2.0 * PI) ** 1.5)
ONE_OVER_TWO_PI_POW_1p05 = 1.0 / ((2.0 * PI) ** 0.5)
INV_PI32 = 1.0 / (PI**1.5)
TWO_OVER_SQRT_PI = 2.0 / SQRT_PI

# =========================================================
# GLOBAL FLOATING POINT PRECISION
# =========================================================
# These constants enforce consistent float32 precision across the codebase.
# Using f32 provides good accuracy for VPM while enabling faster SIMD and GPU operations.

import taichi as ti

NP_FLOAT = np.float32  # NumPy float type (for all numpy arrays)
TI_FLOAT = ti.f32  # Taichi float type (for all Taichi fields)

# =========================================================
# PERIODIC BOX PROPERTIES
# =========================================================
BOX_SIZE = 2 * PI
BOX_MAX = 2 * PI
BOX_MIN = 0.0

# =========================================================
# SYSTEM LIMITS AND CAPACITIES
# =========================================================
# Under-relaxation factor for iterative methods
ALPHA = 0.4

# Maximum number of particles (buffer size)
MAX_N_PARTICLES = 500000
MAX_SOURCES = 50000  # Surface sources for blockage correction
MAX_TARGETS = 100000  # Target evaluation points (e.g. boundary mesh)
MAX_RETRY_ATTEMPTS = 5

# Maximum number of iterations for iterative solvers
MAX_ITERATIONS = 5

# Default buffer size for dynamic arrays
DEFAULT_BUFFER_SIZE = 1024

# =========================================================
# SOLVER DEFAULT PARAMETERS
# =========================================================

# DVH truncation-error parameter β ≈ 0.077 (Durante et al. 2024, Eq. 14-15).
# Controls the Gaussian width of the heat-kernel scatter: 4nu·Δt_d = β·R_d².
_DVH_BETA = 0.077

# Default time step size [s]
DEFAULT_TIME_STEP = 0.01

# Default cutoff radius factor for particle interactions
DEFAULT_CUTOFF_RADIUS_FACTOR = 100

# Default core radius [m]
DEFAULT_CORE_RADIUS = 0.05

# Default particle volume [m³]
DEFAULT_PARTICLE_VOLUME = 1e-3

# -- Gaussian q-kernel small-argument series --------------------------------
# q(r) = [erf(r) - (2/sqrt(pi)) r exp(-r^2)] / (4 pi) is a difference of two
# quantities that both tend to (2/sqrt(pi)) r while their difference is O(r^3),
# so the closed form loses all significance for small r (relative error
# ~1.5 eps / r^2 in f32: 5.6e-2 at r = 1e-2, 7e+4 at r = 1e-4, sign flips).
# Below GAUSSIAN_Q_SERIES_CROSSOVER both the direct kernel
# (kernels/gaussian.py) and the treecode's copy (acceleration/treecode_gpu.py)
# must use the series
#     q(r) = C r^3 [1 - (3/5) r^2 + (3/14) r^4] / (4 pi) + O(r^9),  C = 4/(3 sqrt(pi)).
# Measured f32 accuracy: series <= 4e-6 for r <= 0.2; closed form ~2e-5 at
# r = 0.2 and improving, so 0.2 sits at/below the crossover.
FOUR_OVER_THREE_SQRT_PI = 0.7522527780636751
GAUSSIAN_Q_SERIES_CROSSOVER = 0.2

# Regularization kernels the LBVH treecode can evaluate.  The treecode carries
# its own Taichi implementation of q/zeta (see acceleration/treecode_gpu.py), so
# it supports a strict subset of the kernels available to the direct O(N²) path.
# Single source of truth: Numerics validation and TaichiTreecode.set_kernel_type
# both read this, so the two cannot drift apart.
TREECODE_SUPPORTED_KERNELS = ("GAUSSIAN", "WINCKELMANS")

# =========================================================
# TURBULENCE MODEL CONSTANTS
# =========================================================

# Default VPM Smagorinsky constant C_s.  The vortex-ring stability calibration
# qualified C_s=0.20 for the transposed VPM LES formulation; callers may still
# override it for a case-specific validation.  Internally, the k-equilibrium
# model converts it to C_k via C_k = (C_s² √C_e)^(2/3).
SMAGORINSKY_CONSTANT = 0.20

# =========================================================
# FILE I/O CONSTANTS
# =========================================================

# Default logging frequency (every N time steps)
DEFAULT_LOGGING_TIME_INTERVAL = 1


# Maximum filename length
MAX_FILENAME_LENGTH = 255

# =========================================================
# CONVERGENCE AND TOLERANCE SETTINGS
# =========================================================

# Default convergence tolerance for iterative methods
DEFAULT_TOLERANCE = 1e-6

# Relative tolerance for floating point comparisons
RELATIVE_TOLERANCE = 1e-3

# Absolute tolerance for floating point comparisons
ABSOLUTE_TOLERANCE = 1e-4

# =========================================================
# VLM (VORTEX LATTICE METHOD) CONSTANTS
# =========================================================

# Biot-Savart regularization parameter (prevents singularity at vortex core)
VLM_EPSILON = 1e-8

# Cutoff threshold for zero-velocity detection
VLM_CUTOFF = 1e-14

# Panel-method regularization and cutoff constants
PANEL_EPSILON = 1e-14
PANEL_CUTOFF = 1e-8

# Tolerance for detecting coincident particles (for merging)
VLM_COINCIDENT_TOL = 1e-6

# Wake particle core overlap factor (sigma = factor * V * dt)
VLM_CORE_OVERLAP = 2.0

# Small velocity threshold for VLM computations
VLM_SMALL_VELOCITY = 1e-10

# =========================================================
# DOMAIN AND BOUNDARY CONDITIONS
# =========================================================

# Large domain size for unbounded problems
LARGE_DOMAIN_SIZE = 1e6

# =========================================================
# PERFORMANCE AND OPTIMIZATION
# =========================================================

# Number of threads for CPU parallelization
DEFAULT_NUM_THREADS = 8

# Memory allocation block size
MEMORY_BLOCK_SIZE = 1024

# Cache size for frequently accessed data
DEFAULT_CACHE_SIZE = 1000

THREADS_PER_BLOCK = 256

# =========================================================
# PERIODIC KERNEL PARAMETERS
# =========================================================
# Default box width for periodic boundary conditions
BOX_WIDTH = BOX_SIZE  # Must match DEFAULT_DOMAIN_BOUNDS

# Number of periodic images considered in calculations
MAX_IMAGES = 3

# =========================================================
# DEBUGGING AND DEVELOPMENT
# =========================================================

# Debug mode flag (can be overridden)
DEBUG_MODE = True

# Verbose output flag
VERBOSE_OUTPUT = False

# Profile performance flag
PROFILE_PERFORMANCE = False

# =========================================================
# VERSION INFORMATION
# =========================================================

# VPM module version
VPM_VERSION = "0.0.2"

# Minimum required Taichi version
MIN_TAICHI_VERSION = "1.6.0"

# =========================================================
# RUNTIME STATE (Set during initialization)
# =========================================================

# Taichi backend name (set by initialize_taichi_backend)
TAICHI_BACKEND = None

# Hardware device name (set by initialize_taichi_backend)
TAICHI_DEVICE_NAME = None

# Taichi device memory pool in bytes (set by initialize_taichi_backend).
# None when the backend manages its own pool (CPU, Metal).
TAICHI_POOL_BYTES = None

# =========================================================
# UTILITY FUNCTIONS FOR CONSTANTS
# =========================================================


def get_all_constants():
    """
    Return a dictionary of all constants defined in this module.

    Returns:
        dict: Dictionary mapping constant names to their values
    """
    import inspect

    constants = {}
    for name, value in globals().items():
        # Include only uppercase constants (following Python convention)
        if name.isupper() and not inspect.isfunction(value) and not inspect.ismodule(value):
            constants[name] = value

    return constants


def print_constants():
    """Print all constants with their values in a formatted way."""
    constants = get_all_constants()

    print("=" * 60)
    print("VPM Module Constants")
    print("=" * 60)

    for name, value in sorted(constants.items()):
        print(f"{name:<30} = {value}")

    print("=" * 60)


# =========================================================
# MODULE INITIALIZATION
# =========================================================

# Print constants if module is run directly
if __name__ == "__main__":
    print_constants()

# Precision constants for reference (optional, not required for config system)
F32 = "f32"
F64 = "f64"
