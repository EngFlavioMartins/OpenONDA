"""
Physical and numerical constants shared by the VPM solver.
"""

import numpy as np
import taichi as ti

# Numerical epsilon for stability and convergence checks
EPSILON = 1e-12

# Biot-Savart constant 1/(4 pi)
ONE_OVER_FOUR_PI = 1.0 / (4.0 * np.pi)

# Precision of the VLM linear-system workspaces
NP_FLOAT = np.float32
TI_FLOAT = ti.f32

# Default hard particle capacity
MAX_N_PARTICLES = 500000

# DVH truncation-error parameter β ≈ 0.077 (Durante et al. 2024, Eq. 14-15).
# Controls the Gaussian width of the heat-kernel scatter: 4nu·Δt_d = β·R_d².
_DVH_BETA = 0.077

# Default time step size [s]
DEFAULT_TIME_STEP = 0.01

# Default cutoff radius factor for particle interactions
DEFAULT_CUTOFF_RADIUS_FACTOR = 100

# Gaussian q uses its integrated series below this radius to avoid
# small-radius cancellation. Host, direct/FMM and LBVH share the coefficients
# in kernels/gaussian.py; the number of terms depends on precision.
GAUSSIAN_Q_SERIES_CROSSOVER = 1.0

# Default Smagorinsky constant C_s, shared with the FVM. Internally, the
# k-equilibrium model converts it to C_k via C_k = (C_s² √C_e)^(2/3).
SMAGORINSKY_CONSTANT = 0.17

# Regularization of the singular VLM Biot-Savart segments [m]
VLM_EPSILON = 1e-8

# Small velocity or length below which a VLM direction is undefined
VLM_SMALL_VELOCITY = 1e-10

# Taichi backend name, set by the runtime backend selection
TAICHI_BACKEND = None
