"""
Physical and numerical constants shared by the VPM solver.
"""

# Default hard particle capacity
MAX_N_PARTICLES = 500000

# DVH truncation-error parameter β ≈ 0.077 (Durante et al. 2024, Eq. 14-15).
# Controls the Gaussian width of the heat-kernel scatter: 4nu·Δt_d = β·R_d².
DVH_BETA = 0.077

# Default cutoff radius factor for particle interactions
DEFAULT_CUTOFF_RADIUS_FACTOR = 100

# Default Smagorinsky constant C_s, shared with the FVM. Internally, the
# k-equilibrium model converts it to C_k via C_k = (C_s² √C_e)^(2/3).
SMAGORINSKY_CONSTANT = 0.17
