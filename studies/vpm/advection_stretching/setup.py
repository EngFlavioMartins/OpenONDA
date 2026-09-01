"""Hard-coded physics-facing configuration for this offline qualification."""
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
SEED = 20260901
MODES = ("DIRECT", "TRANSPOSED", "MIXED")
METHODS = ("fractional_x_gamma", "fractional_gamma_x", "parallel_lagged",
           "strang_x_gamma_x", "strang_gamma_x_gamma", "coupled_rk2",
           "coupled_rk3", "coupled_rk4_reference", "reuse_stage_gradients",
           "averaged_gradient_exponential")
MSTEPS = (4, 8, 16, 32, 64)
DSTEPS = (5, 10, 20, 40)
HORIZON = 1.0
DISCRETE_HORIZON = 0.08
SIGMA = 0.28

CHECKPOINTS = (
    ("leapfrog_healthy_050", ROOT / "tutorials/vpm/vortex_interactions/solution/leapfrog_les/vpm_000050.h5", 20 * .035**2 / np.pi),
    ("leapfrog_late_150", ROOT / "tutorials/vpm/vortex_interactions/solution/leapfrog_les/vpm_000150.h5", 20 * .035**2 / np.pi),
    ("rotor_healthy_515", ROOT / "tutorials/vpm/rotor_flow/solution/vpm_rotor_000515.h5", .006),
    ("rotor_prefailure_520", ROOT / "tutorials/vpm/rotor_flow/solution/vpm_rotor_000520.h5", .006),
    ("rotor_rejected_520", ROOT / "tutorials/vpm/rotor_flow/solution/rejected_state.h5", .006),
)

def cloud():
    axis = np.linspace(-.45, .45, 3)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    position = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    base = np.array(((1.,0.,0.), (0.,1.,0.), (0.,0.,1.), (1.,-.5,.25)))
    strength = np.vstack([base[i % 4] for i in range(len(position))])
    return position, strength / np.linalg.norm(strength, axis=1)[:, None]

def mkdirs():
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
