"""Shared geometry and physics for the Re = 300 sphere benchmark pair.

Both cases import this, so the setups can only differ where intended.
"""

from __future__ import annotations

import os

import numpy as np

from openonda.fvm import stretched

# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
DIAMETER = 1.0
U_INF = (1.0, 0.0, 0.0)
RHO = 1.0
REYNOLDS = 300.0
NU = float(np.linalg.norm(U_INF)) * DIAMETER / REYNOLDS

# Re = 300 sheds hairpin vortices: unsteady, periodic, vortex lines closing.
# Johnson & Patel, JFM 378 (1999) 19-70; a sanity anchor only.
LITERATURE = {"Cd": 0.656, "St": 0.137, "Cl_mean": 0.069}

SMOKE = os.environ.get("OPENONDA_SMOKE", "0") == "1"

# ---------------------------------------------------------------------------
# Discretisation
# ---------------------------------------------------------------------------
# h/D = 1/16. The transfer needs ~4 lattice cells per vortex-core radius for
# 1%, and the hairpin cores are about 0.25 D.
SPACING = float(os.environ.get("OPENONDA_SPACING", "0.125" if SMOKE else "0.0625"))
DT_FVM = float(os.environ.get("OPENONDA_FVM_DT", "0.02"))
DT_VPM = float(os.environ.get("OPENONDA_VPM_DT", "0.10"))

# Shedding starts near tU/D = 30, period ~7.3: nine periods of statistics.
T_END = float(os.environ.get("OPENONDA_T_END", "0.4" if SMOKE else "100.0"))
SHEDDING_START = float(os.environ.get("OPENONDA_SHEDDING_START", "40.0"))

FORCE_INTERVAL = float(os.environ.get("OPENONDA_FORCE_INTERVAL", "0.1"))
DIAGNOSTIC_INTERVAL = float(os.environ.get("OPENONDA_DIAGNOSTIC_INTERVAL", "1.0"))
CHECKPOINT_INTERVAL = float(os.environ.get("OPENONDA_CHECKPOINT_INTERVAL", "10.0"))
VOLUME_INTERVAL = float(os.environ.get("OPENONDA_VOLUME_INTERVAL", "25.0"))
SAMPLE_SPACING = float(os.environ.get("OPENONDA_SAMPLE_SPACING", str(SPACING)))

# ---------------------------------------------------------------------------
# Domains
# ---------------------------------------------------------------------------
# The bubble closes by x/D ~ 1.6, so the interface sits downstream: vorticity
# must convect out through the outflow face.
HANDOFF_BOX = (-2.0, 4.0, -2.0, 2.0, -2.0, 2.0)
DOWNSTREAM_BUFFER = float(os.environ.get("OPENONDA_FVM_DOWNSTREAM_BUFFER", "0.5"))
FVM_BOX = (
    HANDOFF_BOX[0],
    HANDOFF_BOX[1] + DOWNSTREAM_BUFFER,
    HANDOFF_BOX[2],
    HANDOFF_BOX[3],
    HANDOFF_BOX[4],
    HANDOFF_BOX[5],
)
VPM_DOMAIN = (-6.0, 24.0, -6.0, 6.0, -6.0, 6.0)

# Uniform near field, then stretched far enough to approximate the unbounded
# problem the hybrid solves.
REFERENCE_CORE = (-2.0, 6.0, -2.0, 2.0, -2.0, 2.0)
REFERENCE_DOMAIN = (-8.0, 20.0, -8.0, 8.0, -8.0, 8.0)
# 1.20 with a 12h cap: the smallest mesh whose largest size jump is 1.20.
STRETCH_RATIO = float(os.environ.get("OPENONDA_STRETCH_RATIO", "1.20"))


def graded_axis(lo: float, core_lo: float, core_hi: float, hi: float, h: float) -> np.ndarray:
    """Uniform ``h`` across the core, geometrically stretched to the far field."""
    n_core = int(round((core_hi - core_lo) / h))
    if abs((core_hi - core_lo) / h - n_core) > 1e-9:
        raise ValueError(f"core span {core_hi - core_lo:g} is not a multiple of h={h:g}")
    core = core_lo + h * np.arange(n_core + 1)
    left = stretched(core_lo, lo, h, STRETCH_RATIO, 12.0 * h)[::-1]
    right = stretched(core_hi, hi, h, STRETCH_RATIO, 12.0 * h)
    return np.concatenate([left, core, right])


def reference_nodes(h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        graded_axis(REFERENCE_DOMAIN[0], REFERENCE_CORE[0], REFERENCE_CORE[1],
                    REFERENCE_DOMAIN[1], h),
        graded_axis(REFERENCE_DOMAIN[2], REFERENCE_CORE[2], REFERENCE_CORE[3],
                    REFERENCE_DOMAIN[3], h),
        graded_axis(REFERENCE_DOMAIN[4], REFERENCE_CORE[4], REFERENCE_CORE[5],
                    REFERENCE_DOMAIN[5], h),
    )


def step_period(name: str, interval: float, dt: float) -> int:
    ratio = interval / dt
    period = int(round(ratio))
    if interval <= 0.0 or dt <= 0.0 or period < 1 or not np.isclose(ratio, period, atol=1e-10):
        raise ValueError(f"{name}={interval:g} must be a positive integer multiple of dt={dt:g}")
    return period
