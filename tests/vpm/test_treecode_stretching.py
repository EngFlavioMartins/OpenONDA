"""
Treecode-accelerated vortex stretching tests.

The stretching rate dΓ/dt is an exact local contraction of the velocity
gradient ∇u:  J·Γ (DIRECT), Jᵀ·Γ (TRANSPOSED), S·Γ (MIXED).  With
StretchingConfig(use_treecode=True) the gradient is evaluated by the O(N log N)
treecode instead of the O(N²) pairwise kernel.  These tests assert that the
treecode rate matches the direct pairwise rate within the Barnes–Hut tolerance
for every mode, and that the per-mode contraction convention is not swapped.
"""

import numpy as np
import pytest

from source.solvers.VPM import Solver, SolverConfig
from source.solvers.VPM.config.types import (
    AdvectionConfig,
    StretchingConfig,
    TurbulenceConfig,
    VelocityConfig,
    ViscousConfig,
)

_MODES = [("DIRECT", 0), ("TRANSPOSED", 1), ("MIXED", 2)]


@pytest.fixture(scope="module")
def solver_and_rates(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("tc_stretch"))
    rng = np.random.default_rng(0)
    N = 2000
    pos = rng.uniform(-1, 1, (N, 3)).astype(np.float32)
    circ = rng.normal(0, 0.1, (N, 3)).astype(np.float32)
    rad = np.full(N, 0.15, np.float32)
    cfg = SolverConfig(
        time_step_size=0.01, processing_unit="CPU",
        advection=AdvectionConfig(scheme="RK3"),
        stretching=StretchingConfig.transposed(scheme="RK3"),
        viscous=ViscousConfig(scheme="NONE"),
        velocity=VelocityConfig.treecode(theta=0.2),
        turbulence=TurbulenceConfig.inviscid(),
        backup_frequency=0, logging_frequency=0,
        backup_directory=out, solution_name=out, clean=True, max_particles=N + 16,
    )
    s = Solver(config=cfg)
    s.add_vortex_particles(
        pos, np.zeros((N, 3), np.float32), circ, rad,
        np.full(N, 1e-3, np.float32), viscosity=np.full(N, 1e-3, np.float32),
        group_id=np.zeros(N, np.int32),
    )
    phys, pc, h = s.physics, s.particles, s.physics._stretching
    h._treecode_theta = 0.2

    direct, treecode = {}, {}
    for name, m in _MODES:
        phys._resize_temp_fields(len(pc)); phys._zero_temp_fields()
        phys.compute_stretching_rate_kernel(
            pc.position, pc.circulation, pc.radius, phys.dstr_dt_temp, m, len(pc)
        )
        direct[name] = phys.dstr_dt_temp.to_numpy()[:len(pc)].copy()
        h._use_treecode = True
        h._rate(pc.position, pc.circulation, pc.radius, phys.dstr_dt_temp2, m, len(pc))
        treecode[name] = phys.dstr_dt_temp2.to_numpy()[:len(pc)].copy()
    yield direct, treecode
    s.reset_gpu()


@pytest.mark.parametrize("name,_m", _MODES)
def test_treecode_rate_matches_direct(solver_and_rates, name, _m):
    """Treecode stretching rate ≈ direct pairwise rate within BH tolerance."""
    direct, treecode = solver_and_rates
    rel = np.linalg.norm(treecode[name] - direct[name]) / (np.linalg.norm(direct[name]) + 1e-30)
    assert rel < 0.10, f"mode {name}: treecode vs direct relL2 ={rel:.3e} exceeds BH tolerance"


def test_mode_conventions_not_swapped(solver_and_rates):
    """Each treecode mode must match its OWN direct mode, not another.

    If the J vs Jᵀ contraction were swapped, treecode-DIRECT would match
    direct-TRANSPOSED instead — this guards that convention.
    """
    direct, treecode = solver_and_rates

    def rel(a, b):
        return np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30)

    # DIRECT and TRANSPOSED are genuinely different fields here, so a swap is
    # detectable: same-mode match must be far tighter than cross-mode.
    same = rel(treecode["DIRECT"], direct["DIRECT"])
    cross = rel(treecode["DIRECT"], direct["TRANSPOSED"])
    assert same < 0.10 < cross, (
        f"convention check failed: DIRECT-vs-DIRECT={same:.2e}, "
        f"DIRECT-vs-TRANSPOSED={cross:.2e}"
    )


def test_config_flag_plumbs_through():
    stab = StretchingConfig.transposed(use_treecode=True, treecode_theta=0.25)
    assert stab.use_treecode is True
    assert stab.treecode_theta == 0.25


def test_velocity_treecode_tuning_flags_plumb_to_physics(tmp_path):
    cfg = SolverConfig(
        time_step_size=0.01,
        processing_unit="CPU",
        advection=AdvectionConfig(scheme="NONE"),
        stretching=StretchingConfig.disabled(),
        viscous=ViscousConfig(scheme="NONE"),
        velocity=VelocityConfig.treecode(
            theta=0.4,
            multipole_order=2,
            sort_particle_targets=True,
            traversal_block_dim=64,
        ),
        turbulence=TurbulenceConfig.inviscid(),
        backup_frequency=0,
        logging_frequency=0,
        backup_directory=str(tmp_path),
        solution_name=str(tmp_path),
        clean=True,
        max_particles=32,
    )
    solver = Solver(config=cfg)
    try:
        assert solver.physics.velocity_theta == 0.4
        assert solver.physics.treecode_multipole_order == 2
        assert solver.physics.treecode_sort_particle_targets is True
        assert solver.physics.treecode_traversal_block_dim == 64
    finally:
        solver.reset_gpu()
