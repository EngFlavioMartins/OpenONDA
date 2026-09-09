"""Small bound/free increments must survive accumulation at tutorial precision."""

from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.boundary_elements.vlm.config import VLMSetup, VLMSurfaceSetup
from source.solvers.vpm.boundary_elements.vlm.solver.vlm_solver import VLMSolver
from source.solvers.vpm.numerics.runge_kutta import RungeKutta
from source.solvers.vpm.physics.stage_rhs import StageRHS, VLMStageContribution
from tutorials.vpm.flat_plate.assets.generate_surface import create_flat_plate


@pytest.mark.parametrize("precision,tolerance", [("f32", 5e-6), ("f64", 2e-13)])
def test_many_small_particle_increments_close_against_large_bound_strength(precision, tolerance):
    ti.reset()
    dtype = ti.f32 if precision == "f32" else ti.f64
    ti.init(arch=ti.cpu, default_fp=dtype, offline_cache=False, cpu_max_num_threads=2)
    try:
        plate = create_flat_plate(chord=1, span=4, n_chordwise_panels=2, n_spanwise_panels=2)
        vlm = VLMSolver(VLMSetup(surfaces=(VLMSurfaceSetup(plate),), dtype=precision))
        vlm.generate_mesh()
        lattice = vlm.lattice
        gamma = lattice.circulation.to_numpy()
        gamma[:] = np.where(lattice.is_mirrored.to_numpy() == 1, -1.0, 1.0)
        lattice.circulation.from_numpy(gamma)
        vlm._compute_cumulative_circulation_cpu()
        vlm._solved = vlm._coupled_mode = True
        old_bound = vlm.compute_total_bound_vortex_strength()
        count = 4096
        x = ti.Vector.field(3, dtype, shape=count)
        alpha = ti.Vector.field(3, dtype, shape=count)
        sigma = ti.field(dtype, shape=count)
        rng = np.random.default_rng(42)
        x.from_numpy(rng.uniform([0.5, 0.1, 0.1], [1.5, 1.5, 0.5], (count, 3)).astype(gamma.dtype))
        alpha.from_numpy(np.tile([1e-4, 2e-4, -1e-4], (count, 1)).astype(gamma.dtype))
        sigma.fill(0.3)
        before = alpha.to_numpy().astype(float)
        physics = SimpleNamespace(
            accumulator_dtype=dtype,
            max_n_particles=count,
            induction=SimpleNamespace(stretching_scheme="transposed"),
        )

        class BoundOnly:
            def evaluate_stage(self, **fields):
                fields["velocity_out"].fill(0.0)
                fields["vortex_strength_rate_out"].fill(0.0)

        rhs = StageRHS(BoundOnly(), (VLMStageContribution(vlm, physics),))
        rk = RungeKutta(max_n_particles=count, dtype=dtype)
        rk.advance(
            position=x,
            vortex_strength=alpha,
            core_radius=sigma,
            count=count,
            time=0.0,
            time_step_size=0.02,
            right_hand_side=rhs,
        )
        change = (alpha.to_numpy().astype(float) - before).sum(axis=0)
        assert np.linalg.norm(change) > 1e-4
        np.testing.assert_allclose(
            vlm._transported_bound.to_numpy().astype(float).sum(axis=0) - old_bound + change,
            0.0,
            atol=tolerance,
        )
    finally:
        ti.reset()
