import numpy as np
import pytest

taichi = pytest.importorskip("taichi", reason="VPM requires taichi")

from source.solvers.vpm.particles.container import Particles
from source.solvers.vpm.stabilization.operators import StabilizationOperators


def _ensure_taichi_cpu() -> None:
    if taichi.lang.impl.get_runtime().prog is None:
        taichi.init(arch=taichi.cpu)


def _particles() -> Particles:
    particles = Particles(max_n_particles=2)
    particles.add_vortex_particles(
        position=np.zeros((2, 3), dtype=np.float32),
        velocity=np.array([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        vortex_strength=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        core_radius=np.array([0.5, 1.0], dtype=np.float32),
        particle_volume=np.ones(2, dtype=np.float32),
        kinematic_viscosity=np.zeros(2, dtype=np.float32),
        velocity_gradient=np.array(
            [
                [[4.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=np.float32,
        ),
    )
    return particles


def test_device_stability_reduction_matches_lagrangian_cfl_number():
    _ensure_taichi_cpu()
    particles = _particles()
    operators = StabilizationOperators(taichi.f32, 2)

    result = operators.inspect_solution(
        particles,
        time_step_size=0.25,
        check_stability=True,
    )

    assert result["valid"]
    assert result["lagrangian_cfl"] == pytest.approx(1.0)


def test_device_stability_scan_rejects_non_finite_particle_state():
    _ensure_taichi_cpu()
    particles = _particles()
    particles.vortex_strength[0] = [np.nan, 0.0, 0.0]
    operators = StabilizationOperators(taichi.f32, 2)

    assert not operators.inspect_solution(particles)["valid"]
