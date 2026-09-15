"""Active-cloud invariance and precision of native per-particle diagnostics."""

import sys

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.particles.container import Particles
from source.solvers.vpm.physics.evaluation import ParticleFieldEvaluation


@pytest.mark.parametrize(
    ("precision", "backend"),
    [
        ("f32", "cpu"),
        ("f64", "cpu"),
        pytest.param(
            "f32",
            "metal",
            marks=(pytest.mark.gpu, pytest.mark.skipif(sys.platform != "darwin", reason="Metal")),
        ),
    ],
)
def test_diagnostics_ignore_removed_sources_and_preserve_precision(precision, backend):
    arch = ti.metal if backend == "metal" else ti.cpu
    ti.init(
        arch=arch,
        default_fp=ti.f32 if precision == "f32" else ti.f64,
        offline_cache=False,
        cpu_max_num_threads=2,
    )
    try:
        assert ti.cfg.arch == arch
        dtype = np.float64 if precision == "f64" else np.float32
        accumulator = ti.f64 if precision == "f64" else ti.f32
        strength = 1.0 + 1.0e-8
        state = {
            "position": np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=dtype),
            "velocity": np.zeros((2, 3), dtype=dtype),
            "vortex_strength": np.array([[0.0, strength, 0.0], [0.0, 1.0, 1.0]], dtype=dtype),
            "core_radius": np.ones(2, dtype=dtype),
            "particle_volume": np.ones(2, dtype=dtype),
            "kinematic_viscosity": np.zeros(2, dtype=dtype),
        }
        particles = Particles(max_n_particles=4, float_dtype=precision)
        particles.add_vortex_particles(**state)
        particles.remove_vortex_particles([1])
        evaluator = ParticleFieldEvaluation(max_n_particles=4, accumulator_dtype=accumulator)
        names = ("kinetic_energy", "helicity", "enstrophy")
        expected = {
            "kinetic_energy": float(state["vortex_strength"][0, 1]) ** 2
            / (6.0 * np.pi**1.5 * np.sqrt(2.0)),
            "helicity": 0.0,
            "enstrophy": float(state["vortex_strength"][0, 1]) ** 2 / (np.pi**1.5 * 2.0**1.5),
        }
        tolerance = 2.0e-6 if precision == "f32" else 2.0e-13
        for name in names:
            result = getattr(evaluator, f"compute_particles_{name}")(particles)
            assert result.shape == (1,)
            assert result.dtype == dtype
            np.testing.assert_allclose(result, [expected[name]], rtol=tolerance, atol=0.0)

        particles.replace_from_numpy(**{key: value[:1] for key, value in state.items()})
        for name in names:
            result = getattr(evaluator, f"compute_particles_{name}")(particles)
            np.testing.assert_allclose(result, [expected[name]], rtol=tolerance, atol=0.0)

        particles.remove_vortex_particles(None, remove_all=True)
        for name in names:
            result = getattr(evaluator, f"compute_particles_{name}")(particles)
            assert result.shape == (0,)
            assert result.dtype == dtype
    finally:
        ti.reset()
