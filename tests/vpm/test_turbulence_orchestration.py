from types import SimpleNamespace

from source.solvers.vpm.turbulence.turbulence import ParticlesLES


def test_zero_smagorinsky_skips_viscosity_kernels():
    model = SimpleNamespace(
        smagorinsky_coefficient=0.0,
        compute=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()),
    )
    les = object.__new__(ParticlesLES)
    les.model = model

    les.compute(object())

    assert les.min_eddy_viscosity == 0.0
    assert les.max_eddy_viscosity == 0.0
    assert les.min_eddy_viscosity_ratio == 0.0
    assert les.max_eddy_viscosity_ratio == 0.0


def test_parallel_les_statistics_use_active_particles_and_reset_each_call():
    import numpy as np
    import pytest
    import taichi as ti

    ti.init(arch=ti.cpu, offline_cache=False)
    try:
        capacity, count = 8192, 4097
        les = ParticlesLES("LES_SMAGORINSKY", max_n_particles=capacity)

        class Cloud:
            eddy_viscosity = ti.field(ti.f32, shape=capacity)
            kinematic_viscosity = ti.field(ti.f32, shape=capacity)

            def __len__(self):
                return count

        cloud = Cloud()
        cloud.kinematic_viscosity.fill(0.001)
        rng = np.random.default_rng(7)
        for active in (
            rng.uniform(0.0002, 0.004, count).astype(np.float32),
            np.full(count, 0.002, dtype=np.float32),
        ):
            values = np.full(capacity, 100, dtype=np.float32)
            values[:count] = active
            cloud.eddy_viscosity.from_numpy(values)
            les.update_turbulence_statistics(cloud)
            assert les.min_eddy_viscosity == float(active.min())
            assert les.max_eddy_viscosity == float(active.max())
            assert les.max_eddy_viscosity_ratio == pytest.approx(float(active.max()) / 0.001)
            np.testing.assert_array_equal(cloud.eddy_viscosity.to_numpy(), values)
    finally:
        ti.reset()
