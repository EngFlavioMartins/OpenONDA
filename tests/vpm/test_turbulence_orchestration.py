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


def test_fixed_les_filter_preserves_closure_under_particle_volume_changes():
    import numpy as np
    import taichi as ti

    from source.solvers.vpm.config.turbulence import TurbulenceConfig

    ti.init(arch=ti.cpu, offline_cache=False)
    try:

        class Cloud:
            particle_volume = ti.field(ti.f32, shape=3)
            strain_rate = ti.Matrix.field(3, 3, ti.f32, shape=3)
            kinematic_viscosity = ti.field(ti.f32, shape=3)
            eddy_viscosity = ti.field(ti.f32, shape=3)
            effective_viscosity = ti.field(ti.f32, shape=3)

            def __len__(self):
                return 3

        cloud = Cloud()
        strain = np.repeat(np.diag([1.0, -1.0, 0.0])[None], 3, axis=0).astype(np.float32)
        cloud.strain_rate.from_numpy(strain)
        cloud.kinematic_viscosity.fill(0.001)
        volume = np.array([0.03, 0.05, 0.08], dtype=np.float32) ** 3
        config = TurbulenceConfig.les_smagorinsky(filter_width=0.07)
        fixed = ParticlesLES.rebuild(config, max_n_particles=3)
        local = ParticlesLES.rebuild(TurbulenceConfig.les_smagorinsky(), max_n_particles=3)
        for scale in (1.0, 8.0):
            cloud.particle_volume.from_numpy(volume * scale)
            fixed.compute(cloud)
            # Uniform incompressible linear strain has |S|=2/s.
            np.testing.assert_allclose(
                cloud.eddy_viscosity.to_numpy(), 0.2**2 * 0.07**2 * 2, rtol=2e-6
            )
            local.compute(cloud)
            np.testing.assert_allclose(
                cloud.eddy_viscosity.to_numpy(), 0.2**2 * (volume * scale) ** (2 / 3) * 2, rtol=2e-6
            )
    finally:
        ti.reset()


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


def test_viscosity_update_does_not_run_optional_statistics():
    calls = []
    les = object.__new__(ParticlesLES)
    les.model = SimpleNamespace(
        smagorinsky_coefficient=0.2, compute=lambda *args: calls.append("viscosity")
    )
    les.update_turbulence_statistics = lambda *args: (_ for _ in ()).throw(
        AssertionError("unexpected synchronization")
    )
    les.compute(object(), 0.01)
    assert calls == ["viscosity"]
