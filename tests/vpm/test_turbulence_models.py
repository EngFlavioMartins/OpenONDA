from types import SimpleNamespace

from openonda.vpm.turbulence.turbulence import ParticlesLES


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

    from openonda.vpm.config.turbulence import TurbulenceConfig

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


def _gradient_cloud(gradients, strengths, viscosity=1.0e-4, volume=0.05**3):
    import numpy as np
    import taichi as ti

    count = len(gradients)

    class Cloud:
        velocity_gradient = ti.Matrix.field(3, 3, ti.f64, shape=count)
        vortex_strength = ti.Vector.field(3, ti.f64, shape=count)
        particle_volume = ti.field(ti.f64, shape=count)
        kinematic_viscosity = ti.field(ti.f64, shape=count)
        eddy_viscosity = ti.field(ti.f64, shape=count)
        effective_viscosity = ti.field(ti.f64, shape=count)

        def __len__(self):
            return count

    cloud = Cloud()
    cloud.velocity_gradient.from_numpy(np.asarray(gradients, dtype=np.float64))
    cloud.vortex_strength.from_numpy(np.asarray(strengths, dtype=np.float64))
    cloud.particle_volume.fill(volume)
    cloud.kinematic_viscosity.fill(viscosity)
    return cloud


def test_sigma_model_matches_singular_values_and_vanishes_for_two_dimensional_flows():
    import numpy as np
    import taichi as ti

    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False)
    try:
        rng = np.random.default_rng(3)
        general = rng.normal(size=(6, 3, 3))
        vanishing = [
            [[0, 1, 0], [0, 0, 0], [0, 0, 0]],  # pure shear
            [[0, -1, 0], [1, 0, 0], [0, 0, 0]],  # solid rotation
            [[1, 0, 0], [0, -1, 0], [0, 0, 0]],  # plane strain
            [[-2, 0, 0], [0, 1, 0], [0, 0, 1]],  # axisymmetric strain
        ]
        gradients = np.concatenate((general, np.asarray(vanishing, dtype=float)))
        cloud = _gradient_cloud(gradients, np.zeros((len(gradients), 3)))
        les = ParticlesLES("LES_SIGMA", max_n_particles=len(gradients))
        les.compute(cloud)
        s = np.linalg.svd(gradients, compute_uv=False)
        expected = (1.35 * 0.05) ** 2 * s[:, 2] * (s[:, 0] - s[:, 1]) * (s[:, 1] - s[:, 2]) / s[:, 0] ** 2
        nu_t = cloud.eddy_viscosity.to_numpy()
        np.testing.assert_allclose(nu_t[:6], expected[:6], rtol=1e-6)
        np.testing.assert_allclose(nu_t[6:], 0.0, atol=1e-9)
        np.testing.assert_allclose(cloud.effective_viscosity.to_numpy(), 1.0e-4 + nu_t)
    finally:
        ti.reset()


def test_consistency_switch_spares_aligned_strengths_and_exchange_keeps_core_viscosity():
    import numpy as np
    import taichi as ti

    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False)
    try:
        gradient = np.random.default_rng(5).normal(size=(3, 3))
        vorticity = np.array(
            [gradient[2, 1] - gradient[1, 2], gradient[0, 2] - gradient[2, 0], gradient[1, 0] - gradient[0, 1]]
        )
        normal = np.cross(vorticity, [1.0, 0.0, 0.0])
        normal /= np.linalg.norm(normal)
        angle = np.radians(3.0)
        tilted = np.cos(angle) * vorticity / np.linalg.norm(vorticity) + np.sin(angle) * normal
        cloud = _gradient_cloud(np.repeat(gradient[None], 2, axis=0), [vorticity, tilted])
        free = ParticlesLES("LES_SIGMA", max_n_particles=2)
        free.compute(cloud)
        unswitched = cloud.eddy_viscosity.to_numpy()[0]
        les = ParticlesLES(
            "LES_SIGMA",
            max_n_particles=2,
            consistency_angle=3.0,
            subgrid_transport="STRENGTH_EXCHANGE",
            particle_dtype=ti.f64,
        )
        les.compute(cloud)
        # Aligned: no eddy viscosity; misaligned by theta_0: half of the model value.
        np.testing.assert_allclose(cloud.eddy_viscosity.to_numpy(), [0.0, 0.5 * unswitched], atol=1e-14)
        np.testing.assert_allclose(cloud.effective_viscosity.to_numpy(), 1.0e-4)
    finally:
        ti.reset()


def test_strength_exchange_conserves_strength_and_matches_pairwise_sum():
    import numpy as np
    import taichi as ti

    from openonda.vpm.turbulence.strength_exchange import StrengthExchange

    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False)
    try:
        rng = np.random.default_rng(1)
        n = 1500
        position = rng.uniform(-0.5, 0.5, (n, 3)) * [3.0, 1.0, 1.0]
        strength = rng.normal(size=(n, 3)) * 1e-3
        core = rng.uniform(0.07, 0.09, n)
        volume = rng.uniform(0.8, 1.2, n) * 0.05**3
        eddy = np.where(rng.random(n) < 0.3, 0.0, rng.uniform(0.0, 2e-3, n))

        class Cloud:
            def __len__(self):
                return n

            def touch_state(self):
                pass

        cloud = Cloud()
        for name, values in (
            ("position", position),
            ("vortex_strength", strength),
            ("core_radius", core),
            ("particle_volume", volume),
            ("eddy_viscosity", eddy),
        ):
            field = ti.Vector.field(3, ti.f64, n) if values.ndim == 2 else ti.field(ti.f64, n)
            field.from_numpy(values)
            setattr(cloud, name, field)
        dt = 3.75e-3
        StrengthExchange(n, ti.f64).apply(cloud, dt)
        updated = cloud.vortex_strength.to_numpy()

        d = position[:, None] - position[None]
        r2 = (d**2).sum(-1)
        eps2 = 0.5 * (core[:, None] ** 2 + core[None] ** 2)
        eta = 4 / np.pi**1.5 * np.exp(-r2 / eps2) / eps2**1.5 * (r2 < 16 * eps2)
        np.fill_diagonal(eta, 0.0)
        pair = 0.5 * (eddy[:, None] + eddy[None]) * eta / eps2
        rate = np.einsum(
            "pq,pqk->pk", pair, volume[:, None, None] * strength[None] - volume[None, :, None] * strength[:, None]
        )
        np.testing.assert_allclose(updated, strength + dt * rate, rtol=0, atol=1e-7 * np.abs(dt * rate).max())
        np.testing.assert_allclose(updated.sum(axis=0), strength.sum(axis=0), rtol=0, atol=1e-15)
    finally:
        ti.reset()
