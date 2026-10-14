"""DVH heat-kernel diffusion between free-slip slab planes."""

import numpy as np

from openonda import vpm

H = 0.08
NU = 1.0 / 150.0


def test_one_layer_dvh_step_is_the_two_dimensional_heat_kernel(tmp_path):
    planes = (-0.5 * H, 0.5 * H)
    viscous = vpm.ViscousConfig.dvh(particle_spacing=H, padding=5.0, kinematic_viscosity=NU)
    solver = vpm.VPMSolver(
        vpm.VPMCase(
            directory=tmp_path,
            name="test",
            numerics=vpm.Numerics(
                time_step_size=viscous.dvh_required_time_step_size(),
                compute_device="cpu",
                precision="f32",
                max_n_particles=1024,
                domain_bounds=(-0.8, 0.8, -0.8, 0.8, *planes),
                induction=vpm.SlipSlabInduction(
                    vpm.DirectInduction(), z_min=planes[0], z_max=planes[1]
                ),
                viscous=viscous,
            ),
        )
    )
    try:
        # One axial particle on the single layer
        circulation = 0.7 * H**3
        solver.add_vortex_particles(
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            np.array([[0.0, 0.0, circulation]]),
            np.full(1, H),
            np.full(1, H**3),
        )
        dt = viscous.dvh_required_time_step_size()
        result = solver.physics.grid_based_diffusion(
            solver.particles,
            time_step_size=dt,
            particle_spacing=H,
            kinematic_viscosity=NU,
            domain_padding=viscous.dvh_domain_padding,
            regen_threshold=1.0e-30,
            regen_threshold_mode="absolute",
            rd_ratio=viscous.dvh_support_radius_ratio,
        )

        # Planar heat kernel sampled on the nodes of the layer
        width = 4.0 * NU * dt
        radius_sq = np.sum(result["position"][:, :2].astype(np.float64) ** 2, axis=1)
        expected = circulation * H**2 / (np.pi * width) * np.exp(-radius_sq / width)
        np.testing.assert_allclose(result["position"][:, 2], 0.0, atol=1e-7)
        np.testing.assert_allclose(result["particle_volume"], H**3, rtol=1e-6)
        np.testing.assert_allclose(result["vortex_strength"][:, 2].sum(), circulation, rtol=1e-5)
        np.testing.assert_allclose(
            result["vortex_strength"][:, 2], expected, rtol=0.0, atol=1e-4 * expected.max()
        )
    finally:
        solver.close()
