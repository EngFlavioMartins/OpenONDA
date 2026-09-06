"""Pressure diagnostics must use one consistent accepted particle state."""

from __future__ import annotations

import numpy as np

from openonda import vpm


def test_pressure_is_invariant_to_redundant_refresh_after_deferred_advance(tmp_path):
    solver = vpm.VPMSolver(
        vpm.VPMCase(
            directory=tmp_path,
            backup=vpm.Backup(interval_steps=0),
            numerics=vpm.Numerics(
                time_step_size=0.01,
                compute_device="CPU",
                precision="f32",
                max_n_particles=16,
                max_evaluation_points=16,
                induction=vpm.DirectInduction(),
                viscous=vpm.ViscousConfig.inviscid(),
                verbose=False,
            ),
        )
    )
    solver.add_vortex_particles(
        position=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        velocity=np.zeros((2, 3)),
        vortex_strength=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        core_radius=np.full(2, 0.2),
        particle_volume=np.full(2, 1.0e-3),
        kinematic_viscosity=np.zeros(2),
    )

    body_gradient = np.diag([1.0, -1.0, 0.0])

    def body_velocity(points, _time):
        return np.asarray(points) * np.array([1.0, -1.0, 0.0])

    def body_velocity_gradient(points, _time):
        return np.broadcast_to(body_gradient, (len(points), 3, 3)).copy()

    solver.set_body_induced_velocity(body_velocity, body_velocity_gradient)
    solver.advance(defer_output=True)
    target = np.array([[0.5, 0.3, 0.1]])

    before = solver.compute_pressure_gradient_at_points(
        target,
        include_viscous=False,
        particle_spacing=0.05,
    )
    solver.stepper._update_velocity_and_gradients()
    after = solver.compute_pressure_gradient_at_points(
        target,
        include_viscous=False,
        particle_spacing=0.05,
    )

    for name in before:
        np.testing.assert_allclose(before[name], after[name], rtol=0.0, atol=1.0e-12, err_msg=name)
    solver.close()
