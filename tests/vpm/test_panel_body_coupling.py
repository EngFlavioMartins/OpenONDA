from types import SimpleNamespace

import numpy as np

from source.solvers.VPM.core.solver import VPMSolver
from source.solvers.VPM.physics.engine import _AdvectionHandler


def test_panel_body_field_is_added_to_target_velocity():
    class Panel:
        lattice = None

        def add_surface(self, uid, path):
            assert uid == "body"
            assert path == "cube.stl"
            self.lattice = SimpleNamespace(num_panels=12)

        def initialize(self, force=False):
            assert force

        @staticmethod
        def compute_induced_velocity(points):
            return np.zeros_like(points)

    solver = VPMSolver.__new__(VPMSolver)
    solver.physics = SimpleNamespace(body_velocity=None)
    panel = Panel()
    solver._init_optional_solvers(
        SimpleNamespace(panel_solver=panel, body_stl="cube.stl", vlm=None)
    )

    assert solver.panel_solver is panel
    assert solver._body_induced_fn == panel.compute_induced_velocity
    assert solver.physics.body_velocity == panel.compute_induced_velocity


def test_body_field_is_added_at_each_advection_velocity_evaluation():
    class Field:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float32)

        def to_numpy(self):
            return self.values.copy()

        def from_numpy(self, values):
            self.values = np.asarray(values, dtype=np.float32)

    class Physics:
        velocity_override = None
        body_velocity = staticmethod(lambda points: np.full_like(points, [0.2, -0.1, 0.0]))

        @staticmethod
        def velocity_self(_pos, _gamma, _radius, output, _background, _count, reuse_tree=False):
            output.values[:] = [1.0, 0.0, 0.0]

    particles = SimpleNamespace(
        circulation=Field(np.zeros((2, 3))),
        radius=Field(np.ones(2)),
        velocity_background=Field(np.zeros((2, 3))),
    )
    positions = Field([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    velocities = Field(np.zeros((2, 3)))

    _AdvectionHandler(Physics())._vel(particles, positions, velocities, 2)

    np.testing.assert_allclose(velocities.values, [[1.2, -0.1, 0.0]] * 2)


def test_target_only_panel_field_is_not_added_to_particle_advection():
    class Panel:
        lattice = SimpleNamespace(num_panels=12)
        coupling_scope = "vpm_bc"

        @staticmethod
        def initialize(force=False):
            assert force

        @staticmethod
        def compute_induced_velocity(points):
            return np.full_like(points, [0.2, 0.0, 0.0])

    solver = VPMSolver.__new__(VPMSolver)
    solver.physics = SimpleNamespace(body_velocity=None)
    panel = Panel()
    solver._init_optional_solvers(SimpleNamespace(panel_solver=panel, body_stl=None, vlm=None))

    assert solver._body_induced_fn == panel.compute_induced_velocity
    assert solver.physics.body_velocity is None


def test_pressure_only_panel_field_is_not_added_to_velocity_targets():
    class Panel:
        lattice = SimpleNamespace(num_panels=12)
        coupling_scope = "pressure"

        @staticmethod
        def initialize(force=False):
            assert force

        @staticmethod
        def compute_induced_velocity(points):
            return np.full_like(points, [0.2, 0.0, 0.0])

    solver = VPMSolver.__new__(VPMSolver)
    solver.physics = SimpleNamespace(body_velocity=None)
    panel = Panel()
    solver._init_optional_solvers(SimpleNamespace(panel_solver=panel, body_stl=None, vlm=None))

    assert solver._body_induced_fn is None
    assert solver._pressure_body_induced_fn == panel.compute_induced_velocity


def test_panel_field_contributes_to_pressure_gradient():
    matrix = np.array([[0.2, 0.1, 0.0], [-0.1, 0.05, 0.0], [0.0, 0.0, -0.25]])

    class PressurePhysics:
        @staticmethod
        def compute_target_pressure_gradients_hierarchical(_particles, points, **kwargs):
            body_velocity = kwargs["body_fn"](points)
            velocity = kwargs["freestream_velocity"] + body_velocity
            return {"grad_p": -np.einsum("mb,ab->ma", velocity, matrix)}

    solver = VPMSolver.__new__(VPMSolver)
    freestream = np.array([1.0, 0.0, 0.0])
    solver.particles = SimpleNamespace(
        number_of_particles=0,
        velocity_background_cpu=lambda: freestream,
    )
    solver._pressure_physics = PressurePhysics()
    solver._body_induced_fn = lambda points: np.asarray(points) @ matrix.T

    points = np.array([[0.4, -0.2, 0.1]])
    result = solver.compute_target_pressure_gradients(
        points,
        include_viscous=False,
        include_temporal=False,
        temporal_method="eulerian",
        treecode_theta=0.3,
    )

    velocity = freestream + points @ matrix.T
    expected = -np.einsum("mb,ab->ma", velocity, matrix)
    np.testing.assert_allclose(result["grad_p"], expected, rtol=1e-8, atol=1e-8)
