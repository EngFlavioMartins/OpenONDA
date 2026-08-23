"""Regression tests for VLM precision, coupling checks, and packaging."""

from pathlib import Path
import tomllib
from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.boundary_elements.vlm.config import VLMSetup, VLMSurfaceSetup
from source.solvers.vpm.boundary_elements.vlm.geometry.aircraft import Aircraft, Wing, WingSegment
from source.solvers.vpm.boundary_elements.vlm.solver.lattice import VLMLattice
from source.solvers.vpm.boundary_elements.vlm.solver.mesh import generate_vlm_mesh
from source.solvers.vpm.boundary_elements.vlm.solver.vlm_solver import VLMSolver
from source.solvers.vpm.core.validation import _validate_time_step_sizing
from source.solvers.vpm.runtime.backend import reset_taichi_backend


def _plate(*, chord: float = 2.0, n_chordwise_panels: int = 4) -> Aircraft:
    segment = WingSegment(
        uid="segment",
        vertex_position={
            "a": np.array([0.0, 0.0, 0.0]),
            "b": np.array([0.0, 1.0, 0.0]),
            "c": np.array([chord, 1.0, 0.0]),
            "d": np.array([chord, 0.0, 0.0]),
        },
        n_chordwise_panels=n_chordwise_panels,
        n_spanwise_panels=1,
    )
    wing = Wing(uid="wing")
    wing.add_segment(segment)
    aircraft = Aircraft(uid="plate")
    aircraft.add_wing(wing)
    return aircraft


@pytest.fixture
def taichi_f64():
    """Provide a clean CPU runtime whose default floating-point type is f64."""
    reset_taichi_backend()
    ti.init(arch=ti.cpu, default_fp=ti.f64)
    yield
    reset_taichi_backend()


def test_coupling_stability_reports_resolved_and_under_resolved_steps():
    vlm = VLMSolver(VLMSetup(surfaces=(VLMSurfaceSetup(_plate()),)))

    resolved = vlm.check_coupling_stability(0.25, [1.0, 0.0, 0.0])
    assert resolved == {
        "stable": True,
        "courant": pytest.approx(0.5),
        "max_dt": pytest.approx(0.5),
        "characteristic_speed": pytest.approx(1.0),
        "min_panel_chord": pytest.approx(0.5),
    }

    with pytest.warns(RuntimeWarning, match="under-resolved"):
        under_resolved = vlm.check_coupling_stability(1.0, [1.0, 0.0, 0.0])
    assert under_resolved["stable"] is False
    assert under_resolved["courant"] == pytest.approx(2.0)


def test_coupling_stability_uses_shortest_edge_of_tapered_panels():
    segment = WingSegment(
        uid="tapered",
        vertex_position={
            "a": np.array([0.0, 0.0, 0.0]),
            "b": np.array([0.0, 1.0, 0.0]),
            "c": np.array([1.0, 1.0, 0.0]),
            "d": np.array([2.0, 0.0, 0.0]),
        },
        n_chordwise_panels=4,
        n_spanwise_panels=1,
    )
    wing = Wing(uid="wing")
    wing.add_segment(segment)
    aircraft = Aircraft(uid="tapered_plate")
    aircraft.add_wing(wing)
    vlm = VLMSolver(VLMSetup(surfaces=(VLMSurfaceSetup(aircraft),)))

    result = vlm.check_coupling_stability(0.25, (1.0, 0.0, 0.0))

    assert result["min_panel_chord"] == pytest.approx(0.25)
    assert result["courant"] == pytest.approx(1.0)


@pytest.mark.parametrize("time_step_size", [0.0, -0.1, np.inf, np.nan])
def test_coupling_stability_rejects_invalid_time_steps(time_step_size):
    vlm = VLMSolver(VLMSetup(surfaces=(VLMSurfaceSetup(_plate()),)))
    with pytest.raises(ValueError, match="finite and positive"):
        vlm.check_coupling_stability(time_step_size, [1.0, 0.0, 0.0])


def test_f64_mesh_generation_preserves_input_precision(taichi_f64):
    x_offset = 1.00000002
    segment = WingSegment(
        uid="segment",
        vertex_position={
            "a": np.array([x_offset, 0.0, 0.0]),
            "b": np.array([x_offset, 1.0, 0.0]),
            "c": np.array([x_offset + 1.0, 1.0, 0.0]),
            "d": np.array([x_offset + 1.0, 0.0, 0.0]),
        },
        n_chordwise_panels=1,
        n_spanwise_panels=1,
    )
    wing = Wing(uid="wing")
    wing.add_segment(segment)
    aircraft = Aircraft(uid="offset_plate")
    aircraft.add_wing(wing)

    lattice = VLMLattice(max_n_panels=1, dtype=ti.f64)
    generate_vlm_mesh(aircraft, lattice)

    stored = lattice.panel_corner_position.to_numpy()[0, 0, 0]
    assert lattice.panel_corner_position.dtype == ti.f64
    assert stored == pytest.approx(x_offset, rel=0.0, abs=1e-14)


def test_set_circulation_uploads_one_contiguous_array(taichi_f64):
    lattice = VLMLattice(max_n_panels=3, dtype=ti.f64)
    lattice.n_panels = 2

    lattice.set_circulation(np.array([[1.25], [-0.5]], dtype=np.float64))

    np.testing.assert_array_equal(lattice.circulation.to_numpy(), [1.25, -0.5, 0.0])


def test_spacing_variation_warning_uses_min_over_max_ratio():
    class Particles:
        n_particles_total = 2

        @staticmethod
        def position_cpu():
            return np.zeros((2, 3))

        @staticmethod
        def velocity_cpu():
            return np.array([[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]])

        @staticmethod
        def core_radius_cpu():
            return np.array([0.05, 0.20])

        @staticmethod
        def kinematic_viscosity_cpu():
            return np.full(2, 1.0e-5)

        @staticmethod
        def velocity_gradient_cpu():
            return np.zeros((2, 3, 3))

    system = SimpleNamespace(
        particles=Particles(),
        LES=None,
        time_step_size=0.01,
        viscous_scheme="NONE",
    )

    result = _validate_time_step_sizing(system, verbose=False)

    assert result["h_ratio"] == pytest.approx(0.25)
    assert any("Large variation in particle spacing" in issue for issue in result["issues"])


def test_core_dependencies_declare_taichi():
    """VPM/VLM ship in the default install, so Taichi is required, not optional."""
    project = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())
    taichi_requirements = [
        requirement
        for requirement in project["project"]["dependencies"]
        if requirement.startswith("taichi")
    ]

    assert taichi_requirements, "taichi must be a core dependency"
    # Intel macOS is pinned to the last compatible wheel; every other platform
    # takes the current release. Both markers must stay present.
    assert any("platform_machine=='x86_64'" in req for req in taichi_requirements)
    assert any("platform_machine!='x86_64'" in req for req in taichi_requirements)
