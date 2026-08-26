"""Focused contracts for the atomic coupled checkpoint."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pytest

from source.coupler import boundary as boundary_module
from source.coupler.checkpoint import (
    config_difference_paths,
    load_coupled_state,
    save_coupled_state,
)


class _MappingSetup:
    def __init__(self, mapping: dict):
        self.mapping = deepcopy(mapping)

    def to_dict(self) -> dict:
        return deepcopy(self.mapping)


class _CouplerSetup(_MappingSetup):
    def __init__(
        self,
        *,
        transfer_method: str = "common_lattice",
        vpm_checkpoint_retention: int = 1,
    ):
        super().__init__(
            {
                "coupler": {
                    "boundary_condition_mode": "pressure_gradient",
                    "transfer_method": transfer_method,
                }
            }
        )
        self.boundary_condition_mode = "pressure_gradient"
        self.coupling_patch = "numericalBoundary"
        self.freestream_velocity = [1.0, 0.0, 0.0]
        self.vpm_checkpoint_retention = vpm_checkpoint_retention


class _Panel:
    def __init__(self, coupling_scope: str):
        self.max_n_panels = 128
        self.float_dtype = "f32"
        self.linear_solver_name = "SCIPY"
        self.force_config = SimpleNamespace(method="BERNOULLI")
        self.boundary_condition_type = "NEUMANN"
        self.density = 1.0
        self.freestream_velocity = np.array([1.0, 0.0, 0.0])
        self.coupling_scope = coupling_scope
        self.raise_on_non_convergence = True
        self.residual_tolerance = None
        self.far_field_acceptance = 5.0
        self.far_field_min_panels = 256
        self.reuse_constrained_factorization = True

    def induced_velocity_diagnostic_is_due(self) -> bool:
        return False


class _FVM:
    def __init__(self):
        self.parallel = SimpleNamespace(is_partitioned=False)
        self.step = 2
        self.time = 0.1

    def save_state(self, path: Path) -> None:
        with path.open("wb") as stream:
            np.savez(stream, step=np.asarray(self.step), time=np.asarray(self.time))

    def load_state(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as state:
            self.step = int(state["step"])
            self.time = float(state["time"])


class _VPM:
    def __init__(
        self,
        *,
        gbd_threshold: float,
        panel_scope: str,
        checkpoint_directory: str,
        velocity: np.ndarray,
        pressure_gradient: np.ndarray,
    ):
        self.setup = _MappingSetup(
            {
                "time_step_size": 0.1,
                "time": 0.0,
                "step": 0,
                "viscous": {
                    "scheme": "GBD",
                    "gbd_threshold": gbd_threshold,
                },
                "precision": "f32",
                "checkpoint_directory": checkpoint_directory,
            }
        )
        self.panel_solver = _Panel(panel_scope)
        self.particles = SimpleNamespace(n_particles_total=0)
        self.step = 1
        self.time = 0.1
        self.current_velocity = np.asarray(velocity, dtype=np.float64).copy()
        self.base_pressure_gradient = np.asarray(pressure_gradient, dtype=np.float64).copy()
        self.last_include_temporal: bool | None = None
        self.last_velocity_previous: np.ndarray | None = None

    def save_numerical_state(self, filename: str) -> None:
        Path(f"{filename}.h5").write_bytes(b"fake-vpm-state")
        Path(f"{filename}.xdmf").write_text("<Xdmf/>", encoding="utf-8")

    def load_numerical_state(self, filename: str) -> None:
        assert Path(filename).is_file()

    def refresh_boundary_element_solution(self) -> None:
        return None

    def compute_velocity_at_points(self, points: np.ndarray, **_kwargs) -> np.ndarray:
        assert len(points) == len(self.current_velocity)
        return self.current_velocity.copy()

    def compute_pressure_gradient_at_points(
        self,
        points: np.ndarray,
        *,
        density: float,
        include_temporal: bool,
        velocity_previous: np.ndarray | None,
        time_step_size: float,
        **_kwargs,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        assert len(points) == len(self.current_velocity)
        self.last_include_temporal = include_temporal
        self.last_velocity_previous = (
            None if velocity_previous is None else np.asarray(velocity_previous).copy()
        )
        pressure_gradient = self.base_pressure_gradient.copy()
        if include_temporal:
            assert velocity_previous is not None
            pressure_gradient += (self.current_velocity - velocity_previous) / time_step_size
        return {"pressure_gradient": density * pressure_gradient}, self.current_velocity.copy()


def _make_coupler(
    *,
    gbd_threshold: float = 1.0e-5,
    panel_scope: str = "vpm_boundary_condition",
    transfer_method: str = "common_lattice",
    checkpoint_directory: str = "original-output",
    vpm_checkpoint_retention: int = 1,
):
    previous_velocity = np.array([[1.0, 0.1, 0.0], [1.0, 0.1, 0.0]])
    previous_pressure_gradient = np.array([[0.2, -0.1, 0.0], [0.3, -0.2, 0.0]])
    vpm = _VPM(
        gbd_threshold=gbd_threshold,
        panel_scope=panel_scope,
        checkpoint_directory=checkpoint_directory,
        velocity=previous_velocity,
        pressure_gradient=np.array([[0.4, 0.2, 0.0], [0.5, 0.1, 0.0]]),
    )
    coupler = SimpleNamespace(
        setup=_CouplerSetup(
            transfer_method=transfer_method,
            vpm_checkpoint_retention=vpm_checkpoint_retention,
        ),
        fvm_solver=_FVM(),
        vpm_solver=vpm,
        vorticity_transfer=SimpleNamespace(step=0),
        n_fvm_substeps=2,
        _is_master=True,
        _velocity_boundary_condition_old=previous_velocity.copy(),
        _normal_velocity_boundary_condition_old=None,
        _normal_velocity_boundary_condition=np.ones(2),
        _tangential_gradient_boundary_condition_old=None,
        _tangential_gradient_boundary_condition=np.ones((2, 3)),
        _kinematic_pressure_gradient_boundary_condition_old=(previous_pressure_gradient.copy()),
        _kinematic_pressure_gradient_boundary_condition=np.full((2, 3), 99.0),
        _pressure_velocity_snapshot=previous_velocity.copy(),
        density=1.0,
        kinematic_viscosity=0.01,
        fvm_time_step_size=0.05,
        vpm_time_step_size=0.1,
        vpm_particle_spacing=0.1,
        freestream_velocity=np.array([1.0, 0.0, 0.0]),
        fvm_box=np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]),
        _last_vpm_boundary_condition_flux_diagnostics={},
    )
    return coupler


def test_checkpoint_retains_a_bounded_post_renewal_vpm_history(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    coupler = _make_coupler(vpm_checkpoint_retention=2)
    for step in (1, 2, 3):
        coupler.fvm_solver.step = 2 * step
        coupler.vpm_solver.step = step
        coupler.vpm_solver.time = 0.1 * step
        save_coupled_state(coupler, checkpoint, coupling_step=step)

    assert sorted(path.name for path in checkpoint.glob("vpm_*.h5")) == [
        "vpm_000002.h5",
        "vpm_000003.h5",
    ]
    assert sorted(path.name for path in checkpoint.glob("vpm_*.xdmf")) == [
        "vpm_000002.xdmf",
        "vpm_000003.xdmf",
    ]
    assert sorted(path.name for path in checkpoint.glob("fvm_*")) == ["fvm_000003.npz"]
    assert sorted(path.name for path in checkpoint.glob("vpm_boundary_condition_*")) == [
        "vpm_boundary_condition_000003.npz"
    ]


def test_config_difference_paths_are_recursive_and_distinguish_missing_from_none():
    stored = {
        "vpm": {
            "viscous": {"scheme": "GBD", "gbd_threshold": 1.0e-5},
            "optional": None,
        }
    }
    current = {"vpm": {"viscous": {"scheme": "GBD", "gbd_threshold": 2.0e-5}}}

    assert config_difference_paths(stored, current) == {
        "vpm.viscous.gbd_threshold",
        "vpm.optional",
    }


@pytest.mark.parametrize(
    ("path", "changed"),
    [
        ("vpm.viscous.gbd_threshold", {"gbd_threshold": 2.0e-5}),
        ("panel.coupling_scope", {"panel_scope": "full"}),
        ("coupler.transfer_method", {"transfer_method": "buffered_m4_renewal"}),
    ],
)
def test_restart_config_changes_require_the_exact_allowlist_path(tmp_path, path, changed):
    checkpoint = tmp_path / "checkpoint"
    save_coupled_state(_make_coupler(), checkpoint, coupling_step=1)

    strict = _make_coupler(**changed)
    with pytest.raises(ValueError, match=re.escape(path)):
        load_coupled_state(strict, checkpoint)

    allowed = _make_coupler(**changed)
    with pytest.warns(RuntimeWarning, match=re.escape(path)):
        restored_step = load_coupled_state(
            allowed,
            checkpoint,
            allowed_config_differences={path},
        )
    assert restored_step == 1
    assert allowed.vorticity_transfer.step == 2


def test_pressure_gradient_restart_matches_uninterrupted_continuation(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    uninterrupted = _make_coupler()
    expected_previous_velocity = uninterrupted._pressure_velocity_snapshot.copy()
    expected_previous_gradient = (
        uninterrupted._kinematic_pressure_gradient_boundary_condition_old.copy()
    )
    save_coupled_state(uninterrupted, checkpoint, coupling_step=1)

    restarted = _make_coupler(checkpoint_directory="relocated-output")
    restarted._pressure_velocity_snapshot = None
    restarted._kinematic_pressure_gradient_boundary_condition_old = None
    load_coupled_state(restarted, checkpoint)

    np.testing.assert_array_equal(restarted._pressure_velocity_snapshot, expected_previous_velocity)
    np.testing.assert_array_equal(
        restarted._kinematic_pressure_gradient_boundary_condition_old,
        expected_previous_gradient,
    )
    assert restarted._kinematic_pressure_gradient_boundary_condition is None
    assert restarted._normal_velocity_boundary_condition is None
    assert restarted._tangential_gradient_boundary_condition is None

    next_velocity = np.array([[1.2, 0.15, 0.0], [1.2, 0.15, 0.0]])
    uninterrupted.vpm_solver.current_velocity = next_velocity.copy()
    restarted.vpm_solver.current_velocity = next_velocity.copy()
    face_centre = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    face_normal = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    face_area = np.ones(2)

    uninterrupted_trace = boundary_module.evaluate_vpm_boundary(
        uninterrupted, face_centre, face_normal, face_area
    )
    restarted_trace = boundary_module.evaluate_vpm_boundary(
        restarted, face_centre, face_normal, face_area
    )

    assert uninterrupted.vpm_solver.last_include_temporal is True
    assert restarted.vpm_solver.last_include_temporal is True
    np.testing.assert_array_equal(
        restarted.vpm_solver.last_velocity_previous,
        uninterrupted.vpm_solver.last_velocity_previous,
    )
    np.testing.assert_allclose(
        restarted._kinematic_pressure_gradient_boundary_condition,
        uninterrupted._kinematic_pressure_gradient_boundary_condition,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(restarted_trace[0], uninterrupted_trace[0], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(restarted_trace[1], uninterrupted_trace[1], rtol=0.0, atol=0.0)

    applied_gradients: list[np.ndarray] = []

    def capture_pressure_gradient(
        _coupler,
        _patch,
        _velocity,
        pressure_gradient,
        **_kwargs,
    ) -> None:
        applied_gradients.append(np.asarray(pressure_gradient).copy())

    monkeypatch.setattr(boundary_module, "apply_fvm_boundary", capture_pressure_gradient)
    boundary_module.advance_fvm_substeps(
        uninterrupted,
        "numericalBoundary",
        face_centre,
        face_normal,
        face_area,
        uninterrupted_trace[0],
        uninterrupted_trace[1],
        uninterrupted._kinematic_pressure_gradient_boundary_condition_old,
        uninterrupted._kinematic_pressure_gradient_boundary_condition,
    )
    expected_applied_gradients = [value.copy() for value in applied_gradients]
    applied_gradients.clear()
    boundary_module.advance_fvm_substeps(
        restarted,
        "numericalBoundary",
        face_centre,
        face_normal,
        face_area,
        restarted_trace[0],
        restarted_trace[1],
        restarted._kinematic_pressure_gradient_boundary_condition_old,
        restarted._kinematic_pressure_gradient_boundary_condition,
    )
    assert len(applied_gradients) == len(expected_applied_gradients) == 2
    for restarted_gradient, uninterrupted_gradient in zip(
        applied_gradients, expected_applied_gradients, strict=True
    ):
        np.testing.assert_allclose(restarted_gradient, uninterrupted_gradient, rtol=0.0, atol=0.0)
