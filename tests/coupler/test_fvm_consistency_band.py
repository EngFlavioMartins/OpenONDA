"""Contracts for resolved-scale VPM-to-FVM consistency."""

from types import SimpleNamespace

import numpy as np
import pytest

from source.coupler.config.types import CouplerSetup
from source.coupler.consistency import (
    RATE_FIELD,
    TARGET_FIELD,
    FVMConsistencyBand,
    build_consistency_rate,
    maximum_consistency_rate,
)
from source.solvers.fvm.core.solver import FVMSolver


class _FVM:
    def __init__(self, centres: np.ndarray):
        self.centres = np.asarray(centres, dtype=np.float64)
        self.parallel = SimpleNamespace(is_root=True)
        self.fields: dict[str, np.ndarray] = {}

    def get_cell_centre_coordinates(self) -> np.ndarray:
        return self.centres.copy()

    def set_cell_scalar_field(self, name: str, values: np.ndarray) -> None:
        self.fields[name] = np.asarray(values).copy()

    def set_cell_vector_field(
        self,
        name: str,
        component_x: np.ndarray,
        component_y: np.ndarray,
        component_z: np.ndarray,
    ) -> None:
        self.fields[name] = np.column_stack((component_x, component_y, component_z))


def test_consistency_rate_is_c1_and_confined_to_outer_buffer():
    points = np.array(
        [
            [-1.5, 0.0, 0.0],
            [-1.375, 0.0, 0.0],
            [-1.25, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    rate = build_consistency_rate(
        points,
        np.array([-1.5, 1.5, -1.5, 1.5, -1.5, 1.5]),
        width=0.25,
        maximum_rate=16.0,
    )

    np.testing.assert_allclose(rate, [16.0, 8.0, 0.0, 0.0], atol=2e-15)
    epsilon = 1.0e-7
    near_outer = build_consistency_rate(
        np.array([[-1.5 + epsilon, 0.0, 0.0]]),
        np.array([-1.5, 1.5, -1.5, 1.5, -1.5, 1.5]),
        width=0.25,
        maximum_rate=16.0,
    )[0]
    near_inner = build_consistency_rate(
        np.array([[-1.25 - epsilon, 0.0, 0.0]]),
        np.array([-1.5, 1.5, -1.5, 1.5, -1.5, 1.5]),
        width=0.25,
        maximum_rate=16.0,
    )[0]
    assert 16.0 - near_outer < 1.0e-10
    assert near_inner < 1.0e-10


def test_consistency_rate_is_transit_scaled_and_time_step_capped():
    assert maximum_consistency_rate(1.0, 0.25, 0.05) == pytest.approx(16.0)
    assert maximum_consistency_rate(10.0, 0.25, 0.05) == pytest.approx(20.0)


def test_consistency_band_interpolates_time_levels_and_resynchronizes_endpoint():
    setup = CouplerSetup(
        transfer_region_bounds=(-1.25, 1.25, -1.25, 1.25, -1.25, 1.25),
        fvm_consistency_width=0.25,
    )
    fvm = _FVM(np.array([[-1.5, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    band = FVMConsistencyBand(
        setup,
        fvm,
        coupling_time_step_size=0.05,
        fvm_box=np.array([-1.5, 1.5, -1.5, 1.5, -1.5, 1.5]),
    )

    assert fvm.fields[RATE_FIELD][0] > 0.0
    assert fvm.fields[RATE_FIELD][1] == 0.0
    band.update_target(np.array([[1.0, 0.0, 0.0]]))
    band.update_target(np.array([[3.0, 2.0, 0.0]]))
    band.push_target(0.25)
    np.testing.assert_allclose(fvm.fields[TARGET_FIELD][0], [1.5, 0.5, 0.0])
    np.testing.assert_allclose(fvm.fields[TARGET_FIELD][1], setup.freestream_velocity)

    band.update_endpoint(np.array([[2.0, 4.0, 0.0]]))
    band.push_target(1.0)
    np.testing.assert_allclose(fvm.fields[TARGET_FIELD][0], [2.0, 4.0, 0.0])


def test_consistency_width_cannot_enter_fvm_authority():
    setup = CouplerSetup(
        transfer_region_bounds=(-1.25, 1.25, -1.25, 1.25, -1.25, 1.25),
        fvm_consistency_width=0.26,
    )
    with pytest.raises(ValueError, match="must fit"):
        setup.validate_transfer_region_box(np.array([-1.5, 1.5, -1.5, 1.5, -1.5, 1.5]))


def test_fvm_consistency_source_preserves_high_pass_velocity():
    solver = FVMSolver.__new__(FVMSolver)
    solver.mesh_data = {"n_cells": 2}
    solver.geo_data = {}
    solver.velocity = np.array([[3.0, 1.0, 0.0], [2.0, -1.0, 0.0]])
    solver.registered_fields = {
        RATE_FIELD: np.array([4.0, 0.0]),
        TARGET_FIELD: np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    }
    low_pass = np.array([[2.5, 0.25, 0.0], [2.5, -0.25, 0.0]])
    solver._coupling_consistency_filter = lambda _velocity: low_pass

    explicit, implicit = solver._coupling_consistency_source()

    np.testing.assert_allclose(explicit[0], 4.0 * np.array([1.5, 0.75, 0.0]))
    np.testing.assert_allclose(explicit[1], 0.0)
    np.testing.assert_allclose(implicit, [4.0, 0.0])
