"""Pressure-datum invariance for the closed coupled FVM boundary."""

from __future__ import annotations

import numpy as np

from source.coupler.pressure_reference import PressureReference


class _ClosedFVM:
    boundaries = [{"name": "cut", "pressure_type": "zeroGradient"}]

    def __init__(self, pressure: np.ndarray, centres: np.ndarray):
        self.pressure = pressure.copy()
        self.centres = centres

    def get_cell_centre_coordinates(self):
        return self.centres

    def get_pressure_field(self):
        return self.pressure

    def shift_pressure_field(self, delta):
        self.pressure += delta


def _corrected_pressure(initial_shift: float) -> tuple[np.ndarray, float]:
    centres = np.array(
        [
            [-0.95, -0.2, -0.2],
            [-0.95, -0.2, 0.2],
            [-0.95, 0.2, -0.2],
            [-0.95, 0.2, 0.2],
        ]
    )
    fvm = _ClosedFVM(np.full(4, 3.0 + initial_shift), centres)
    reference = PressureReference(
        fvm,
        fvm_box=np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]),
        freestream_velocity=np.array([1.0, 0.0, 0.0]),
        particle_spacing=0.1,
        boundary_mode="vorticity_mixed",
        enabled=True,
        is_master=True,
    )
    reference.prepare()
    reference.correct(np.tile([1.0, 0.0, 0.0], (4, 1)))
    return fvm.pressure, reference.last_shift


def test_pressure_anchor_is_invariant_to_the_incoming_datum():
    base, base_delta = _corrected_pressure(0.0)
    shifted, shifted_delta = _corrected_pressure(17.0)

    np.testing.assert_allclose(base, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(shifted, base, atol=1.0e-14)
    assert shifted_delta == base_delta - 17.0
