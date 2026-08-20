"""Typed finite-volume field state shared by execution backends."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FieldState:
    """Cell velocity, kinematic pressure, and face flux for one time level."""

    velocity: np.ndarray
    kinematic_pressure: np.ndarray
    face_flux: np.ndarray

    def __post_init__(self) -> None:
        self.velocity = np.ascontiguousarray(
            self.velocity,
            dtype=np.float64,
        )
        self.kinematic_pressure = np.ascontiguousarray(
            self.kinematic_pressure,
            dtype=np.float64,
        )
        self.face_flux = np.ascontiguousarray(
            self.face_flux,
            dtype=np.float64,
        )

        if self.velocity.ndim != 2 or self.velocity.shape[1] != 3:
            raise ValueError("velocity must have shape (n_cells_with_ghosts, 3)")
        if self.kinematic_pressure.shape != (len(self.velocity),):
            raise ValueError("kinematic_pressure must contain one value per velocity row")
        if self.face_flux.ndim != 1:
            raise ValueError("face_flux must be one-dimensional")
        if not all(
            np.all(np.isfinite(values))
            for values in (
                self.velocity,
                self.kinematic_pressure,
                self.face_flux,
            )
        ):
            raise ValueError("field state must contain only finite values")

    def copy(self) -> FieldState:
        """Return an independent field-state copy."""
        return FieldState(
            self.velocity.copy(),
            self.kinematic_pressure.copy(),
            self.face_flux.copy(),
        )
