"""Typed finite-volume field state shared by execution backends."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FieldState:
    """Cell-centred velocity, pressure, and face flux for one time step.

    Enforces a contiguous ``float64`` memory layout and finite-value
    validation in ``__post_init__``. Velocity and pressure contain one row per
    cell including boundary ghost cells; ``face_flux`` instead contains one
    value per mesh face.

    Call :meth:`copy` to produce an independent checkpoint that is not
    affected by subsequent solver updates.

    Attributes
    ----------
    velocity : np.ndarray
        Cell and boundary-ghost velocity [m/s], shape
        ``(n_cells_with_ghosts, 3)``.
    pressure : np.ndarray
        Cell and boundary-ghost kinematic pressure ``p/ρ`` [m²/s²], shape
        ``(n_cells_with_ghosts,)``.
    face_flux : np.ndarray
        Volumetric face flux ``U·Sf`` [m³/s], shape ``(n_faces,)``.

    Examples
    --------
    >>> state = FieldState(
    ...     velocity=np.zeros((n_cells, 3)),
    ...     pressure=np.zeros(n_cells),
    ...     face_flux=np.zeros(n_faces),
    ... )
    """

    velocity: np.ndarray
    pressure: np.ndarray
    face_flux: np.ndarray

    def __post_init__(self) -> None:
        self.velocity = np.ascontiguousarray(self.velocity, dtype=np.float64)
        self.pressure = np.ascontiguousarray(self.pressure, dtype=np.float64)
        self.face_flux = np.ascontiguousarray(self.face_flux, dtype=np.float64)
        if self.velocity.ndim != 2 or self.velocity.shape[1] != 3:
            raise ValueError("velocity must have shape (n_cells_with_ghosts, 3)")
        if self.pressure.shape != (len(self.velocity),):
            raise ValueError("pressure must contain one value per velocity row")
        if self.face_flux.ndim != 1:
            raise ValueError("face_flux must be one-dimensional")
        if not all(
            np.all(np.isfinite(values)) for values in (self.velocity, self.pressure, self.face_flux)
        ):
            raise ValueError("field state must contain only finite values")

    def copy(self) -> FieldState:
        """Return an independent checkpoint-ready field state."""
        return FieldState(self.velocity.copy(), self.pressure.copy(), self.face_flux.copy())
