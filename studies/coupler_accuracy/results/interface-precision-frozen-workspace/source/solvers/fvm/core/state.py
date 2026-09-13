"""Typed finite-volume field state shared by execution backends."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FieldState:
    """Synchronized primary FVM fields for one mesh state.

    Parameters
    ----------
    velocity : numpy.ndarray
        Cell-centred velocity in m/s with shape
        ``(n_cells_with_ghosts, 3)``. The first ``n_cells`` rows are fluid
        cells; remaining rows are solver-owned boundary ghosts.
    kinematic_pressure : numpy.ndarray
        Pressure divided by constant density, in m²/s², with one value per
        velocity row and shape ``(n_cells_with_ghosts,)``.
    volumetric_face_flux : numpy.ndarray
        Face flux ``phi = U_f · Sf`` in m³/s with shape ``(n_faces,)``. On an
        interior face it is positive from the owner cell to the neighbour.

    Notes
    -----
    Inputs are converted to contiguous ``float64`` arrays and validated for
    finite values. Already-compatible inputs may share storage; callers that
    require independent ownership should use :meth:`copy`. The solver may
    update the published arrays in place when exposing a newly solved state.
    """

    velocity: np.ndarray
    kinematic_pressure: np.ndarray
    volumetric_face_flux: np.ndarray

    def __post_init__(self) -> None:
        self.velocity = np.ascontiguousarray(
            self.velocity,
            dtype=np.float64,
        )
        self.kinematic_pressure = np.ascontiguousarray(
            self.kinematic_pressure,
            dtype=np.float64,
        )
        self.volumetric_face_flux = np.ascontiguousarray(
            self.volumetric_face_flux,
            dtype=np.float64,
        )

        if self.velocity.ndim != 2 or self.velocity.shape[1] != 3:
            raise ValueError("velocity must have shape (n_cells_with_ghosts, 3)")
        if self.kinematic_pressure.shape != (len(self.velocity),):
            raise ValueError("kinematic_pressure must contain one value per velocity row")
        if self.volumetric_face_flux.ndim != 1:
            raise ValueError("volumetric_face_flux must be one-dimensional")
        if not all(
            np.all(np.isfinite(values))
            for values in (
                self.velocity,
                self.kinematic_pressure,
                self.volumetric_face_flux,
            )
        ):
            raise ValueError("field state must contain only finite values")

    def copy(self) -> FieldState:
        """Return an independent deep copy of all three primary fields.

        Returns
        -------
        FieldState
            A new state whose arrays do not share writable storage with this
            object. Units and shapes are unchanged.
        """
        return FieldState(
            self.velocity.copy(),
            self.kinematic_pressure.copy(),
            self.volumetric_face_flux.copy(),
        )
