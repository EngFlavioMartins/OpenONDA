"""Pressure-datum correction for closed velocity coupling boundaries."""

from __future__ import annotations

import logging

import numpy as np

from source.solvers.FVM.utils.cavity_utils import needs_pressure_reference

logger = logging.getLogger("coupler")


class PressureReference:
    """Set the mean upstream total pressure to its freestream value."""

    def __init__(
        self,
        fvm,
        *,
        fvm_box: np.ndarray,
        freestream_velocity: np.ndarray,
        particle_spacing: float,
        boundary_mode: str,
        enabled: bool,
        is_master: bool,
        comm=None,
    ) -> None:
        self.fvm_solver = fvm
        self.fvm_box = np.asarray(fvm_box, dtype=np.float64)
        self.freestream_velocity = np.asarray(freestream_velocity, dtype=np.float64)
        self.particle_spacing = float(particle_spacing)
        self.boundary_mode = boundary_mode
        self.enabled = bool(enabled)
        self.is_master = bool(is_master)
        self.comm = comm
        self.available = False
        self.cell_indices: np.ndarray | None = None
        self.last_shift = 0.0

    def prepare(self) -> None:
        """Determine whether the FVM pressure field has a free datum."""
        self.available = False
        if not self.enabled:
            return
        if self.boundary_mode == "directional_outflow":
            if self.is_master:
                logger.info("[Init] The downstream pressure boundary fixes the pressure datum.")
            return
        if not needs_pressure_reference(self.fvm_solver.boundaries):
            if self.is_master:
                logger.info("[Init] A Dirichlet pressure patch fixes the pressure datum.")
            return
        self.available = True
        if self.is_master:
            logger.info("[Init] Pressure datum anchored in the undisturbed upstream stream.")

    def _selection(self) -> np.ndarray | None:
        if self.cell_indices is not None:
            return self.cell_indices if self.cell_indices.size else None

        centres = np.asarray(
            self.fvm_solver.get_cell_centre_coordinates(), dtype=np.float64
        ).reshape(-1, 3)
        if centres.shape[0] == 0:
            self.cell_indices = np.empty(0, dtype=np.int64)
            return None

        axis = (
            int(np.argmax(np.abs(self.freestream_velocity)))
            if np.any(self.freestream_velocity != 0.0)
            else 0
        )
        sign = (
            float(np.sign(self.freestream_velocity[axis]))
            if self.freestream_velocity[axis] != 0.0
            else 1.0
        )
        plane = self.fvm_box[2 * axis] if sign >= 0 else self.fvm_box[2 * axis + 1]
        near = np.abs(centres[:, axis] - plane) <= 2.0 * self.particle_spacing
        for other in range(3):
            if other == axis:
                continue
            lo, hi = self.fvm_box[2 * other], self.fvm_box[2 * other + 1]
            margin = 0.25 * (hi - lo)
            near &= (centres[:, other] >= lo + margin) & (centres[:, other] <= hi - margin)
        self.cell_indices = np.flatnonzero(near)
        return self.cell_indices if self.cell_indices.size else None

    def correct(self, velocity: np.ndarray) -> None:
        """Shift kinematic pressure without changing its gradient."""
        if not self.available:
            return
        pressure = np.asarray(self.fvm_solver.get_pressure_field(), dtype=np.float64).ravel()
        velocity = np.asarray(velocity, dtype=np.float64).reshape(-1, 3)
        index = self._selection()

        delta = 0.0
        if self.is_master and index is not None and index.size and index.max() < len(pressure):
            head = pressure[index] + 0.5 * np.einsum("ij,ij->i", velocity[index], velocity[index])
            freestream_head = 0.5 * float(
                np.dot(self.freestream_velocity, self.freestream_velocity)
            )
            delta = float(freestream_head - np.mean(head))
        if self.comm is not None and self.comm.Get_size() > 1:
            delta = float(self.comm.bcast(delta if self.is_master else None, root=0))
        if not np.isfinite(delta):
            raise FloatingPointError("non-finite pressure datum shift")
        self.fvm_solver.shift_pressure_field(delta)
        self.last_shift = delta


__all__ = ["PressureReference"]
