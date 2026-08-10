"""FVM fringe relaxation toward the VPM velocity."""

from __future__ import annotations

import logging

import numpy as np

_logger = logging.getLogger("coupler")
FRINGE_STRENGTH = 4.0


def build_lambda(
    cell_centres: np.ndarray,
    fvm_box: tuple[float, float, float, float, float, float],
    buffer_thickness: float,
    lambda_max: float,
    dead_zone: float = 0.0,
) -> np.ndarray:
    """Return the C1 relaxation profile complementary to VPM authority."""
    points = np.atleast_2d(cell_centres)
    lo = np.asarray(fvm_box)[[0, 2, 4]]
    hi = np.asarray(fvm_box)[[1, 3, 5]]
    distance = np.minimum(points - lo, hi - points).min(axis=1)
    dead_zone = max(dead_zone, 0.0)
    width = max(buffer_thickness - dead_zone, 1.0e-30)
    relaxation = np.zeros(len(points))
    relaxation[distance <= dead_zone] = lambda_max
    active = (distance > dead_zone) & (distance < buffer_thickness)
    phase = (distance[active] - dead_zone) / width
    relaxation[active] = 0.5 * lambda_max * (1.0 + np.cos(np.pi * phase))
    return relaxation


def lambda_max_from_scales(
    u_char: float,
    buffer_thickness: float,
    dt: float,
) -> float:
    return float(
        min(
            FRINGE_STRENGTH * u_char / max(buffer_thickness, 1.0e-12),
            1.0 / max(dt, 1.0e-12),
        )
    )


class FringeFields:
    def __init__(self, cfg, vpm, fvm, *, coupling_dt: float):
        self.cfg = cfg
        self.vpm = vpm
        self.fvm = fvm
        self.cell_centres = np.asarray(fvm.get_cell_center_coordinates()).reshape(-1, 3)
        u_char = float(np.linalg.norm(cfg.U_inf))
        lambda_max = lambda_max_from_scales(u_char, cfg.buffer_thickness, coupling_dt)
        dead_zone = float(cfg.dead_zone_h) * float(cfg.h)
        self.relaxation = build_lambda(
            self.cell_centres,
            cfg.fvm_box,
            cfg.buffer_thickness,
            lambda_max,
            dead_zone,
        )
        self.fvm.set_cell_scalar_field("lambdaRelax", np.ascontiguousarray(self.relaxation))
        self._previous: np.ndarray | None = None
        self._next: np.ndarray | None = None
        _logger.info(
            "[Fringe] strength=%.1f lambda_max=%.3e 1/s cells=%d",
            FRINGE_STRENGTH,
            lambda_max,
            int(np.count_nonzero(self.relaxation)),
        )

    @property
    def active_cell_centres(self) -> np.ndarray:
        """Cell centres that require a VPM target evaluation."""
        return self.cell_centres[self.relaxation > 0.0]

    def update_target(self, active_velocity: np.ndarray | None = None) -> None:
        """Refresh the fringe target, optionally from a shared target solve.

        The coupler normally evaluates active fringe cells and donor faces in
        one treecode call, then passes the fringe slice here.  Keeping the
        fallback preserves the standalone helper contract.
        """
        active = self.relaxation > 0.0
        target = np.tile(self.cfg.U_inf, (len(self.cell_centres), 1)).astype(float)
        if active.any():
            if active_velocity is None:
                active_velocity = self.vpm.compute_target_velocities(
                    self.cell_centres[active],
                    include_freestream=True,
                    zone_mask=None,
                    include_body=True,
                )
            values = np.asarray(active_velocity, dtype=float).reshape(-1, 3)
            if len(values) != int(np.count_nonzero(active)):
                raise ValueError("active fringe velocity count does not match active cells")
            target[active] = values
        self._previous = target if self._next is None else self._next
        self._next = target
        self._push(target)

    def push_target(self, alpha: float) -> None:
        if self._next is None:
            return
        assert self._previous is not None
        fraction = float(np.clip(alpha, 0.0, 1.0))
        self._push(self._previous + fraction * (self._next - self._previous))

    def _push(self, target: np.ndarray) -> None:
        self.fvm.set_cell_vector_field(
            "Utarget",
            np.ascontiguousarray(target[:, 0]),
            np.ascontiguousarray(target[:, 1]),
            np.ascontiguousarray(target[:, 2]),
        )
