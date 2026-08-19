"""FVM blending-zone relaxation toward the VPM velocity."""

from __future__ import annotations

import logging

import numpy as np

_logger = logging.getLogger("coupler")
BLEND_STRENGTH = 4.0


def build_lambda(
    cell_centres: np.ndarray,
    fvm_box: tuple[float, float, float, float, float, float],
    overlap_zone_ramp_width: float,
    lambda_max: float,
    overlap_zone_dead_zone: float = 0.0,
) -> np.ndarray:
    """Return the C1 relaxation profile complementary to VPM authority."""
    points = np.atleast_2d(cell_centres)
    lo = np.asarray(fvm_box)[[0, 2, 4]]
    hi = np.asarray(fvm_box)[[1, 3, 5]]
    distance = np.minimum(points - lo, hi - points).min(axis=1)
    overlap_zone_dead_zone = max(overlap_zone_dead_zone, 0.0)
    width = max(overlap_zone_ramp_width - overlap_zone_dead_zone, 1.0e-30)
    relaxation = np.zeros(len(points))
    relaxation[distance <= overlap_zone_dead_zone] = lambda_max
    active = (distance > overlap_zone_dead_zone) & (distance < overlap_zone_ramp_width)
    phase = (distance[active] - overlap_zone_dead_zone) / width
    relaxation[active] = 0.5 * lambda_max * (1.0 + np.cos(np.pi * phase))
    return relaxation


def lambda_max_from_scales(
    u_char: float,
    overlap_zone_ramp_width: float,
    time_step_size: float,
) -> float:
    return float(
        min(
            BLEND_STRENGTH * u_char / max(overlap_zone_ramp_width, 1.0e-12),
            1.0 / max(time_step_size, 1.0e-12),
        )
    )


class BlendingZone:
    def __init__(self, cfg, vpm, fvm, *, coupling_time_step_size: float, fvm_box):
        self.cfg = cfg
        self.vpm = vpm
        self.fvm = fvm
        self.cell_centres = np.asarray(fvm.get_cell_center_coordinates()).reshape(-1, 3)
        u_char = float(np.linalg.norm(cfg.freestream_velocity_vector))
        lambda_max = lambda_max_from_scales(
            u_char, cfg.overlap_zone_ramp_width, coupling_time_step_size
        )
        overlap_zone_dead_zone = float(cfg.overlap_zone_dead_zone_width)
        self.relaxation = build_lambda(
            self.cell_centres,
            fvm_box,
            cfg.overlap_zone_ramp_width,
            lambda_max,
            overlap_zone_dead_zone,
        )
        self.fvm.set_cell_scalar_field("lambdaRelax", np.ascontiguousarray(self.relaxation))
        self._previous: np.ndarray | None = None
        self._next: np.ndarray | None = None
        _logger.info(
            "[Blending] strength=%.1f lambda_max=%.3e 1/s cells=%d",
            BLEND_STRENGTH,
            lambda_max,
            int(np.count_nonzero(self.relaxation)),
        )

    @property
    def active_cell_centres(self) -> np.ndarray:
        """Cell centres that require a VPM target evaluation."""
        return self.cell_centres[self.relaxation > 0.0]

    def update_target(self, active_velocity: np.ndarray | None = None) -> None:
        """Refresh the blending target from the shared VPM field evaluation."""
        active = self.relaxation > 0.0
        target = np.tile(self.cfg.freestream_velocity_vector, (len(self.cell_centres), 1)).astype(
            float
        )
        if active.any():
            if active_velocity is None:
                raise ValueError(
                    "active blending cells require velocities from the VPM boundary evaluation"
                )
            values = np.asarray(active_velocity, dtype=float).reshape(-1, 3)
            if len(values) != int(np.count_nonzero(active)):
                raise ValueError("active blending-zone velocity count does not match active cells")
            target[active] = values
        self._previous = target if self._next is None else self._next
        self._next = target
        self._push(target)

    def update_endpoint(self, active_velocity: np.ndarray | None = None) -> None:
        """Replace the interval endpoint after the vorticity transfer."""
        if self._next is None or active_velocity is None:
            return
        active = self.relaxation > 0.0
        values = np.asarray(active_velocity, dtype=float).reshape(-1, 3)
        if len(values) != int(np.count_nonzero(active)):
            raise ValueError(
                "resynchronised blending-zone velocity count does not match active cells"
            )
        endpoint = self._next.copy()
        endpoint[active] = values
        self._next = endpoint

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
