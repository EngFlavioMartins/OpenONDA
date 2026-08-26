"""Resolved-scale VPM-to-FVM consistency in the outer numerical buffer."""

from __future__ import annotations

import logging

import numpy as np

from source.coupler.reporting import format_coupler_log

logger = logging.getLogger("coupler")

CONSISTENCY_STRENGTH = 4.0
RATE_FIELD = "couplingConsistencyRate"
TARGET_FIELD = "couplingConsistencyTargetVelocity"


def build_consistency_rate(
    cell_centres: np.ndarray,
    fvm_box: np.ndarray,
    width: float,
    maximum_rate: float,
) -> np.ndarray:
    """Return a C1 boundary-attached relaxation profile.

    The rate is zero at least ``width`` from every outer FVM face and rises
    smoothly to ``maximum_rate`` at the numerical boundary.  When the transfer
    box is inset by ``width``, the source is therefore confined to the
    VPM-authoritative buffer and cannot modify the FVM-authoritative core.
    """
    points = np.asarray(cell_centres, dtype=np.float64).reshape(-1, 3)
    bounds = np.asarray(fvm_box, dtype=np.float64).reshape(6)
    width = float(width)
    maximum_rate = float(maximum_rate)
    if not np.isfinite(width) or width < 0.0:
        raise ValueError("consistency width must be finite and non-negative")
    if not np.isfinite(maximum_rate) or maximum_rate < 0.0:
        raise ValueError("maximum consistency rate must be finite and non-negative")
    if width == 0.0 or len(points) == 0:
        return np.zeros(len(points), dtype=np.float64)

    lo = bounds[::2]
    hi = bounds[1::2]
    distance_to_boundary = np.min(np.minimum(points - lo, hi - points), axis=1)
    phase = np.clip(distance_to_boundary / width, 0.0, 1.0)
    rate = 0.5 * maximum_rate * (1.0 + np.cos(np.pi * phase))
    rate[distance_to_boundary >= width] = 0.0
    return rate


def maximum_consistency_rate(
    freestream_speed: float,
    width: float,
    coupling_time_step_size: float,
) -> float:
    """Choose a transit-scale rate capped at one inverse coupling step."""
    speed = float(freestream_speed)
    width = float(width)
    time_step_size = float(coupling_time_step_size)
    if not np.isfinite(speed) or speed < 0.0:
        raise ValueError("freestream speed must be finite and non-negative")
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("consistency width must be finite and positive")
    if not np.isfinite(time_step_size) or time_step_size <= 0.0:
        raise ValueError("coupling time step must be finite and positive")
    return float(min(CONSISTENCY_STRENGTH * speed / width, 1.0 / time_step_size))


class FVMConsistencyBand:
    """Own the resolved VPM velocity target used by the FVM source term."""

    def __init__(
        self,
        setup,
        fvm_solver,
        *,
        coupling_time_step_size: float,
        fvm_box: np.ndarray,
    ) -> None:
        self.setup = setup
        self.fvm_solver = fvm_solver
        self.cell_centres = np.asarray(
            fvm_solver.get_cell_centre_coordinates(), dtype=np.float64
        ).reshape(-1, 3)
        width = float(setup.fvm_consistency_width)
        rate_maximum = maximum_consistency_rate(
            float(np.linalg.norm(setup.freestream_velocity_vector)),
            width,
            coupling_time_step_size,
        )
        self.rate = build_consistency_rate(self.cell_centres, fvm_box, width, rate_maximum)
        self._active = self.rate > 0.0
        self._previous: np.ndarray | None = None
        self._next: np.ndarray | None = None
        self.fvm_solver.set_cell_scalar_field(RATE_FIELD, np.ascontiguousarray(self.rate))

        if getattr(self.fvm_solver.parallel, "is_root", True):
            logger.info(
                format_coupler_log(
                    "fvm consistency band",
                    ("width", f"{width:.6g}", "m"),
                    ("maximum rate", f"{rate_maximum:.6g}", "1/s"),
                    ("active cells", f"{int(np.count_nonzero(self._active)):,}"),
                    ("source", "resolved-scale implicit relaxation"),
                )
            )

    @property
    def is_initialized(self) -> bool:
        return self._next is not None

    @property
    def active_cell_centres(self) -> np.ndarray:
        return self.cell_centres[self._active]

    def update_target(self, active_velocity: np.ndarray | None) -> None:
        """Store the two physical-time endpoints for FVM substep interpolation."""
        target = np.tile(self.setup.freestream_velocity_vector, (len(self.cell_centres), 1)).astype(
            np.float64
        )
        if np.any(self._active):
            if active_velocity is None:
                raise ValueError("active consistency cells require a VPM velocity target")
            values = np.asarray(active_velocity, dtype=np.float64).reshape(-1, 3)
            if len(values) != int(np.count_nonzero(self._active)):
                raise ValueError("VPM consistency target count does not match active FVM cells")
            if not np.all(np.isfinite(values)):
                raise ValueError("VPM consistency target must contain only finite values")
            target[self._active] = values
        self._previous = target if self._next is None else self._next
        self._next = target
        self._push(target)

    def update_endpoint(self, active_velocity: np.ndarray | None) -> None:
        """Replace the interval endpoint after same-time FVM-to-VPM renewal."""
        if self._next is None:
            raise RuntimeError("FVM consistency history must be initialized first")
        if not np.any(self._active):
            return
        if active_velocity is None:
            raise ValueError("active consistency cells require a post-renewal VPM target")
        values = np.asarray(active_velocity, dtype=np.float64).reshape(-1, 3)
        if len(values) != int(np.count_nonzero(self._active)):
            raise ValueError(
                "post-renewal consistency target count does not match active FVM cells"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("post-renewal consistency target must contain only finite values")
        endpoint = self._next.copy()
        endpoint[self._active] = values
        self._next = endpoint

    def push_target(self, alpha: float) -> None:
        """Push the linearly interpolated target for one native FVM substep."""
        if self._previous is None or self._next is None:
            raise RuntimeError("FVM consistency history must be initialized before advancing")
        fraction = float(np.clip(alpha, 0.0, 1.0))
        self._push(self._previous + fraction * (self._next - self._previous))

    def _push(self, target: np.ndarray) -> None:
        self.fvm_solver.set_cell_vector_field(
            TARGET_FIELD,
            np.ascontiguousarray(target[:, 0]),
            np.ascontiguousarray(target[:, 1]),
            np.ascontiguousarray(target[:, 2]),
        )


def evaluate_active_vpm_velocity(coupler) -> np.ndarray | None:
    """Evaluate the complete VPM velocity only at active consistency cells."""
    band = getattr(coupler, "fvm_consistency_band", None)
    if band is None or not coupler._is_master:
        return None
    assert coupler.vpm_solver is not None
    points = band.active_cell_centres
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float64)
    values = coupler.vpm_solver.compute_velocity_at_points(
        points,
        include_freestream=True,
        zone_mask=None,
        include_body=True,
    )
    velocity = np.asarray(values, dtype=np.float64).reshape(-1, 3)
    if velocity.shape != points.shape or not np.all(np.isfinite(velocity)):
        raise RuntimeError("VPM consistency evaluation returned invalid velocity data")
    return velocity
