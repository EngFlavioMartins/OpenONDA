"""Sampler base class, field-sample columns and sample-point generation.

Both solvers sample the flow at points on lines and on axis-aligned planes.
The points, the CSV columns and the output arguments of a sampler are the
same in both, so a sample from the FVM and one from the VPM compare row by
row.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import numpy as np

from .schedules import EverySteps, EveryTime, FinalOnly, Schedule, schedule_from_dict

__all__ = [
    "FIELD_SAMPLE_COLUMNS",
    "Sampler",
    "line_points",
    "surface_points",
]

FIELD_SAMPLE_COLUMNS = (
    "time",
    "position_x",
    "position_y",
    "position_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "vorticity_x",
    "vorticity_y",
    "vorticity_z",
)
"""Leading CSV columns of every field sample: time in s, position in m,
velocity in m/s and vorticity in 1/s. Solver-specific columns follow them."""


class Sampler:
    """Base class of the samplers of both solvers.

    A sampler owns its output file name, its schedule and whether it records
    the initial state. Subclasses take their geometry as positional
    arguments, then solver-specific numerical arguments, then these
    keyword-only output arguments, and extend :meth:`config_dict` with their
    own constructor arguments.

    Parameters
    ----------
    file_name : str or None, default=None
        Output file stem. ``None`` uses the lowercase class name without the
        ``sampler`` suffix.
    schedule : EverySteps, EveryTime or FinalOnly, default=EverySteps(1)
        Accepted states at which the sampler writes.
    initial : bool, default=True
        Whether the sampler writes the initial state.

    Raises
    ------
    TypeError
        If ``schedule`` is not a shared schedule or ``initial`` is not a bool.
    ValueError
        If ``file_name`` is empty or contains a path separator.
    """

    def __init__(
        self,
        *,
        file_name: str | None = None,
        schedule: Schedule = EverySteps(1),
        initial: bool = True,
    ) -> None:
        if file_name is not None and (
            not isinstance(file_name, str) or not file_name or "/" in file_name or "\\" in file_name
        ):
            raise ValueError("file_name must be a non-empty file stem without a directory")
        if not isinstance(schedule, EverySteps | EveryTime | FinalOnly):
            raise TypeError("schedule must be EverySteps, EveryTime or FinalOnly")
        if not isinstance(initial, bool):
            raise TypeError("initial must be a bool")
        self.file_name = file_name
        self.schedule = schedule
        self.initial = initial

    @property
    def name(self) -> str:
        """Output file stem: ``file_name`` or the lowercase class name."""
        return self.file_name or type(self).__name__.lower().removesuffix("sampler")

    def is_due(self, step: int, time: float, time_step_size: float) -> bool:
        """Return whether the accepted state at ``step`` and ``time`` (s) is written."""
        return self.schedule.is_due(step, time, time_step_size)

    def config_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible constructor arguments.

        Subclasses add their geometry and numerical arguments to this
        dictionary, under the constructor argument names, so
        :meth:`from_config` rebuilds an equal sampler.
        """
        return {
            "file_name": self.file_name,
            "schedule": self.schedule.to_dict(),
            "initial": self.initial,
        }

    @classmethod
    def from_config(cls, data: dict[str, Any]) -> Sampler:
        """Rebuild a sampler from :meth:`config_dict` data."""
        values = dict(data)
        values["schedule"] = schedule_from_dict(values["schedule"])
        return cls(**values)

    def __eq__(self, other: object) -> bool:
        """Samplers are equal when they have the same type and configuration."""
        return type(self) is type(other) and self.config_dict() == other.config_dict()

    # Samplers may cache interpolation stencils, so they are not hashable.
    __hash__ = None  # type: ignore[assignment]


def line_points(
    start: Sequence[float] | np.ndarray,
    end: Sequence[float] | np.ndarray,
    n_points: int | None = None,
    spacing: float | None = None,
) -> np.ndarray:
    """Return equally spaced points on a line, both end points included.

    Parameters
    ----------
    start, end : sequence of float or numpy.ndarray
        End points of the line in m, shape ``(3,)``.
    n_points : int or None, default=None
        Number of points, at least two.
    spacing : float or None, default=None
        Largest spacing between points in m. The line has
        ``max(2, ceil(length / spacing) + 1)`` points.

    Returns
    -------
    numpy.ndarray
        Points in m, shape ``(n_points, 3)``, float64.

    Raises
    ------
    ValueError
        If not exactly one of ``n_points`` and ``spacing`` is given, the end
        points are not finite three-vectors, ``n_points`` is below two, or
        ``spacing`` is not positive and finite.
    """
    if (n_points is None) == (spacing is None):
        raise ValueError("Give exactly one of n_points or spacing")
    first = np.asarray(start, dtype=np.float64)
    last = np.asarray(end, dtype=np.float64)
    if first.shape != (3,) or last.shape != (3,):
        raise ValueError("start and end must be three-vectors")
    if not (np.all(np.isfinite(first)) and np.all(np.isfinite(last))):
        raise ValueError("start and end must be finite")
    if spacing is not None:
        if not math.isfinite(spacing) or spacing <= 0.0:
            raise ValueError(f"spacing must be positive and finite, got {spacing}")
        length = float(np.linalg.norm(last - first))
        n_points = max(2, math.ceil(length / spacing) + 1)
    if isinstance(n_points, bool) or not isinstance(n_points, int | np.integer) or n_points < 2:
        raise ValueError(f"n_points must be an integer of at least 2, got {n_points}")
    fraction = np.linspace(0.0, 1.0, int(n_points))
    return first + fraction[:, None] * (last - first)


def surface_points(
    point: Sequence[float] | np.ndarray,
    normal: Sequence[float] | np.ndarray,
    bounds: Sequence[float] | np.ndarray,
    spacing: float,
) -> tuple[np.ndarray, tuple[int, int], tuple[str, str]]:
    """Return a uniform grid of points on an axis-aligned plane.

    Each in-plane axis from ``lower`` to ``upper`` has
    ``max(1, ceil((upper - lower) / spacing))`` equal intervals, so both
    bounds are grid points and the realized spacing is at most ``spacing``.

    Parameters
    ----------
    point : sequence of float or numpy.ndarray
        A point on the plane in m, shape ``(3,)``.
    normal : sequence of float or numpy.ndarray
        Plane normal, shape ``(3,)``. The plane is normal to the coordinate
        axis of its largest component.
    bounds : sequence of float or numpy.ndarray
        ``(lower_1, upper_1, lower_2, upper_2)`` in m on the two in-plane
        axes in coordinate order: (y, z) for an x-normal, (x, z) for a
        y-normal and (x, y) for a z-normal plane.
    spacing : float
        Largest grid spacing in m.

    Returns
    -------
    points : numpy.ndarray
        Grid points in m, shape ``(n_1 * n_2, 3)``, float64, with the second
        in-plane axis varying fastest.
    shape : tuple of int
        ``(n_1, n_2)``, the number of points on each in-plane axis.
    axes : tuple of str
        Names of the two in-plane axes, for example ``("x", "y")``.

    Raises
    ------
    ValueError
        If the point or normal is not a finite three-vector, the normal is
        zero, a lower bound is not below its upper bound, or ``spacing`` is
        not positive and finite.
    """
    origin = np.asarray(point, dtype=np.float64)
    direction = np.asarray(normal, dtype=np.float64)
    limits = np.asarray(bounds, dtype=np.float64)
    if origin.shape != (3,) or direction.shape != (3,):
        raise ValueError("point and normal must be three-vectors")
    if not (np.all(np.isfinite(origin)) and np.all(np.isfinite(direction))):
        raise ValueError("point and normal must be finite")
    if not np.any(direction):
        raise ValueError("normal must be non-zero")
    if limits.shape != (4,) or not np.all(np.isfinite(limits)):
        raise ValueError("bounds must be four finite values")
    if limits[0] >= limits[1] or limits[2] >= limits[3]:
        raise ValueError("each lower bound must be below its upper bound")
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError(f"spacing must be positive and finite, got {spacing}")

    normal_axis = int(np.argmax(np.abs(direction)))
    in_plane = [axis for axis in range(3) if axis != normal_axis]

    def axis_coordinates(lower: float, upper: float) -> np.ndarray:
        intervals = max(1, math.ceil((upper - lower) / spacing))
        return np.linspace(lower, upper, intervals + 1)

    first = axis_coordinates(float(limits[0]), float(limits[1]))
    second = axis_coordinates(float(limits[2]), float(limits[3]))
    grid_first, grid_second = np.meshgrid(first, second, indexing="ij")
    points = np.empty((grid_first.size, 3), dtype=np.float64)
    points[:, normal_axis] = origin[normal_axis]
    points[:, in_plane[0]] = grid_first.ravel()
    points[:, in_plane[1]] = grid_second.ravel()
    names = ("x", "y", "z")
    return points, grid_first.shape, (names[in_plane[0]], names[in_plane[1]])
