"""Butcher tableaus for the coupled VPM Runge--Kutta schemes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RKTableau:
    """Validated explicit Runge--Kutta tableau."""

    name: str
    order: int
    a: tuple[tuple[float, ...], ...]
    b: tuple[float, ...]
    c: tuple[float, ...]

    def __post_init__(self) -> None:
        stages = len(self.b)
        if stages == 0 or len(self.a) != stages or len(self.c) != stages:
            raise ValueError("RK tableau arrays must have the same non-zero stage count")
        for row_index, row in enumerate(self.a):
            if len(row) != stages:
                raise ValueError("RK tableau coefficient matrix must be square")
            if any(value != 0.0 for value in row[row_index:]):
                raise ValueError("RK tableau must be explicit")
        if self.order < 1:
            raise ValueError("RK tableau order must be positive")

    @property
    def stages(self) -> int:
        """Number of right-hand-side evaluations per step."""
        return len(self.b)


class RK2(RKTableau):
    """Heun's explicit second-order method."""

    def __init__(self) -> None:
        super().__init__(
            name="RK2",
            order=2,
            a=((0.0, 0.0), (1.0, 0.0)),
            b=(0.5, 0.5),
            c=(0.0, 1.0),
        )


class SSPRK3(RKTableau):
    """Three-stage strong-stability-preserving third-order method."""

    def __init__(self) -> None:
        super().__init__(
            name="SSPRK3",
            order=3,
            a=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.25, 0.25, 0.0)),
            b=(1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0),
            c=(0.0, 1.0, 0.5),
        )


class RK4(RKTableau):
    """Classical four-stage fourth-order method."""

    def __init__(self) -> None:
        super().__init__(
            name="RK4",
            order=4,
            a=(
                (0.0, 0.0, 0.0, 0.0),
                (0.5, 0.0, 0.0, 0.0),
                (0.0, 0.5, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
            ),
            b=(1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0),
            c=(0.0, 0.5, 0.5, 1.0),
        )


__all__ = ["RK2", "RK4", "RKTableau", "SSPRK3"]
