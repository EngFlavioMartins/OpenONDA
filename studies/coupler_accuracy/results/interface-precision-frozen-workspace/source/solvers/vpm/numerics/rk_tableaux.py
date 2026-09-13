"""Butcher tableaus for the coupled VPM Runge--Kutta schemes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RKTableau:
    """Validated explicit Runge--Kutta Butcher tableau.

    Parameters
    ----------
    name : str
        Stable configuration/reporting name.
    order : int
        Formal temporal order of accuracy.
    a : tuple[tuple[float, ...], ...]
        Square lower-triangular stage coefficient matrix. ``a[i][j]`` weights
        rate ``j`` when constructing stage ``i``.
    b : tuple[float, ...]
        Final-combination weights, one per stage.
    c : tuple[float, ...]
        Stage-time fractions; stage ``i`` is evaluated at ``t + c[i] * dt``.

    Attributes
    ----------
    stages : int
        Number of right-hand-side evaluations per accepted step.

    Notes
    -----
    The VPM integrator applies the same tableau to particle position and
    vortex strength. ``a`` must be explicit (zero diagonal and upper triangle)
    and all three coefficient arrays must have the same non-zero length.

    Raises
    ------
    ValueError
        If dimensions are inconsistent, the tableau is not explicit, or
        ``order`` is not positive.
    """

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
    """Heun's two-stage explicit second-order Runge--Kutta tableau.

    Construct without arguments. The VPM integrator evaluates rates at
    ``t`` and ``t + dt`` and combines them with equal weights. Position and
    vector circulation are advanced together; construction has no side effects.
    """

    def __init__(self) -> None:
        """Create the immutable two-stage Heun tableau."""
        super().__init__(
            name="RK2",
            order=2,
            a=((0.0, 0.0), (1.0, 0.0)),
            b=(0.5, 0.5),
            c=(0.0, 1.0),
        )


class SSPRK3(RKTableau):
    """Three-stage, third-order strong-stability-preserving RK tableau.

    Construct without arguments. Stage times are ``0``, ``dt``, and
    ``0.5 * dt``; the final weights are ``(1/6, 1/6, 2/3)``. The tableau is
    shared by VPM position and vector-circulation integration.
    """

    def __init__(self) -> None:
        """Create the immutable three-stage SSPRK3 tableau."""
        super().__init__(
            name="SSPRK3",
            order=3,
            a=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.25, 0.25, 0.0)),
            b=(1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0),
            c=(0.0, 1.0, 0.5),
        )


class RK4(RKTableau):
    """Classical four-stage, fourth-order explicit Runge--Kutta tableau.

    Construct without arguments. It evaluates two midpoint stages and one
    endpoint stage before the classical ``(1, 2, 2, 1) / 6`` combination.
    Position and vector circulation use the same temporary stage state.
    """

    def __init__(self) -> None:
        """Create the immutable classical RK4 tableau."""
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
