"""Lagrangian marker representation of an immersed body.

An :class:`ImmersedBody` is a cloud of surface markers ``X`` (Ns, 3) with a
target velocity ``U_target`` (Ns, 3) — zero for a fixed body, the local body
velocity for moving bodies.  Markers must be spaced ``alpha * h`` apart with
``alpha ≈ 1`` relative to the local Eulerian grid spacing ``h`` (Pinelli et
al. 2010; Constant et al. Table 1), which the factory methods enforce.
"""

from __future__ import annotations

import numpy as np


class ImmersedBody:
    """Marker cloud for one immersed obstacle.

    Attributes:
        name:     Identifier used in force logs.
        X:        Marker positions ``(Ns, 3)``.
        U_target: Desired fluid velocity at the markers ``(Ns, 3)``.
    """

    def __init__(self, name: str, X: np.ndarray, U_target: np.ndarray | None = None):
        self.name = name
        self.X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if self.X.shape[1] != 3:
            raise ValueError(f"Marker array must be (Ns, 3), got {self.X.shape}")
        if U_target is None:
            self.U_target = np.zeros_like(self.X)
        else:
            self.U_target = np.broadcast_to(
                np.asarray(U_target, dtype=np.float64), self.X.shape
            ).copy()

    @property
    def n_markers(self) -> int:
        return self.X.shape[0]

    # ------------------------------------------------------------------ #
    # Factories
    # ------------------------------------------------------------------ #

    @classmethod
    def cylinder_z(
        cls,
        centre,
        diameter: float,
        h: float,
        alpha: float = 1.0,
        name: str = "cylinder",
    ) -> ImmersedBody:
        """Circle of markers in the (x, y) plane (a z-extruded 2D cylinder).

        Markers are placed on the circle of the given diameter around
        ``centre`` (the z-coordinate of ``centre`` should be the mid-plane of
        the single-cell-thick 2D mesh), spaced ``alpha * h`` along the arc.

        Args:
            centre:   Cylinder centre ``[x, y, z]``.
            diameter: Cylinder diameter.
            h:        Local Eulerian grid spacing near the body.
            alpha:    Marker spacing / grid spacing ratio (default 1.0).
            name:     Body name for force logs.
        """
        centre = np.asarray(centre, dtype=np.float64)
        n = max(int(round(np.pi * diameter / (alpha * h))), 4)
        theta = 2.0 * np.pi * np.arange(n) / n
        X = np.empty((n, 3))
        X[:, 0] = centre[0] + 0.5 * diameter * np.cos(theta)
        X[:, 1] = centre[1] + 0.5 * diameter * np.sin(theta)
        X[:, 2] = centre[2]
        return cls(name, X)

    @classmethod
    def sphere(
        cls,
        centre,
        diameter: float,
        h: float,
        alpha: float = 1.0,
        name: str = "sphere",
    ) -> ImmersedBody:
        """Fibonacci-lattice sphere with ~one marker per ``(alpha h)^2`` area."""
        centre = np.asarray(centre, dtype=np.float64)
        n = max(int(round(np.pi * diameter**2 / (alpha * h) ** 2)), 8)
        k = np.arange(n) + 0.5
        phi = np.arccos(1.0 - 2.0 * k / n)
        golden = np.pi * (1.0 + np.sqrt(5.0))
        theta = golden * k
        r = 0.5 * diameter
        X = np.empty((n, 3))
        X[:, 0] = centre[0] + r * np.cos(theta) * np.sin(phi)
        X[:, 1] = centre[1] + r * np.sin(theta) * np.sin(phi)
        X[:, 2] = centre[2] + r * np.cos(phi)
        return cls(name, X)

    @classmethod
    def from_points(cls, X, U_target=None, name: str = "body") -> ImmersedBody:
        """Arbitrary marker cloud (e.g. sampled from an STL surface)."""
        return cls(name, X, U_target)
