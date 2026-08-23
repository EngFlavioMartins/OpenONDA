"""
Aircraft geometry primitives for the VLM: WingSegment, Wing, and Aircraft
containers that generate lattice panels.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class WingSegment:
    """
    Quadrilateral wing segment with parametric definition.

    A segment is defined by 4 corner vertex_position (a, b, c, d):
        a -------- b    (Leading edge: a-b)
        |          |
        |          |
        d -------- c    (Trailing edge: d-c)

    Vertices are ordered counter-clockwise looking from above (+Z direction).
    Normal should point upward for typical wing orientation.

    Attributes:
        uid: Unique identifier for this segment
        vertex_position: Dict with keys 'a', 'b', 'c', 'd' mapping to [x, y, z] coordinates
        n_chordwise_panels: Number of chordwise panels
        n_spanwise_panels: Number of spanwise panels
        airfoils: Optional dict with 'inner' and 'outer' airfoil file paths
        geometry: Optional geometric properties (chord, span, sweep, dihedral, twist)
    """

    uid: str
    vertex_position: dict[str, np.ndarray]
    n_chordwise_panels: int = 8
    n_spanwise_panels: int = 16
    airfoils: dict[str, str] | None = None
    geometry: dict[str, float] | None = None

    def __post_init__(self):
        """Validate segment definition."""
        required_vertices = {"a", "b", "c", "d"}
        if not required_vertices.issubset(self.vertex_position.keys()):
            raise ValueError(
                f"Segment {self.uid} missing vertex_position. Required: {required_vertices}"
            )

        # Convert to numpy arrays if needed
        for key in self.vertex_position:
            if not isinstance(self.vertex_position[key], np.ndarray):
                self.vertex_position[key] = np.array(self.vertex_position[key], dtype=float)

        if self.n_chordwise_panels < 1:
            raise ValueError(f"n_chordwise_panels must be >= 1, got {self.n_chordwise_panels}")
        if self.n_spanwise_panels < 1:
            raise ValueError(f"n_spanwise_panels must be >= 1, got {self.n_spanwise_panels}")

    @property
    def area(self) -> float:
        """Approximate segment area (assumes planar quadrilateral)."""
        # Split into two triangles: (a,b,c) and (a,c,d)
        v1 = self.vertex_position["b"] - self.vertex_position["a"]
        v2 = self.vertex_position["c"] - self.vertex_position["a"]
        v3 = self.vertex_position["d"] - self.vertex_position["a"]

        area1 = 0.5 * np.linalg.norm(np.cross(v1, v2))
        area2 = 0.5 * np.linalg.norm(np.cross(v2, v3))
        return area1 + area2

    @property
    def centre(self) -> np.ndarray:
        """Segment centre point."""
        return 0.25 * (
            self.vertex_position["a"]
            + self.vertex_position["b"]
            + self.vertex_position["c"]
            + self.vertex_position["d"]
        )

    @property
    def normal(self) -> np.ndarray:
        """Segment normal vector (normalized)."""
        # Use diagonals to get normal
        diag1 = self.vertex_position["c"] - self.vertex_position["a"]
        diag2 = self.vertex_position["d"] - self.vertex_position["b"]
        n = np.cross(diag1, diag2)
        n_mag = np.linalg.norm(n)
        return n / n_mag if n_mag > 1e-10 else np.array([0.0, 0.0, 1.0])


@dataclass
class Wing:
    """
    Wing composed of one or more segments.

    Attributes:
        uid: Unique identifier
        segments: Ordered dict of WingSegment objects
        symmetry: Mirror symmetry plane
            0: No symmetry
            1: XY plane (mirror in Z)
            2: XZ plane (mirror in Y)
            3: YZ plane (mirror in X)
    """

    uid: str
    segments: dict[str, WingSegment] = field(default_factory=dict)
    symmetry: int = 0

    def __post_init__(self):
        """Validate wing definition."""
        if self.symmetry not in (0, 1, 2, 3):
            raise ValueError(f"symmetry must be 0, 1, 2, or 3, got {self.symmetry}")

    @property
    def area(self) -> float:
        """Total wing area (excluding symmetry)."""
        return sum(seg.area for seg in self.segments.values())

    @property
    def span(self) -> float:
        """Approximate wing span."""
        if not self.segments:
            return 0.0

        # Get all y-coordinates
        y_coords = []
        for seg in self.segments.values():
            for key in ["a", "b", "c", "d"]:
                y_coords.append(seg.vertex_position[key][1])

        return max(y_coords) - min(y_coords)

    def add_segment(self, segment: WingSegment):
        """Add a segment to this wing."""
        if segment.uid in self.segments:
            raise ValueError(f"Segment {segment.uid} already exists in wing {self.uid}")
        self.segments[segment.uid] = segment


@dataclass
class Aircraft:
    """
    Complete aircraft model consisting of multiple wings.

    Attributes:
        uid: Unique identifier
        wings: Dict of Wing objects
        refs: Reference values for coefficient calculations
            - 'area': Reference area (m²)
            - 'chord': Reference chord (m)
            - 'span': Reference span (m)
            - 'geometry_centre': Geometry centre [x, y, z] (m)
            - 'reference_point': Moment reference centre [x, y, z] (m)
    """

    uid: str
    wings: dict[str, Wing] = field(default_factory=dict)
    refs: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize reference values if not provided."""
        if not self.refs:
            self.compute_default_refs()

    def add_wing(self, wing: Wing):
        """Add a wing to this aircraft."""
        if wing.uid in self.wings:
            raise ValueError(f"Wing {wing.uid} already exists in aircraft {self.uid}")
        self.wings[wing.uid] = wing
        # Recompute reference values when geometry changes
        self.compute_default_refs()

    def compute_default_refs(self):
        """Compute default reference values from geometry."""
        if not self.wings:
            # Set minimal defaults
            self.refs = {
                "area": 1.0,
                "chord": 1.0,
                "span": 1.0,
                "geometry_centre": np.array([0.0, 0.0, 0.0]),
                "reference_point": np.array([0.0, 0.0, 0.0]),
            }
            return

        # Compute total area and span (accounting for symmetry)
        total_area = 0.0
        max_span = 0.0
        for wing in self.wings.values():
            wing_area = wing.area
            wing_span = wing.span
            # Account for symmetry (wings are mirrored)
            if wing.symmetry > 0:
                wing_area *= 2  # Mirrored half
                wing_span *= 2  # Full span
            total_area += wing_area
            max_span = max(max_span, wing_span)

        # Compute geometry centre
        total_vol = 0.0
        weighted_centre = np.zeros(3)
        for wing in self.wings.values():
            for seg in wing.segments.values():
                vol = seg.area
                weighted_centre += vol * seg.centre
                total_vol += vol

        geometry_centre = (
            weighted_centre / total_vol if total_vol > 1e-10 else np.array([0.0, 0.0, 0.0])
        )

        self.refs = {
            "area": total_area if total_area > 0 else 1.0,
            "chord": total_area / max_span if max_span > 0 else 1.0,
            "span": max_span,
            "geometry_centre": geometry_centre,
            "reference_point": geometry_centre.copy(),
        }

    @property
    def total_area(self) -> float:
        """Total aircraft area."""
        return sum(wing.area for wing in self.wings.values())

    def total_n_panels(self) -> int:
        """Total number of panels (including symmetry)."""
        count = 0
        for wing in self.wings.values():
            for seg in wing.segments.values():
                n_panels = seg.n_chordwise_panels * seg.n_spanwise_panels
                count += n_panels * (2 if wing.symmetry > 0 else 1)
        return count
