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

    A segment is defined by 4 corner vertices (a, b, c, d):
        a -------- b    (Leading edge: a-b)
        |          |
        |          |
        d -------- c    (Trailing edge: d-c)

    Vertices are ordered counter-clockwise looking from above (+Z direction).
    Normal should point upward for typical wing orientation.

    Attributes:
        uid: Unique identifier for this segment
        vertices: Dict with keys 'a', 'b', 'c', 'd' mapping to [x, y, z] coordinates
        panels_chord: Number of chordwise panels
        panels_span: Number of spanwise panels
        airfoils: Optional dict with 'inner' and 'outer' airfoil file paths
        geometry: Optional geometric properties (chord, span, sweep, dihedral, twist)
    """

    uid: str
    vertices: dict[str, np.ndarray]
    panels_chord: int = 8
    panels_span: int = 16
    airfoils: dict[str, str] | None = None
    geometry: dict[str, float] | None = None

    def __post_init__(self):
        """Validate segment definition."""
        required_vertices = {"a", "b", "c", "d"}
        if not required_vertices.issubset(self.vertices.keys()):
            raise ValueError(f"Segment {self.uid} missing vertices. Required: {required_vertices}")

        # Convert to numpy arrays if needed
        for key in self.vertices:
            if not isinstance(self.vertices[key], np.ndarray):
                self.vertices[key] = np.array(self.vertices[key], dtype=float)

        if self.panels_chord < 1:
            raise ValueError(f"panels_chord must be >= 1, got {self.panels_chord}")
        if self.panels_span < 1:
            raise ValueError(f"panels_span must be >= 1, got {self.panels_span}")

    @property
    def area(self) -> float:
        """Approximate segment area (assumes planar quadrilateral)."""
        # Split into two triangles: (a,b,c) and (a,c,d)
        v1 = self.vertices["b"] - self.vertices["a"]
        v2 = self.vertices["c"] - self.vertices["a"]
        v3 = self.vertices["d"] - self.vertices["a"]

        area1 = 0.5 * np.linalg.norm(np.cross(v1, v2))
        area2 = 0.5 * np.linalg.norm(np.cross(v2, v3))
        return area1 + area2

    @property
    def center(self) -> np.ndarray:
        """Segment center point."""
        return 0.25 * (
            self.vertices["a"] + self.vertices["b"] + self.vertices["c"] + self.vertices["d"]
        )

    @property
    def normal(self) -> np.ndarray:
        """Segment normal vector (normalized)."""
        # Use diagonals to get normal
        diag1 = self.vertices["c"] - self.vertices["a"]
        diag2 = self.vertices["d"] - self.vertices["b"]
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
                y_coords.append(seg.vertices[key][1])

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
            - 'gcenter': Geometry center [x, y, z] (m)
            - 'rcenter': Moment reference center [x, y, z] (m)
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
                "gcenter": np.array([0.0, 0.0, 0.0]),
                "rcenter": np.array([0.0, 0.0, 0.0]),
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

        # Compute geometry center
        total_vol = 0.0
        weighted_center = np.zeros(3)
        for wing in self.wings.values():
            for seg in wing.segments.values():
                vol = seg.area
                weighted_center += vol * seg.center
                total_vol += vol

        gcenter = weighted_center / total_vol if total_vol > 1e-10 else np.array([0.0, 0.0, 0.0])

        self.refs = {
            "area": total_area if total_area > 0 else 1.0,
            "chord": total_area / max_span if max_span > 0 else 1.0,
            "span": max_span,
            "gcenter": gcenter,
            "rcenter": gcenter.copy(),
        }

    @property
    def total_area(self) -> float:
        """Total aircraft area."""
        return sum(wing.area for wing in self.wings.values())

    def total_num_panels(self) -> int:
        """Total number of panels (including symmetry)."""
        count = 0
        for wing in self.wings.values():
            for seg in wing.segments.values():
                n_panels = seg.panels_chord * seg.panels_span
                count += n_panels * (2 if wing.symmetry > 0 else 1)
        return count
