"""Declarative panel-body setup shared by VPM and boundary-element coupling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PanelBodySetup:
    """Declarative setup for one closed STL body in the panel solver."""

    stl: str
    uid: str
    group_id: int = 0
    kinematics: object | None = None
    translation: tuple[float, float, float] | None = None
    rotation_degrees: tuple[float, float, float] | None = None
    rotation_centre: tuple[float, float, float] | None = None
    reference_area: float | None = None

    def __post_init__(self) -> None:
        if not str(self.stl).strip():
            raise ValueError("PanelBodySetup.stl must be a non-empty path")
        if not str(self.uid).strip():
            raise ValueError("PanelBodySetup.uid must be non-empty")
        for field_name in ("translation", "rotation_degrees", "rotation_centre"):
            value = getattr(self, field_name)
            if value is not None:
                if len(value) != 3:
                    raise ValueError(f"{field_name} must contain three coordinates")
                object.__setattr__(self, field_name, tuple(float(item) for item in value))
        if self.group_id < 0:
            raise ValueError("Panel body group_id must be non-negative")
        if self.reference_area is not None and self.reference_area <= 0.0:
            raise ValueError("Panel body reference_area must be positive when provided")


__all__ = ["PanelBodySetup"]
