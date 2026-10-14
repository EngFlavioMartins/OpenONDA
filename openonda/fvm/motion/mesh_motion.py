"""Configuration of the prescribed rigid motion of a whole FVM mesh.

The motion classes themselves are shared with the VLM surfaces and live in
:mod:`openonda.core.kinematics`; this module adds the initial placement of the
mesh and its description for the run metadata and restart digest.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
import math

import numpy as np

from openonda.core.kinematics import RigidMotion, RigidPose, axis_rotation_matrix, finite_vector

__all__ = ["MeshMotion", "mesh_motion_to_dict", "placement_pose"]


def placement_pose(
    translation: Sequence[float] | None,
    rotation_degrees: Sequence[float] | None,
    rotation_centre: Sequence[float] | None,
) -> RigidPose:
    """Return the initial placement used by :class:`MeshMotion`.

    The rotations about global x, then y, then z (degrees) act about
    ``rotation_centre`` and are followed by ``translation``, the convention of
    :class:`~openonda.vpm.boundary_elements.vlm.config.VLMSurface`.
    """
    rotation = np.eye(3)
    if rotation_degrees is not None:
        for axis, angle in zip(np.eye(3), rotation_degrees, strict=True):
            rotation = axis_rotation_matrix(axis, math.radians(angle)) @ rotation
    centre = np.zeros(3) if rotation_centre is None else np.asarray(rotation_centre, dtype=float)
    shift = np.zeros(3) if translation is None else np.asarray(translation, dtype=float)
    return RigidPose(rotation, centre - rotation @ centre + shift)


@dataclass(frozen=True, slots=True)
class MeshMotion:
    """Prescribed rigid-body motion of the whole FVM mesh.

    The mesh, every boundary patch on it, and every moment centre on it move
    with ``kinematics``. Wall patches become moving no-slip walls; inlet,
    outlet and freestream values stay in the absolute (inertial) frame. Cell
    velocity and pressure are absolute-frame quantities.

    Parameters
    ----------
    kinematics : RigidMotion
        Motion of the mesh after its initial placement.
    translation : array-like, shape (3,) or None
        Initial translation of the constructed mesh in m.
    rotation_degrees : array-like, shape (3,) or None
        Initial rotations about global x, y and z in degrees.
    rotation_centre : array-like, shape (3,) or None
        Pivot of the initial rotation in m, in mesh coordinates.

    Notes
    -----
    The rotation centre of ``kinematics`` is given in the placed (global)
    coordinates. A mesh with ``empty`` patches (a 2-D model) may only move in
    their plane.
    """

    kinematics: RigidMotion
    translation: tuple[float, float, float] | None = None
    rotation_degrees: tuple[float, float, float] | None = None
    rotation_centre: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kinematics, RigidMotion):
            raise TypeError("MeshMotion.kinematics must be a RigidMotion instance")
        for name in ("translation", "rotation_degrees", "rotation_centre"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, finite_vector(f"MeshMotion.{name}", value))

    def placement(self) -> RigidPose:
        """Return the initial placement pose of the constructed mesh."""
        return placement_pose(self.translation, self.rotation_degrees, self.rotation_centre)


def _motion_to_dict(motion: RigidMotion) -> dict[str, object]:
    """Describe one rigid motion by class name and constructor fields."""
    data: dict[str, object] = {"type": type(motion).__name__}
    for item in fields(motion):
        value = getattr(motion, item.name)
        if item.name == "components":
            value = [_motion_to_dict(component) for component in value]
        data[item.name] = value
    return data


def mesh_motion_to_dict(motion: MeshMotion) -> dict[str, object]:
    """Describe a mesh motion by its placement and kinematics.

    Velocity functions of a :class:`~openonda.core.kinematics.ManeuverMotion`
    are kept as the function objects; the restart digest identifies them by
    qualified name.
    """
    return {
        "kinematics": _motion_to_dict(motion.kinematics),
        "translation": motion.translation,
        "rotation_degrees": motion.rotation_degrees,
        "rotation_centre": motion.rotation_centre,
    }
