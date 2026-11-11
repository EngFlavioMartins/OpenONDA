"""Wall projection of particles that cross an FVM wall."""

import numpy as np


class PlacedSolidBoundary:
    """A body-coordinate solid boundary seen from the VPM frame at one mesh pose.

    Queries map frame points to body coordinates; corrected positions and
    displacements map back. Particles that need no correction keep their
    stored positions exactly.

    Parameters
    ----------
    boundary : SolidBoundary
        Walls in body coordinates.
    frame : MeshFrame
        Frame of the rotating mesh.
    pose : RigidPose
        Mesh pose the walls take.
    time : float
        Time in s of ``pose``; it labels the geometry revision.
    """

    def __init__(self, boundary, frame, pose, time):
        self._boundary = boundary
        self._frame = frame
        self._pose = pose
        self.revision = (boundary.revision, float(time))
        corners = np.array(np.meshgrid(*boundary.bounds.reshape(3, 2), indexing="ij"))
        points = frame.from_body(corners.reshape(3, -1).T, pose)
        self.bounds = np.ravel(np.column_stack((points.min(axis=0), points.max(axis=0))))

    def _to_body(self, points):
        return self._frame.to_body(np.asarray(points, dtype=np.float64).reshape(-1, 3), self._pose)

    def contains(self, points, *, include_boundary=False):
        return self._boundary.contains(self._to_body(points), include_boundary=include_boundary)

    def near_surface(self, points, margin):
        return self._boundary.near_surface(self._to_body(points), margin)

    def segments_cross_wall(self, starts, ends):
        return self._boundary.segments_cross_wall(self._to_body(starts), self._to_body(ends))

    def constrain_motion(self, positions, spacing, *, starts=None, coordinate_scale=None):
        """Project wall crossings in body coordinates, at the storage precision."""
        original = np.asarray(positions)
        body = self._to_body(original).astype(original.dtype)
        body_starts = None if starts is None else self._to_body(starts)
        corrected, changed, _, maximum = self._boundary.constrain_motion(
            body, spacing, starts=body_starts, coordinate_scale=coordinate_scale
        )
        result = original.copy()
        result[changed] = self._frame.from_body(corrected[changed], self._pose)
        displacement = result[changed].astype(np.float64) - original[changed]
        return result, changed, displacement, maximum


class WallProjection:
    """Project Runge--Kutta stage positions and accepted positions back into the fluid.

    The first Runge--Kutta stage supplies the start of each particle path.
    Diagnostic and backup evaluations have no stage index and never change it.
    The strengths are unchanged; the displacements ``δx_p`` and the resulting
    impulse changes are reported.
    """

    def __init__(self, boundary, physics, spacing, logger):
        self.boundary = boundary
        self.physics = physics
        self.spacing = spacing
        self.logger = logger
        self.revision = boundary.revision
        self._reference = None
        self._pending_logs = None

    def begin_trial(self):
        """Hold the projection warnings until this Runge--Kutta step is accepted."""
        self._pending_logs = []

    def end_trial(self, *, accepted):
        pending = self._pending_logs or []
        self._pending_logs = None
        self._reference = None
        return pending if accepted else []

    def _project(self, position, strength, count, *, accepted, starts=None):
        if self.boundary.revision != self.revision:
            raise RuntimeError("Solid geometry changed after the GBD mask was built")
        positions = self.physics.download_vector_field(position, count)
        if count == 0:
            return positions, False
        # Only particles within reach of a wall can be inside it or cross it: a
        # correction is at most a quarter spacing, a path at most its own length.
        reach = 0.25 * self.spacing
        if starts is not None:
            reach += float(np.max(np.linalg.norm(positions - starts, axis=1), initial=0.0))
        near = self.boundary.near_surface(positions, reach)
        if starts is not None:
            near |= self.boundary.near_surface(starts, reach)
        rows = np.flatnonzero(near)
        if not len(rows):
            return positions, False
        corrected_rows, changed_rows, delta, maximum = self.boundary.constrain_motion(
            positions[rows],
            self.spacing,
            starts=None if starts is None else starts[rows],
            coordinate_scale=float(np.max(np.abs(positions), initial=0.0)),
        )
        if not np.any(changed_rows):
            return positions, False
        corrected = positions.copy()
        corrected[rows] = corrected_rows
        changed = np.zeros(count, dtype=bool)
        changed[rows[changed_rows]] = True
        strengths = self.physics.download_vector_field(strength, count)[changed]
        self.physics.upload_vector_array(corrected, position, count)
        impulse_change = 0.5 * np.cross(delta, strengths).sum(axis=0)
        budget = getattr(self.physics, "last_solid_projection", None)
        if budget is None:
            budget = {
                "stage_count": 0,
                "accepted_count": 0,
                "stage_displacement_l1": 0.0,
                "accepted_displacement_l1": 0.0,
                "stage_impulse_change": np.zeros(3),
                "accepted_impulse_change": np.zeros(3),
            }
            self.physics.last_solid_projection = budget
        kind = "accepted" if accepted else "stage"
        budget[f"{kind}_count"] += int(np.count_nonzero(changed))
        budget[f"{kind}_displacement_l1"] += float(np.linalg.norm(delta, axis=1).sum())
        budget[f"{kind}_impulse_change"] += impulse_change
        log_args = (
            "Solid-wall exclusion projected %d %s particles; max displacement %.3e m, "
            "|Gamma| L1 %.3e, impulse change %s",
            int(np.count_nonzero(changed)),
            kind,
            maximum,
            float(np.linalg.norm(strengths, axis=1).sum()),
            impulse_change,
        )
        if self._pending_logs is None:
            self.logger.warning(*log_args)
        else:
            self._pending_logs.append(log_args)
        return corrected, True

    def stage(self, state):
        """Project stage positions in place; return whether any particle moved."""
        stage = getattr(state, "stage_index", None)
        starts = self._reference if stage is not None and stage > 0 else None
        positions, moved = self._project(
            state.position,
            state.vortex_strength,
            int(state.count),
            accepted=False,
            starts=starts,
        )
        if stage == 0:
            self._reference = positions.copy()
        return moved

    def accepted(self, position, strength, count):
        self._project(position, strength, count, accepted=True, starts=self._reference)
        self._reference = None
