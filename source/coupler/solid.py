"""Geometry-independent particle exclusion at static FVM walls."""

import numpy as np


class SolidParticleGuard:
    """Constrain RK candidates and accepted positions using the same geometry.

    The first RK stage supplies the physical path origin. Diagnostic and
    backup evaluations have no stage index and never change that origin.
    Circulation is untouched; displacement and impulse changes are reported.
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
        """Buffer projection warnings until this RK trial is accepted."""
        self._pending_logs = []

    def end_trial(self, *, accepted):
        pending = self._pending_logs or []
        self._pending_logs = None
        self._reference = None
        return pending if accepted else []

    def _project(self, position, strength, count, *, accepted, starts=None):
        if self.boundary.revision != self.revision:
            raise RuntimeError("Solid geometry changed after the GBD mask was built")
        positions = self.physics._download_vector_field(position, count)
        if count == 0:
            return positions
        corrected, changed, delta, maximum = self.boundary.constrain_motion(
            positions,
            self.spacing,
            starts=starts,
        )
        if not np.any(changed):
            return positions
        strengths = self.physics._download_vector_field(strength, count)[changed]
        self.physics._upload_vector_array(corrected, position, count)
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
        return corrected

    def stage(self, state):
        stage = getattr(state, "stage_index", None)
        starts = self._reference if stage is not None and stage > 0 else None
        positions = self._project(
            state.position,
            state.vortex_strength,
            int(state.count),
            accepted=False,
            starts=starts,
        )
        if stage == 0:
            self._reference = positions.copy()

    def accepted(self, position, strength, count):
        self._project(position, strength, count, accepted=True, starts=self._reference)
        self._reference = None
