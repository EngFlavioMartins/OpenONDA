"""Ephemeral, safeguarded initial guesses for consecutive interface solves.

This is iteration history, not physical state or a cached operator result.
Native restarts deliberately start cold; every proposed trace must pass a
fresh complete FVM/renewal sweep and the unchanged interface residual gates.
"""

from __future__ import annotations

from functools import wraps
import hashlib

import numpy as np

from .parallel import collective_phase


def discard_interface_prediction_on_failure(method):
    """Leave no optimization history after an aborted coupled operation.

    This wrapper adds no collectives and never retries or suppresses an error.
    Local-only work still needs its own collective_phase before another rank
    can enter a numerical collective.
    """

    @wraps(method)
    def guarded(coupler, *args, **kwargs):
        try:
            return method(coupler, *args, **kwargs)
        except BaseException:
            history = getattr(coupler, "interface_predictor", None)
            if history is not None:
                history.reset()
            raise

    return guarded


def _discard_history_on_failure(method):
    """Make direct predictor lifecycle calls exception-safe too."""

    @wraps(method)
    def guarded(history, *args, **kwargs):
        try:
            return method(history, *args, **kwargs)
        except BaseException:
            history.reset()
            raise

    return guarded


def _same_arrays(left, right):
    return all(
        a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
        for a, b in zip(left, right, strict=True)
    )


def _copy_trace(trace):
    return tuple(np.array(value, copy=True) for value in trace)


def _clock(coupler):
    fvm = coupler.fvm_solver
    return int(fvm.step), float(fvm.time)


def _identity(coupler, geometry):
    """Recompute value identities; object references only detect replacement.

    No mutable array is admitted by object identity. Interface order and mesh
    geometry are hashed by value, and the existing native numerical identities
    cover solver/operator controls. Opaque external forcing is conservatively
    excluded: its hidden mutable state has no numerical fingerprint contract.
    """
    from source.solvers.fvm.io.backup import config_hash, mesh_hash

    from .backup import _backup_config, config_mapping_digest

    fvm = coupler.fvm_solver
    if not hasattr(fvm, "setup") or not hasattr(fvm, "mesh_data"):
        return None
    digest = hashlib.sha256()
    for value in geometry:
        array = np.ascontiguousarray(value)
        if not np.all(np.isfinite(array)):
            return None
        digest.update(repr((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    digest.update(config_hash(getattr(fvm, "_resolved_setup", fvm.setup)).encode())
    digest.update(mesh_hash(fvm.mesh_data).encode())
    digest.update(
        repr(
            (
                coupler.fvm_time_step_size,
                getattr(fvm, "time_step_size", None),
                coupler.vpm_time_step_size,
                coupler.n_fvm_substeps,
                coupler.vpm_particle_spacing,
                coupler.vpm_diffusion_grid_spacing,
                coupler.vpm_core_radius_ratio,
            )
        ).encode()
    )
    operators = [fvm, coupler.vorticity_transfer]
    if coupler._is_master:
        vpm = coupler.vpm_solver
        physics = vpm.physics
        rhs = vpm.stage_rhs
        if int(getattr(vpm, "n_sources", 0)) or any(
            getattr(physics, name, None) is not None
            for name in (
                "body_velocity",
                "body_velocity_field",
                "body_velocity_gradient",
                "body_velocity_gradient_field",
                "velocity_override",
                "velocity_override_gradient",
            )
        ):
            return None
        from source.solvers.vpm.config.fingerprint import _canonical_value
        from source.solvers.vpm.physics.stage_rhs import ParticleExternalStageContribution

        if any(
            type(provider) is not ParticleExternalStageContribution for provider in rhs.providers
        ):
            return None
        digest.update(config_mapping_digest(_backup_config(coupler)).encode())
        # Numerics.induction is a construction object; runtime backends are
        # cloned/bound separately. Include their live numerical controls too.
        digest.update(
            config_mapping_digest(
                {
                    "runtime_induction": _canonical_value(vpm.induction),
                    "rhs_induction": _canonical_value(rhs.induction),
                    "strength_enabled": rhs.strength_enabled,
                    "runtime_time_step_size": getattr(vpm, "time_step_size", None),
                }
            ).encode()
        )
        digest.update(np.asarray(coupler.freestream_velocity).tobytes())
        operators.extend((vpm, physics, vpm.induction, rhs, *rhs.providers))
        # The live RHS can be rebound independently of the serialized setup.
        operators.extend((rhs.induction, getattr(vpm.induction, "base", None)))
    return digest.hexdigest(), tuple(operators)


def _same_identity(left, right):
    return (
        left is not None
        and right is not None
        and left[0] == right[0]
        and len(left[1]) == len(right[1])
        and all(a is b for a, b in zip(left[1], right[1], strict=True))
    )


class SafeguardedInterfacePredictor:
    """Reuse only the last accepted renewal correction as an initial guess.

    Resetting, loading or beginning a solve clears history. A seed failing
    the residual gates gets one provisional probe and
    then the *complete* original Picard allowance. Execution exceptions remain
    fail-fast: arbitrary numerical, allocation or MPI failures are not retried.
    No history is saved in native checkpoints;
    restart reproducibility therefore remains residual-tolerance, not bitwise,
    equivalence between different initial guesses of the same interface solve.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Discard optimization history without touching any physical state."""
        self._history = None
        self._pending = None
        self._active_identity = None

    @_discard_history_on_failure
    def prepare(self, coupler, geometry, old, raw):
        """Collectively admit a seed; all ranks receive the same decision."""
        previous, self._history = self._history, None
        self._pending = None
        self._active_identity = None
        comm = getattr(coupler.fvm_solver.parallel, "comm", None)
        parallel = bool(coupler.fvm_solver.parallel.is_parallel)
        with collective_phase(comm, "interface predictor identity"):
            identity = _identity(coupler, geometry)
            self._active_identity = identity
            local = bool(
                previous is not None
                and _same_identity(identity, previous["identity"])
                and _clock(coupler) == previous["clock"]
            )
        valid = all(comm.allgather(local)) if parallel else local
        seed = None
        info = None
        with collective_phase(comm, "interface predictor seed construction"):
            if coupler._is_master:
                reason = "cold_or_changed_identity"
                if identity is None:
                    reason = "unsupported_identity"
                elif valid and _same_arrays(old, previous["endpoint"]):
                    correction = previous["correction"]
                    velocity = raw[0] + correction[0]
                    normal = np.einsum("ij,ij->i", velocity, geometry[1])
                    gradient = raw[2] + correction[2]
                    if all(np.all(np.isfinite(value)) for value in (velocity, normal, gradient)):
                        seed = velocity, normal, gradient
                        reason = "previous_accepted_correction"
                info = {"enabled": True, "attempted": seed is not None, "reason": reason}
            elif valid:
                # Root owns the global trace; advance_fvm scatters it. Copy
                # the worker's placeholder before collective error admission,
                # never between the decision broadcast and the FVM collective.
                seed = _copy_trace(raw)
        if parallel:
            info = comm.bcast(info, root=0)
        assert info is not None
        if not info["attempted"]:
            seed = None
        return seed, info

    @_discard_history_on_failure
    def stage(self, coupler, geometry, raw, endpoint, *, converged):
        """Stage history only; the driver commits it after health and output."""
        self._pending = None
        if not converged or self._active_identity is None:
            self._history = None
            return
        # Recheck after callbacks/trials: changed operators cannot seed the
        # following interval, even when this endpoint met its residual gates.
        current = _identity(coupler, geometry)
        if not _same_identity(current, self._active_identity):
            self._history = None
            return
        pending = {"identity": current, "clock": _clock(coupler)}
        if coupler._is_master:
            pending["endpoint"] = _copy_trace(endpoint)
            pending["correction"] = tuple(
                after - before for after, before in zip(endpoint, raw, strict=True)
            )
        self._pending = pending

    def commit(self):
        """Publish only history belonging to a successfully accepted exchange."""
        self._history, self._pending = self._pending, None
