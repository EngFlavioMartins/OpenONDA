"""Scoped Picard sweeps of one FVM interval with a fixed VPM predictor."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
import time

import numpy as np

from source.coupler import solver as driver
from source.coupler.vorticity_transfer import _particle_state_snapshot, _restore_particle_state
from source.solvers.fvm.io.backup import load_backup, save_backup

TRACE_FIELDS = ("velocity_boundary_condition", "normal_velocity_boundary_condition", "tangential_gradient_boundary_condition")
FVM_FIELDS = ("velocity", "kinematic_pressure", "volumetric_face_flux", "volumetric_face_flux_old",
              "volumetric_face_flux_older", "velocity_old", "velocity_older")


def read_trace(coupler, *, old=False, velocity=None):
    suffix = "_old" if old else ""
    # The driver passes its predicted vector velocity as an argument; only
    # normal velocity and tangential gradient have current-state attributes.
    if old:
        velocity = coupler._velocity_boundary_condition_old
    if velocity is None:
        raise ValueError("The predicted vector velocity must be supplied")
    values = (np.asarray(velocity, dtype=float).copy(),
              *(np.asarray(getattr(coupler, "_"+name+suffix), dtype=float).copy() for name in TRACE_FIELDS[1:]))
    n = len(values[0])
    if tuple(value.shape for value in values) != ((n, 3), (n,), (n, 3)) or not all(np.all(np.isfinite(v)) for v in values):
        raise ValueError("A finite complete mixed boundary trace is required")
    return values


def write_trace(coupler, values, *, old=False):
    suffix = "_old" if old else ""
    for name, value in zip(TRACE_FIELDS, values, strict=True):
        if not old and name == TRACE_FIELDS[0]:
            continue
        setattr(coupler, "_"+name+suffix, value.copy())


def fingerprint(coupler):
    """Hash numerical fields and clocks needed for this fixed-predictor map."""
    fvm, vpm = coupler.fvm_solver, coupler.vpm_solver
    values = {"fvm_"+name: np.asarray(getattr(fvm, name)) for name in FVM_FIELDS}
    values.update({"vpm_"+name: value for name, value in _particle_state_snapshot(vpm).items()})
    values.update({"post_trace_"+name: value for name, value in zip(TRACE_FIELDS, read_trace(coupler, old=True), strict=True)})
    values["clocks"] = np.array([fvm.time, fvm.step, fvm._n_committed_time_steps, fvm.time_step_size,
                                fvm._accepted_time_step_size, fvm._previous_time_step_size, vpm.time, vpm.step,
                                coupler.vorticity_transfer.step, fvm.max_courant_number, fvm._kinematic_viscosity])
    values["acceptance_counters"] = np.array([fvm._n_consecutive_accepted_steps[key] for key in sorted(fvm._n_consecutive_accepted_steps)])
    values["eddy_viscosity"] = np.empty(0) if fvm.eddy_viscosity is None else fvm.eddy_viscosity
    result = {}
    for name, value in values.items():
        array = np.ascontiguousarray(value)
        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
        result[name] = digest.hexdigest()
    return result


@contextmanager
def fixed_predictor_iteration(directory: Path, *, iterations: int, relaxation: float = 1.,
                              normal_tolerance: float = 1e-6, gradient_tolerance: float = 1e-6,
                              callback=None):
    """Repeat FVM/renewal from fixed states; always verify the first map replay.

    This experimental wrapper supports the serial mixed/flux-pressure,
    buffered-M4 path with no consistency band. It does not query a reference.
    Extra sweeps reuse the same accepted FVM start and the same advected VPM
    predictor. They neither advance VPM again nor renew a previously renewed
    particle field. An unconverged final sweep is explicitly reported.
    """
    if isinstance(iterations, bool) or int(iterations) != iterations or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if not 0 < relaxation <= 1 or not all(np.isfinite(v) and v > 0 for v in (normal_tolerance, gradient_tolerance)):
        raise ValueError("A relaxation in (0,1] and positive finite tolerances are required")
    directory.mkdir(parents=True, exist_ok=False)
    original_advance, original_update = driver.advance_fvm, driver.update_boundary_history_after_replacement
    original_transfer = driver.FVMVPMCoupler._transfer_vorticity_to_vpm
    runtime = {"state": None, "skip_update": None}
    records = []

    def capture_advance(coupler, *arguments):
        if (not coupler._is_master or coupler.fvm_solver.parallel.is_parallel
                or coupler.setup.boundary_condition_mode != "vorticity_mixed"
                or coupler.setup.transfer_method != "buffered_m4_renewal"
                or getattr(coupler, "fvm_consistency_band", None) is not None):
            raise ValueError("The interface-iteration study requires serial mixed/flux-pressure buffered M4 without a consistency band")
        if runtime["state"] is not None or runtime["skip_update"] is not None:
            raise RuntimeError("An earlier interface interval is unfinished")
        fvm = coupler.fvm_solver
        path = directory / f"start-fvm-{int(coupler.vpm_solver.step):06d}.npz"
        save_backup(fvm, path)
        runtime["state"] = {"coupler": coupler, "backup": path, "old": read_trace(coupler, old=True),
                            "candidate": read_trace(coupler, velocity=arguments[-1]), "start_time": float(fvm.time), "start_step": int(fvm.step)}
        return original_advance(coupler, *arguments)

    def iterate_transfer(coupler, *geometry):
        state = runtime["state"]
        if state is None:
            return original_transfer(coupler, *geometry)  # Initial synchronization.
        if state["coupler"] is not coupler:
            raise RuntimeError("The interface interval belongs to another coupler")
        started = time.perf_counter()
        fvm, vpm, transfer = coupler.fvm_solver, coupler.vpm_solver, coupler.vorticity_transfer
        predictor = _particle_state_snapshot(vpm)
        transfer_start_step = transfer.step
        predictor_clock = (float(vpm.time), int(vpm.step))
        area = np.asarray(geometry[2])
        candidate = state["candidate"]
        evaluation_count = 0

        def evaluate(value, *, restore):
            nonlocal evaluation_count
            if restore:
                load_backup(fvm, state["backup"])
                _restore_particle_state(vpm, predictor)
                transfer.step = transfer_start_step
                write_trace(coupler, state["old"], old=True)
                write_trace(coupler, value)
                np.testing.assert_array_equal([fvm.time, fvm.step], [state["start_time"], state["start_step"]])
                original_advance(coupler, *geometry, state["old"][0], value[0])
            result, _ = original_transfer(coupler, *geometry)
            original_update(coupler, *geometry)
            evaluation_count += 1
            np.testing.assert_array_equal([vpm.time, vpm.step], predictor_clock)
            assert fvm.step == state["start_step"]+coupler.n_fvm_substeps and transfer.step == transfer_start_step+1
            np.testing.assert_allclose(fvm.time, vpm.time, rtol=0, atol=1e-12)
            return read_trace(coupler, old=True), result

        for sweep in range(1, iterations+1):
            post, result = evaluate(candidate, restore=sweep > 1)
            replay = None
            if sweep == 1:
                first = fingerprint(coupler)
                post, result = evaluate(candidate, restore=True)
                repeated = fingerprint(coupler)
                differences = [name for name in first if first[name] != repeated[name]]
                if differences:
                    raise RuntimeError("Fixed-predictor map does not replay bitwise: "+", ".join(differences))
                replay = {"numerical_arrays_and_clocks": len(first), "bitwise_equal": True, "sha256": first}
            normal_change = post[1]-candidate[1]
            gradient_change = post[2]-candidate[2]
            normal_rms = float(np.sqrt(np.average(normal_change**2, weights=area)))
            gradient_rms = float(np.sqrt(np.average(np.sum(gradient_change**2, axis=1), weights=area)))
            converged = normal_rms <= normal_tolerance and gradient_rms <= gradient_tolerance
            row = {"coupling_step": int(vpm.step), "elapsed_time": float(vpm.time), "sweep": sweep,
                   "normal_residual_rms": normal_rms, "gradient_residual_rms": gradient_rms, "converged": converged,
                   "map_evaluations": evaluation_count, "map_replay": replay,
                   "accepted_fvm_step": int(fvm.step), "accepted_transfer_step": int(transfer.step)}
            records.append(row)
            if callback is not None:
                callback(coupler, row, candidate, post)
            if converged:
                break
            candidate = tuple(old+relaxation*(new-old) for old, new in zip(candidate, post, strict=True))
        runtime["state"] = None
        runtime["skip_update"] = coupler
        return result, time.perf_counter()-started

    def update_once(coupler, *geometry):
        if runtime["skip_update"] is coupler:
            runtime["skip_update"] = None  # The final renewal was already observed.
            return
        original_update(coupler, *geometry)

    driver.advance_fvm = capture_advance
    driver.FVMVPMCoupler._transfer_vorticity_to_vpm = iterate_transfer
    driver.update_boundary_history_after_replacement = update_once
    try:
        yield records
        if runtime["state"] is not None or runtime["skip_update"] is not None:
            raise RuntimeError("The interface-iteration run stopped inside an interval")
    finally:
        driver.advance_fvm = original_advance
        driver.FVMVPMCoupler._transfer_vorticity_to_vpm = original_transfer
        driver.update_boundary_history_after_replacement = original_update
