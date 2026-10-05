"""Bounded process execution and measurements for the cylinder study."""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import suppress
from dataclasses import replace
import json
from numbers import Integral
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

import numpy as np
import psutil
from scipy.integrate import trapezoid

from openonda.cylinder_startup import startup_velocity


def positive_coupling_steps(value: str) -> int:
    """Parse a positive accepted-exchange limit for the cylinder CLI."""
    try:
        steps = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if steps < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return steps


def run_coupled_cylinder(
    build_case,
    *,
    start_from="latest",
    output_root=None,
    end_time=None,
    restart_from=None,
    max_coupling_steps=None,
    overrides=None,
    startup_duration: float | None = None,
    startup_transition_duration: float = 0.0,
    steady_freestream_velocity: tuple[float, float, float] | None = None,
) -> int:
    """Execute a local physical factory with native continuation and owned output.

    Supplying both startup options returns the factory's startup freestream
    to ``steady_freestream_velocity`` by ``startup_duration``. A positive
    ``startup_transition_duration`` removes the trigger with a C2 quintic
    taper during the final part of startup; zero retains the original step
    switch. Both durations must end on exchange boundaries. Always rebuild
    the factory with the startup background, including on resume: its transfer lattice
    must retain the same buffer. The startup speed must cover the steady
    speed. Force normalization and viscosity remain the factory's choices.

    The optional schedule is recorded in ``solution/cylinder_startup.json``
    and checked on resume, alongside strict native checkpoint admission.
    A checkpoint copied elsewhere needs this sidecar in its parent folder.
    Exact legacy sidecars preserve the original switch during continuation;
    a fresh run is needed to apply the taper to existing histories.
    ``max_coupling_steps`` caps the total accepted steps across all stages.
    """
    from openonda.coupler import create_coupler

    staged = startup_duration is not None or steady_freestream_velocity is not None
    if (staged and (startup_duration is None or steady_freestream_velocity is None)) or (
        not staged and startup_transition_duration != 0
    ):
        raise ValueError(
            "startup_duration and steady_freestream_velocity must be supplied together"
        )
    fvm_setup, vpm_case, coupler_setup, mesh = build_case(end_time=end_time, overrides=overrides)
    schedule = None
    if staged:
        schedule = _cylinder_startup_schedule(
            fvm_setup,
            vpm_case,
            coupler_setup,
            startup_duration,
            steady_freestream_velocity,
            startup_transition_duration,
        )
        if max_coupling_steps is not None and (
            isinstance(max_coupling_steps, bool)
            or not isinstance(max_coupling_steps, Integral)
            or max_coupling_steps <= 0
        ):
            raise ValueError("max_coupling_steps must be a positive integer or None")
    with create_coupler(
        fvm_setup, vpm_case, coupler_setup, mesh=mesh, case_dir=output_root
    ) as solver:
        if schedule is not None:
            return _run_cylinder_startup(
                solver, vpm_case, schedule, start_from, restart_from, max_coupling_steps
            )
        if restart_from is None:
            solver.initialize()
            induction = vpm_case.numerics.induction
            initialize_cylinder_perturbation(solver.fvm_solver, induction.z_max - induction.z_min)
        return solver.run(
            restart_from=restart_from,
            start_from=start_from if restart_from is None else None,
            max_coupling_steps=max_coupling_steps,
            backup_at_stop=max_coupling_steps is not None,
        )


def _cylinder_startup_schedule(
    flow, particles, policy, duration, steady, transition_duration=0.0
) -> dict:
    """Validate a fixed physical schedule before allocating solver resources."""
    duration = float(duration)
    transition_duration = float(transition_duration)
    startup = np.asarray(policy.freestream_velocity, dtype=np.float64)
    steady = np.asarray(steady, dtype=np.float64)
    if steady.shape != (3,) or not np.all(np.isfinite(steady)):
        raise ValueError("steady_freestream_velocity must contain three finite values")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("startup_duration must be finite and positive")
    if (
        not np.isfinite(transition_duration)
        or transition_duration < 0
        or transition_duration > duration
    ):
        raise ValueError(
            "startup_transition_duration must be finite and lie in [0, startup_duration]"
        )
    if np.linalg.norm(steady) > np.linalg.norm(startup) + 1e-12:
        raise ValueError("Construct the startup lattice with at least the steady freestream speed")
    exchange_dt = float(particles.numerics.time_step_size)
    fvm_dt = float(flow.time.time_step_size)
    end_time = float(flow.time.end_time)
    if flow.time.start_time != 0 or flow.time.adjustment is not None:
        raise ValueError("Cylinder startup requires fixed FVM steps starting at time zero")
    switch_step = round(duration / exchange_dt)
    transition_steps = round(transition_duration / exchange_dt)
    end_step = round(end_time / exchange_dt)
    if switch_step < 1 or not np.isclose(switch_step * exchange_dt, duration, rtol=0, atol=1e-12):
        raise ValueError("startup_duration must be an integer number of coupling steps")
    if not np.isclose(transition_steps * exchange_dt, transition_duration, rtol=0, atol=1e-12):
        raise ValueError("startup_transition_duration must be an integer number of coupling steps")
    if (
        not np.isclose(end_step * exchange_dt, end_time, rtol=0, atol=1e-12)
        or particles.run.steps != end_step
        or not np.isclose(round(exchange_dt / fvm_dt) * fvm_dt, exchange_dt, rtol=0, atol=1e-12)
    ):
        raise ValueError(
            "Cylinder startup requires matching FVM/VPM end times and integer substeps"
        )
    schedule = {
        "schema": "openonda-cylinder-startup/1",
        "startup_duration": duration,
        "switch_step": switch_step,
        "startup_freestream_velocity": startup.tolist(),
        "steady_freestream_velocity": steady.tolist(),
        "exchange_time_step": exchange_dt,
        "fvm_time_step": fvm_dt,
        "end_time": end_time,
        "initial_disturbance": "compact divergence-free 3D curl; amplitude 0.001",
    }
    if transition_duration > 0:
        schedule.update(
            schema="openonda-cylinder-startup/2",
            startup_transition_duration=transition_duration,
            transition_step=switch_step - transition_steps,
            transition_shape="quintic C2",
        )
    return schedule


def _set_cylinder_freestream(solver, velocity) -> None:
    """Change the runtime background without rebuilding transfer geometry."""
    background = tuple(float(value) for value in velocity)
    solver.setup = replace(solver.setup, freestream_velocity=list(background))
    solver.freestream_velocity = np.asarray(background, dtype=np.float64)
    solver.vorticity_transfer.config = solver.setup
    if solver._is_master:
        particles = solver.vpm_solver
        particles.setup = replace(particles.setup, freestream_velocity=background)
        particles.case = replace(particles.case, numerics=particles.setup)
        particles._set_freestream_velocity(background)


def _admit_cylinder_startup_schedule(recorded, requested, *, allow_legacy=False):
    """Admit the exact policy, or an authenticated original step-switch run.

    A newly configured taper cannot change already published physics. When
    selecting a native checkpoint, the exact schema-1 counterpart is retained
    for its entire continuation. Unrelated schedule differences stay strict.
    """
    if recorded == requested:
        return recorded
    if allow_legacy and requested.get("schema") == "openonda-cylinder-startup/2":
        legacy = dict(requested)
        legacy["schema"] = "openonda-cylinder-startup/1"
        for key in ("startup_transition_duration", "transition_step", "transition_shape"):
            legacy.pop(key)
        if recorded == legacy:
            return recorded
    raise ValueError("Staged resume requires an identical cylinder_startup.json schedule")


def _startup_velocity_at_step(schedule, step):
    return startup_velocity(
        step * schedule["exchange_time_step"],
        schedule["startup_duration"],
        schedule.get("startup_transition_duration", 0.0),
        schedule["startup_freestream_velocity"],
        schedule["steady_freestream_velocity"],
    )


def _run_smooth_cylinder_startup(solver, schedule, selected, step, limit):
    """Apply each accepted endpoint inside one native coupling invocation."""
    from source.coupler.parallel import collective_phase

    missing = object()
    previous_override = solver.__dict__.get("_advance_vpm", missing)
    original_advance = solver._advance_vpm
    previous_velocity = tuple(solver.setup.freestream_velocity)
    startup = tuple(schedule["startup_freestream_velocity"])
    steady = tuple(schedule["steady_freestream_velocity"])
    transition_step, switch_step = schedule["transition_step"], schedule["switch_step"]

    def advance(step, time_end):
        nonlocal previous_velocity
        if step <= transition_step:
            velocity = startup
        elif step >= switch_step:
            velocity = steady
        else:
            velocity = _startup_velocity_at_step(schedule, step)
        if velocity != previous_velocity:
            # Every rank updates its local background before the existing
            # collective VPM advance. The native loop then refreshes the
            # boundary, transfers vorticity and publishes this endpoint state.
            with collective_phase(solver._comm, "cylinder startup background"):
                _set_cylinder_freestream(solver, velocity)
            previous_velocity = velocity
        return original_advance(step, time_end)

    solver._advance_vpm = advance
    try:
        if selected is None:
            return solver.run(start_from="initial", max_coupling_steps=limit, backup_at_stop=True)
        return solver.solve(start_step=step, max_coupling_steps=limit, backup_at_stop=True)
    finally:
        if previous_override is missing:
            del solver._advance_vpm
        else:
            solver._advance_vpm = previous_override


def _run_cylinder_startup(solver, particles, schedule, start_from, restart_from, limit) -> int:
    """Run accepted-clock segments, admitting each saved stage before switching."""
    from source.coupler.parallel import collective_phase
    from source.restart import select_backup

    metadata = solver.solution_dir / "cylinder_startup.json"
    selected = saved_velocity = None
    switch = schedule["switch_step"]
    startup = schedule["startup_freestream_velocity"]
    steady = schedule["steady_freestream_velocity"]
    with collective_phase(solver._comm, "cylinder startup restart selection"):
        if solver._is_master:
            selector = restart_from if restart_from is not None else start_from
            selected = select_backup(selector, directory=solver.solution_dir, kind="coupled")
            if selected is not None:
                identity = selected.parent / metadata.name
                if not identity.is_file():
                    raise ValueError("Staged resume requires an identical cylinder_startup.json")
                schedule = _admit_cylinder_startup_schedule(
                    json.loads(identity.read_text()), schedule, allow_legacy=True
                )
                saved = json.loads((selected / "manifest.json").read_text())
                step = saved["coupling_step"]
                if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                    raise ValueError("Invalid startup checkpoint coupling_step")
                saved_velocity = saved["config"]["coupler"]["freestream_velocity"]
                # A stop exactly at the switch may store either side of it.
                if schedule.get("startup_transition_duration", 0) > 0:
                    permitted = [list(_startup_velocity_at_step(schedule, step))]
                else:
                    permitted = [startup] if step < switch else [steady]
                    if step == switch:
                        permitted.append(startup)
                if saved_velocity not in permitted:
                    raise ValueError("Checkpoint freestream is inconsistent with its startup stage")
                if metadata.exists():
                    _admit_cylinder_startup_schedule(json.loads(metadata.read_text()), schedule)
            elif start_from != "initial" and metadata.exists():
                _admit_cylinder_startup_schedule(json.loads(metadata.read_text()), schedule)
    if solver._comm is not None:
        selected, saved_velocity, schedule = solver._comm.bcast(
            (selected, saved_velocity, schedule), root=0
        )

    # Initialization always uses the factory's startup background, so native
    # geometry checks see the same transfer lattice on both sides of the switch.
    solver.initialize()
    step = 0
    if selected is not None:
        with collective_phase(solver._comm, "cylinder saved freestream policy"):
            _set_cylinder_freestream(solver, saved_velocity)
        step = solver.load_backup(selected)
        if schedule.get("startup_transition_duration", 0) > 0:
            with collective_phase(solver._comm, "cylinder restored startup background"):
                _set_cylinder_freestream(solver, _startup_velocity_at_step(schedule, step))
        elif step >= switch:
            with collective_phase(solver._comm, "cylinder steady freestream policy"):
                _set_cylinder_freestream(solver, steady)
    else:
        induction = particles.numerics.induction
        initialize_cylinder_perturbation(
            solver.fvm_solver,
            induction.z_max - induction.z_min,
            freestream_velocity=startup,
        )

    with collective_phase(solver._comm, "cylinder startup schedule record"):
        if solver._is_master and (not metadata.exists() or start_from == "initial"):
            temporary = metadata.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(schedule, indent=2, sort_keys=True) + "\n")
            temporary.replace(metadata)
    end_step = round(schedule["end_time"] / schedule["exchange_time_step"])
    if step >= end_step:
        return step
    if schedule.get("startup_transition_duration", 0) > 0:
        return _run_smooth_cylinder_startup(solver, schedule, selected, step, limit)
    first_limit = min(switch - step, end_step - step) if step < switch else end_step - step
    if limit is not None:
        first_limit = min(first_limit, limit)
    if selected is None:
        accepted = solver.run(
            start_from="initial", max_coupling_steps=first_limit, backup_at_stop=True
        )
    else:
        accepted = solver.solve(
            start_step=step, max_coupling_steps=first_limit, backup_at_stop=True
        )
    remaining = None if limit is None else limit - (accepted - step)
    if accepted == switch and accepted < end_step and (remaining is None or remaining > 0):
        with collective_phase(solver._comm, "cylinder steady freestream policy"):
            _set_cylinder_freestream(solver, steady)
        accepted = solver.solve(
            start_step=accepted, max_coupling_steps=remaining, backup_at_stop=True
        )
    return accepted


def cylinder_initial_velocity(
    positions: np.ndarray, span: float, amplitude: float = 1e-3
) -> np.ndarray:
    """Return a reproducible divergence-free 3D perturbation of unit inflow.

    Coordinates and span are in metres for the D=1, U=1 cylinder case. The
    perturbation is curl(0, A_y, A_z), with compact XY support around x=.65,
    sinusoidal A_y vanishing at the slip planes and span-independent A_z.
    The latter breaks transverse reflection symmetry to seed shedding without
    relying on mesh asymmetry or roundoff. Thus w=0 and the normal derivative
    of tangential velocity vanishes on both planes. Amplitude is
    the dimensionless vector-potential coefficient; this is an initial test
    disturbance, not a sustained forcing or a prescribed turbulent state.
    """
    if not np.isfinite(span) or span <= 0 or not np.isfinite(amplitude):
        raise ValueError("span must be positive and perturbation amplitude finite")
    position = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    x, y, z = position.T
    radius_squared = 0.5**2
    support = np.maximum(1 - ((x - 0.65) ** 2 + y**2) / radius_squared, 0)
    wave_number = np.pi / span
    phase = wave_number * (z + 0.5 * span)
    velocity = np.zeros_like(position)
    velocity[:, 0] = (
        1
        - amplitude * wave_number * support**4 * np.cos(phase)
        - 8 * amplitude * y / radius_squared * support**3
    )
    velocity[:, 1] = 8 * amplitude * (x - 0.65) / radius_squared * support**3
    velocity[:, 2] = -8 * amplitude * (x - 0.65) / radius_squared * support**3 * np.sin(phase)
    return velocity


def initialize_cylinder_perturbation(
    solver, span: float, *, freestream_velocity=(1.0, 0.0, 0.0)
) -> None:
    """Install the same small 3D initial disturbance on each local FVM mesh."""
    count = solver.mesh_data["n_cells"]
    velocity = cylinder_initial_velocity(solver.geo_data["cell_centre"][:count], span)
    velocity += np.asarray(freestream_velocity, dtype=np.float64) - [1.0, 0.0, 0.0]
    solver.set_initial_velocity(velocity)


def profile_statistics(path: Path, start: float = 40, end: float = 100) -> dict:
    """Time-weight a transverse velocity profile over an exact covered window.

    Returns y coordinates, three-component means and fluctuation RMS values
    in SI units. The physical span probe uses fixed x=1 transverse lines.
    Missing/duplicate timestamps and changing spatial support are rejected.
    """
    data = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    required = ("time", "position_y", "velocity_x", "velocity_y", "velocity_z")
    if not all(name in (data.dtype.names or ()) for name in required):
        raise ValueError(f"missing velocity profile columns: {path}")
    if any(not np.all(np.isfinite(data[name])) for name in required):
        raise ValueError(f"nonfinite profile values: {path}")
    coordinates = np.unique(data["position_y"])
    means, fluctuations = [], []
    common_times = None
    for coordinate in coordinates:
        rows = data[data["position_y"] == coordinate]
        time_values = rows["time"]
        if len(rows) < 2 or np.any(np.diff(time_values) <= 0):
            raise ValueError(f"profile times are not strictly increasing: {path}")
        if common_times is None:
            common_times = time_values
        elif not np.array_equal(common_times, time_values):
            raise ValueError(f"profile spatial support changes with time: {path}")
        if time_values[0] > start or time_values[-1] < end:
            raise ValueError(f"profile does not cover [{start}, {end}]: {path}")
        times = np.r_[start, time_values[(time_values > start) & (time_values < end)], end]
        values = np.column_stack(
            [np.interp(times, time_values, rows[f"velocity_{axis}"]) for axis in "xyz"]
        )
        mean = trapezoid(values, times, axis=0) / (end - start)
        centered = values - mean
        variance = np.sum(
            np.diff(times)[:, None]
            * (centered[:-1] ** 2 + centered[:-1] * centered[1:] + centered[1:] ** 2)
            / 3,
            axis=0,
        ) / (end - start)
        rms = np.sqrt(np.maximum(variance, 0))
        means.append(mean.tolist())
        fluctuations.append(rms.tolist())
    return {"y": coordinates.tolist(), "mean_velocity": means, "rms_velocity": fluctuations}


def compare_profiles(candidate: dict, reference: dict) -> dict:
    """Return spatial L2 errors of time-mean and RMS velocity, normalized by U=1."""
    left, right = np.asarray(candidate["y"]), np.asarray(reference["y"])
    if min(len(left), len(right)) < 2 or not np.allclose(left[[0, -1]], right[[0, -1]], atol=1e-7):
        raise ValueError("profile endpoints must describe the same physical observation line")
    coordinate = np.unique(np.r_[left, right])
    result = {}
    for field in ("mean_velocity", "rms_velocity"):
        a, b = np.asarray(candidate[field]), np.asarray(reference[field])
        difference = np.column_stack(
            [
                np.interp(coordinate, left, a[:, axis]) - np.interp(coordinate, right, b[:, axis])
                for axis in range(3)
            ]
        )
        result[field + "_l2"] = float(
            np.sqrt(
                trapezoid(np.sum(difference**2, axis=1), coordinate)
                / (coordinate[-1] - coordinate[0])
            )
        )
    return result


def _normalized_trial_command(command: list[str]) -> list[str]:
    """Normalize resume invocations so repeated attempts share one identity."""
    return [argument for argument in command if argument != "--resume"]


def _prior_trial_attempts(
    path: Path,
    normalized_command: list[str],
    *,
    accept_command_changes: bool = False,
) -> list[dict]:
    """Read and validate prior attempts for a resumed trial."""
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot resume malformed trial record: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"cannot resume malformed trial record: {path}")

    def validate(attempts: object) -> list[dict]:
        if not isinstance(attempts, list) or not all(isinstance(item, dict) for item in attempts):
            raise ValueError(f"cannot resume malformed trial attempts: {path}")
        for item in attempts:
            command = item.get("command")
            wall_seconds = item.get("wall_seconds")
            if (
                not isinstance(command, list)
                or not all(isinstance(argument, str) for argument in command)
                or isinstance(wall_seconds, bool)
                or not isinstance(wall_seconds, (int, float))
                or not np.isfinite(wall_seconds)
                or wall_seconds < 0.0
            ):
                raise ValueError(f"cannot resume malformed trial attempt: {path}")
        return attempts

    if payload.get("normalized_command") == normalized_command and "attempts" in payload:
        return validate(payload["attempts"])
    if accept_command_changes and "attempts" in payload:
        return validate(payload["attempts"])
    if accept_command_changes:
        raise ValueError(f"cannot resume trial record with unsupported schema: {path}")
    return []


class _ProcessTreeRSSSampler:
    """Sample a worker and all descendants without coupling to solver code."""

    def __init__(self, pid: int, period: float = 0.25):
        try:
            self._process = psutil.Process(pid)
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            self._process = None
        self.period = period
        self.peak_bytes = 0
        self.sample_count = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        if self._process is None:
            return
        try:
            children = self._process.children(recursive=True)
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            return
        rss = 0
        for process in (self._process, *children):
            try:
                rss += process.memory_info().rss
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        if rss <= 0:
            return
        self.peak_bytes = max(self.peak_bytes, rss)
        self.sample_count += 1

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.period)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, 2.0 * self.period))


def run_trial(
    command: list[str],
    directory: Path,
    *,
    cwd: Path,
    wall_limit: float,
    resume: bool = False,
) -> dict:
    """Run one case, retain its log, and stop its MPI process group on timeout.

    Parameters
    ----------
    command : list of str
        Explicit Python command and arguments; no shell interpretation.
    directory : Path
        Trial record directory, separate from the worker's output directory.
    cwd : Path
        Working directory for the child process.
    wall_limit : float
        Positive total wall-clock budget in seconds for this case, including
        meshing and startup. Compatible resumed attempts consume the same
        budget.
    resume : bool, default False
        Treat an existing compatible trial record as a resumed attempt. The
        command's literal ``--resume`` flag also enables this mode so existing
        pipeline callers retain cumulative execution accounting.

    Returns
    -------
    dict
        Command, status, elapsed time and log path. The same record is saved
        as ``trial.json`` even when a process fails or exceeds its allowance.
    """
    if not np.isfinite(wall_limit) or wall_limit <= 0:
        raise ValueError("wall_limit must be positive and finite")
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / "console.log"
    trial_path = directory / "trial.json"
    normalized_command = _normalized_trial_command(command)
    resumed = resume or "--resume" in command
    prior_attempts = (
        _prior_trial_attempts(trial_path, normalized_command, accept_command_changes=True)
        if resumed
        else []
    )
    prior_wall_seconds = sum(float(item.get("wall_seconds", 0.0)) for item in prior_attempts)
    remaining_wall_limit = wall_limit - prior_wall_seconds
    if remaining_wall_limit <= 0.0:
        cumulative_peak_rss = max(
            (int(item.get("peak_process_tree_rss_bytes") or 0) for item in prior_attempts),
            default=0,
        )
        cumulative_rss_samples = sum(
            int(item.get("rss_sample_count", 0)) for item in prior_attempts
        )
        sample_period = float(
            prior_attempts[-1].get("rss_sample_period_seconds", 0.25) if prior_attempts else 0.25
        )
        log.touch(exist_ok=True)
        with log.open("a", encoding="utf-8") as stream:
            stream.write(
                "wall budget exhausted before launching resumed attempt; "
                f"prior={prior_wall_seconds:.6f}s limit={wall_limit:.6f}s\n"
            )
        record = {
            "command": command,
            "returncode": 124,
            "timed_out": True,
            "budget_exhausted": True,
            "normalized_command": normalized_command,
            "resume": resumed,
            "attempt_wall_seconds": 0.0,
            "cumulative_wall_seconds": prior_wall_seconds,
            "peak_process_tree_rss_bytes": cumulative_peak_rss or None,
            "rss_sample_count": cumulative_rss_samples,
            "rss_sample_period_seconds": sample_period,
            "wall_limit_seconds": wall_limit,
            "remaining_wall_seconds": 0.0,
            "wall_seconds": prior_wall_seconds,
            "attempts": prior_attempts,
            "console_log": str(log),
        }
        trial_path.write_text(json.dumps(record, indent=2) + "\n")
        return record
    started = time.monotonic()
    timed_out = False
    with log.open("a", encoding="utf-8") as stream:
        child = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        sampler = _ProcessTreeRSSSampler(child.pid)
        sampler.start()
        try:
            try:
                returncode = child.wait(timeout=remaining_wall_limit)
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                timed_out = isinstance(error, subprocess.TimeoutExpired)
                if os.name == "posix":
                    with suppress(ProcessLookupError):
                        os.killpg(child.pid, signal.SIGTERM)
                else:
                    child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        with suppress(ProcessLookupError):
                            os.killpg(child.pid, signal.SIGKILL)
                    else:
                        child.kill()
                    child.wait()
                returncode = 124 if timed_out else 130
        finally:
            sampler.stop()
    attempt = {
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_seconds": time.monotonic() - started,
        "peak_process_tree_rss_bytes": sampler.peak_bytes,
        "rss_sample_count": sampler.sample_count,
        "rss_sample_period_seconds": sampler.period,
        "wall_limit_seconds": wall_limit,
        "remaining_wall_seconds": remaining_wall_limit,
    }
    attempts = [*prior_attempts, attempt]
    cumulative_wall_seconds = sum(float(item.get("wall_seconds", 0.0)) for item in attempts)
    cumulative_peak_rss = max(
        (int(item.get("peak_process_tree_rss_bytes") or 0) for item in attempts), default=0
    )
    cumulative_rss_samples = sum(int(item.get("rss_sample_count", 0)) for item in attempts)
    record = {
        **attempt,
        "normalized_command": normalized_command,
        "resume": resumed,
        "attempt_wall_seconds": attempt["wall_seconds"],
        "cumulative_wall_seconds": cumulative_wall_seconds,
        "peak_process_tree_rss_bytes": cumulative_peak_rss or None,
        "rss_sample_count": cumulative_rss_samples,
        "rss_sample_period_seconds": sampler.period,
        "wall_seconds": cumulative_wall_seconds,
        "attempts": attempts,
        "console_log": str(log),
    }
    trial_path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def collect_cost(root: Path) -> dict:
    """Summarize saved solver measurements without loading whole histories.

    Late step samples are bounded to 256 records per journal. RSS is the
    reported aggregate over ranks; process startup and I/O are represented by
    the enclosing trial's wall time, not reconstructed from phase sums.
    """
    result = {
        "peak_rank_aggregate_rss_bytes": None,
        "peak_particles": 0,
        "journals": [],
        "accepted_coupling_intervals": 0,
        "unconverged_stationary_intervals": 0,
    }
    for path in sorted(root.rglob("performance.jsonl")):
        late = deque(maxlen=256)
        count = 0
        first_time = last_time = None
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                count += 1
                late.append(float(row["step_seconds"]["max"]))
                first_time = row["time"] if first_time is None else first_time
                last_time = row["time"]
                rss = row.get("memory", {}).get("aggregate_peak_rss_end_bytes")
                if rss is not None and float(rss) > 0.0:
                    result["peak_rank_aggregate_rss_bytes"] = max(
                        result["peak_rank_aggregate_rss_bytes"] or 0.0, float(rss)
                    )
        result["journals"].append(
            {
                "path": str(path),
                "steps": count,
                "first_time": first_time,
                "last_time": last_time,
                "late_step_median_seconds": float(np.median(late)) if late else None,
                "late_step_p90_seconds": float(np.quantile(late, 0.9)) if late else None,
            }
        )
    for path in root.rglob("coupler_diagnostics.jsonl"):
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                # Retain the raw final diagnostic so every conservation and
                # interface gate remains inspectable alongside runtime.
                result["last_coupler_diagnostic"] = row
                result["accepted_coupling_intervals"] += 1
                if (
                    float(row.get("time", 0)) >= 40
                    and row.get("interface_iteration", {}).get("converged") is False
                ):
                    result["unconverged_stationary_intervals"] += 1
                result["peak_particles"] = max(
                    result["peak_particles"], int(row.get("n_transfer_particles", 0))
                )
    return result


__all__ = ["collect_cost", "run_trial", "run_coupled_cylinder"]
