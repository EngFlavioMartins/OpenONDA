"""Local accepted-clock startup and continuation for this tutorial."""

from dataclasses import replace
import json
from numbers import Integral

import numpy as np


def startup_velocity(time, duration, transition_duration, startup, steady) -> tuple[float, ...]:
    """Hold the startup flow, then remove it with a C2 quintic taper.

    The transition occupies the final ``transition_duration`` seconds of
    startup. Its first and second time derivatives vanish at both ends, so
    removing the crossflow does not introduce an impulsive acceleration.
    Caller-side schedule validation keeps this hot-path function small.
    """
    if time >= duration:
        return tuple(float(value) for value in steady)
    if time <= duration - transition_duration:
        return tuple(float(value) for value in startup)
    fraction = (float(time) - (duration - transition_duration)) / transition_duration
    blend = fraction**3 * (10.0 + fraction * (-15.0 + 6.0 * fraction))
    return tuple(
        float(first + blend * (last - first)) for first, last in zip(startup, steady, strict=True)
    )


def run_coupled_cylinder(
    build_case,
    *,
    start_from="latest",
    output_root=None,
    end_time=None,
    restart_from=None,
    max_coupling_steps=None,
    overrides=None,
    perturbation,
    startup_duration: float,
    startup_transition_duration: float,
    steady_freestream_velocity: tuple[float, float, float],
) -> int:
    """Execute a local physical factory with native continuation and owned output.

    The startup schedule returns the factory's startup freestream
    to ``steady_freestream_velocity`` by ``startup_duration``. A positive
    ``startup_transition_duration`` removes the trigger with a C2 quintic
    taper during the final part of startup. Both durations must end on
    exchange boundaries. Always rebuild
    the factory with the startup background, including on resume: its transfer lattice
    must retain the same buffer. The startup speed must cover the steady
    speed. Force normalization and viscosity remain the factory's choices.

    The schedule is recorded in ``solution/cylinder_startup.json``
    and checked on resume, alongside strict native checkpoint admission.
    A checkpoint copied elsewhere needs this sidecar in its parent folder.
    ``max_coupling_steps`` caps the total accepted steps across all stages.
    """
    from openonda.coupler import create_coupler

    if startup_duration is None or steady_freestream_velocity is None:
        raise ValueError(
            "startup_duration and steady_freestream_velocity must be supplied together"
        )
    fvm_setup, vpm_case, coupler_setup, mesh = build_case(end_time=end_time, overrides=overrides)
    schedule = _cylinder_startup_schedule(
        fvm_setup,
        vpm_case,
        coupler_setup,
        startup_duration,
        steady_freestream_velocity,
        startup_transition_duration,
        perturbation,
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
        return _run_cylinder_startup(
            solver, vpm_case, schedule, start_from, restart_from, max_coupling_steps, perturbation
        )


def _cylinder_startup_schedule(
    flow, particles, policy, duration, steady, transition_duration, perturbation
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
        or transition_duration <= 0
        or transition_duration > duration
    ):
        raise ValueError(
            "startup_transition_duration must be finite and lie in (0, startup_duration]"
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
        "schema": "openonda-cylinder-startup/2",
        "startup_duration": duration,
        "switch_step": switch_step,
        "startup_freestream_velocity": startup.tolist(),
        "steady_freestream_velocity": steady.tolist(),
        "exchange_time_step": exchange_dt,
        "fvm_time_step": fvm_dt,
        "end_time": end_time,
        "initial_disturbance": f"compact divergence-free 3D curl; amplitude {perturbation['amplitude']:g}",
    }
    schedule.update(
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


def _admit_cylinder_startup_schedule(recorded, requested):
    """Require the exact current forcing policy for continuation."""
    if recorded != requested:
        raise ValueError("Staged resume requires an identical cylinder_startup.json schedule")
    return recorded


def _startup_velocity_at_step(schedule, step):
    return startup_velocity(
        step * schedule["exchange_time_step"],
        schedule["startup_duration"],
        schedule["startup_transition_duration"],
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

    def advance(step, time_end):
        nonlocal previous_velocity
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


def _run_cylinder_startup(
    solver, particles, schedule, start_from, restart_from, limit, perturbation
) -> int:
    """Run accepted-clock segments, admitting each saved stage before switching."""
    from source.coupler.parallel import collective_phase
    from source.restart import select_backup

    metadata = solver.solution_dir / "cylinder_startup.json"
    selected = saved_velocity = None
    startup = schedule["startup_freestream_velocity"]
    with collective_phase(solver._comm, "cylinder startup restart selection"):
        if solver._is_master:
            selector = restart_from if restart_from is not None else start_from
            selected = select_backup(selector, directory=solver.solution_dir, kind="coupled")
            if selected is not None:
                identity = selected.parent / metadata.name
                if not identity.is_file():
                    raise ValueError("Staged resume requires an identical cylinder_startup.json")
                schedule = _admit_cylinder_startup_schedule(
                    json.loads(identity.read_text()), schedule
                )
                saved = json.loads((selected / "manifest.json").read_text())
                step = saved["coupling_step"]
                if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                    raise ValueError("Invalid startup checkpoint coupling_step")
                saved_velocity = saved["config"]["coupler"]["freestream_velocity"]
                permitted = [list(_startup_velocity_at_step(schedule, step))]
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
        with collective_phase(solver._comm, "cylinder restored startup background"):
            _set_cylinder_freestream(solver, _startup_velocity_at_step(schedule, step))
    else:
        induction = particles.numerics.induction
        initialize_cylinder_perturbation(
            solver.fvm_solver,
            induction.z_max - induction.z_min,
            freestream_velocity=startup,
            perturbation=perturbation,
        )

    with collective_phase(solver._comm, "cylinder startup schedule record"):
        if solver._is_master and (not metadata.exists() or start_from == "initial"):
            temporary = metadata.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(schedule, indent=2, sort_keys=True) + "\n")
            temporary.replace(metadata)
    end_step = round(schedule["end_time"] / schedule["exchange_time_step"])
    if step >= end_step:
        return step
    return _run_smooth_cylinder_startup(solver, schedule, selected, step, limit)


def cylinder_initial_velocity(
    positions: np.ndarray,
    span: float,
    *,
    amplitude: float,
    radius: float,
    centre: float,
    freestream_velocity,
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
    radius_squared = radius**2
    support = np.maximum(1 - ((x - centre) ** 2 + y**2) / radius_squared, 0)
    wave_number = np.pi / span
    phase = wave_number * (z + 0.5 * span)
    velocity = np.tile(np.asarray(freestream_velocity, dtype=np.float64), (len(position), 1))
    velocity[:, 0] += (
        -amplitude * wave_number * support**4 * np.cos(phase)
        - 8 * amplitude * y / radius_squared * support**3
    )
    velocity[:, 1] += 8 * amplitude * (x - centre) / radius_squared * support**3
    velocity[:, 2] += -8 * amplitude * (x - centre) / radius_squared * support**3 * np.sin(phase)
    return velocity


def initialize_cylinder_perturbation(
    solver, span: float, *, perturbation, freestream_velocity
) -> None:
    """Install the same small 3D initial disturbance on each local FVM mesh."""
    count = solver.mesh_data["n_cells"]
    velocity = cylinder_initial_velocity(
        solver.geo_data["cell_centre"][:count],
        span,
        **perturbation,
        freestream_velocity=freestream_velocity,
    )
    solver.set_initial_velocity(velocity)
