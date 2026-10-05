"""Reference-cylinder startup forcing on the native FVM accepted clock."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .initial_conditions import initialize_cylinder_perturbation, startup_velocity


def _schedule(solver, span, duration, startup, steady, transition_duration, perturbation):
    setup = getattr(solver, "_resolved_setup", solver.setup)
    clock = setup.time
    dt = float(clock.time_step_size)
    span, duration = float(span), float(duration)
    transition_duration = float(transition_duration)
    startup = np.asarray(startup, dtype=np.float64)
    steady = np.asarray(steady, dtype=np.float64)
    if not np.isfinite(span) or span <= 0:
        raise ValueError("span must be finite and positive")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("startup_duration must be finite and positive")
    if not np.isfinite(transition_duration) or not 0 < transition_duration <= duration:
        raise ValueError(
            "startup_transition_duration must lie greater than zero and at most startup_duration"
        )
    if clock.start_time != 0 or clock.adjustment is not None:
        raise ValueError("Reference startup requires fixed FVM steps starting at time zero")
    switch = round(duration / dt)
    if switch < 1 or not np.isclose(switch * dt, duration, atol=1e-12, rtol=0):
        raise ValueError("startup_duration must end on an accepted FVM step")
    for name, vector in (("startup", startup), ("steady", steady)):
        if vector.shape != (3,) or not np.all(np.isfinite(vector)) or vector[2] != 0:
            raise ValueError(f"{name} freestream must be a finite in-plane three-component vector")
    if steady[1] != 0:
        raise ValueError("Steady transverse speed must be zero to recover lateral slip")
    patches = {boundary.name: boundary for boundary in setup.boundaries}
    if "inlet" not in patches or patches["inlet"].velocity_type != "fixedValue":
        raise ValueError("Reference startup requires a fixedValue inlet")
    if not np.array_equal(patches["inlet"].velocity_value, startup):
        raise ValueError(
            "Configure the inlet with the startup freestream for native restart identity"
        )
    if not np.array_equal(setup.initial_velocity, startup):
        raise ValueError("Configure initial_velocity with the startup freestream")
    for name in ("ymin", "ymax"):
        if (
            name not in patches
            or patches[name].velocity_type != "slip"
            or patches[name].pressure_type != "zeroGradient"
        ):
            raise ValueError(
                "Reference startup requires configured lateral slip/zeroGradient boundaries"
            )
    schedule = {
        "schema": "openonda-reference-cylinder-startup/2",
        "startup_duration": duration,
        "switch_step": switch,
        "startup_freestream_velocity": startup.tolist(),
        "steady_freestream_velocity": steady.tolist(),
        "fvm_time_step": dt,
        "span": span,
        "inlet": "fixedValue",
        "lateral_velocity": "startup prescribed normal background/zero tangential gradient; steady slip",
        "lateral_pressure": "zeroGradient",
        "initial_disturbance": f"compact divergence-free 3D curl; amplitude {perturbation['amplitude']:g}",
    }
    schedule.update(
        startup_transition_duration=transition_duration,
        startup_transition_profile="quintic-C2",
        boundary_time_evaluation="accepted endpoint",
    )
    return schedule


def run_reference_cylinder(
    solver,
    *,
    span: float,
    start_from="latest",
    startup_duration: float,
    startup_transition_duration: float,
    startup_freestream_velocity,
    steady_freestream_velocity,
    perturbation,
) -> None:
    """Apply the cylinder trigger while retaining the native FVM run lifecycle.

    Build the solver with the startup inlet/initial velocity, lateral slip
    with zero-gradient pressure, and its ordinary wall/spanwise/outlet
    conditions. The lateral runtime trace admits the imposed crossflow during
    startup. At the switch the original slip condition is restored, including
    its diffusion assembly and shared momentum-matrix eligibility.
    Viscosity and force reference speed remain the factory's nominal values.

    Native start selection happens before initial seeding. Boundary traces are
    reconstructed from the restored accepted step. A positive transition
    duration applies a quintic C2 taper over the final part of startup; each
    implicit FVM step uses its accepted endpoint velocity. Once steady, the
    boundary is left unchanged. Every evolution step, output, backup and
    finalization still runs through ``solver.run``/``advance``.

    ``solution/reference_startup.json`` identifies the immutable forcing
    policy; copy it beside an explicitly relocated native backup. End time is
    excluded so native continuation may extend the physical horizon.
    """
    from source.restart import select_backup
    from source.simulation.parallel import collective_phase

    schedule = _schedule(
        solver,
        span,
        startup_duration,
        startup_freestream_velocity,
        steady_freestream_velocity,
        startup_transition_duration,
        perturbation,
    )
    setup = getattr(solver, "_resolved_setup", solver.setup)
    solution = Path(solver.solution_dir)
    metadata = solution / "reference_startup.json"
    comm = solver.parallel.comm
    root = solver.parallel.is_root
    in_memory_resume = start_from is None and (
        bool(getattr(solver, "_restart_loaded", False)) or solver.step > 0
    )
    with collective_phase(comm, "reference startup schedule admission"):
        if root:
            selected = select_backup(
                start_from, directory=solution, kind="fvm", backup_path=setup.backup.path
            )
            source = metadata if selected is None else selected.parent / metadata.name
            if (selected is not None or in_memory_resume) and not source.is_file():
                raise ValueError(
                    "Reference resume requires its reference_startup.json; "
                    "use './allrun.sh --fresh' to archive old outputs and start this policy from zero"
                )
            admitted = schedule
            for candidate in dict.fromkeys((source, metadata)):
                if not candidate.exists():
                    continue
                try:
                    recorded = json.loads(candidate.read_text())
                except (OSError, json.JSONDecodeError) as error:
                    raise ValueError(
                        "Cannot read reference_startup.json; use './allrun.sh --fresh' "
                        "to archive old outputs and start this policy from zero"
                    ) from error
                if recorded != admitted:
                    raise ValueError(
                        "reference_startup.json describes a different forcing policy; "
                        "use './allrun.sh --fresh' to archive old outputs and start this policy from zero"
                    )
            schedule = admitted
    if comm is not None:
        schedule = comm.bcast(schedule if root else None, root=0)

    restored = (
        solver.start_from(start_from)
        if start_from is not None
        else (bool(getattr(solver, "_restart_loaded", False)) or solver.step > 0)
    )
    with collective_phase(comm, "reference startup accepted clock"):
        if not np.isclose(solver.time, solver.step * schedule["fvm_time_step"], atol=1e-10, rtol=0):
            raise ValueError("Reference startup requires a consistent accepted FVM step/time")

    # All ranks enter each public getter/setter. Getters return the globally
    # ordered patch on root and empty arrays on workers; setters scatter it.
    normals = {name: solver.get_boundary_face_normal(name) for name in ("inlet", "ymin", "ymax")}
    previous_background = None

    def update_boundary(*, endpoint=False):
        nonlocal previous_background
        evaluation_step = solver.step + int(endpoint)
        background = np.asarray(
            startup_velocity(
                evaluation_step * schedule["fvm_time_step"],
                schedule["startup_duration"],
                schedule["startup_transition_duration"],
                schedule["startup_freestream_velocity"],
                schedule["steady_freestream_velocity"],
            ),
            dtype=np.float64,
        )
        if previous_background is not None and np.array_equal(previous_background, background):
            return
        inlet_values = np.tile(background, (len(normals["inlet"]), 1))
        solver.set_dirichlet_velocity_boundary_condition_vec(inlet_values, "inlet")
        for name in ("ymin", "ymax"):
            solver.set_normal_velocity_tangential_gradient_boundary_condition(
                normals[name] @ background, np.zeros_like(normals[name]), name
            )
        if evaluation_step >= schedule["switch_step"]:
            # The public setter above installs the zero-normal slip trace and
            # invalidates derived fields on every owning rank. Restore the
            # native slip strategy too: zero-normal mixed diffusion otherwise
            # retains a normal-component penalty absent from the original
            # slip condition, and disables the shared momentum-matrix path.
            for boundary in solver.boundaries:
                if boundary["name"] in ("ymin", "ymax"):
                    boundary["velocity_type"] = "slip"
                    for key in (
                        "normal_velocity_field",
                        "tangential_gradient_field",
                        "max_removed_tangential_gradient_normal_component",
                    ):
                        boundary.pop(key, None)
        previous_background = background.copy()

    update_boundary()
    if not restored:
        initialize_cylinder_perturbation(
            solver,
            span,
            freestream_velocity=startup_freestream_velocity,
            perturbation=perturbation,
        )
    with collective_phase(comm, "reference startup schedule publication"):
        if root and not metadata.exists():
            temporary = metadata.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(schedule, sort_keys=True, indent=2) + "\n")
            temporary.replace(metadata)
    # Calling start_from separately lets us restore the runtime trace before
    # initial output. Preserve run(start_from=...)'s initial native checkpoint.
    if not restored and start_from is not None:
        backup = Path(setup.backup.path)
        solver.save_state(backup if backup.is_absolute() else solution / backup)

    missing = object()
    previous_override = solver.__dict__.get("advance", missing)
    original_advance = solver.advance

    def advance():
        update_boundary(endpoint=True)
        return original_advance()

    solver.advance = advance
    try:
        solver.run()
    finally:
        if previous_override is missing:
            del solver.advance
        else:
            solver.advance = previous_override


__all__ = ["run_reference_cylinder"]
