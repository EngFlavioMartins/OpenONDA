"""Subcycle native RK2/GBD splitting inside one accepted hybrid interval."""

from __future__ import annotations

from contextlib import contextmanager
import json
import operator

import numpy as np

from source.coupler.solver import _validate_gbd_moment_recovery
from source.solvers.vpm.core.evolution import EvolutionStepper
from source.solvers.vpm.core.solver import VPMSolver
from source.solvers.vpm.numerics.runge_kutta import RungeKutta
from source.solvers.vpm.physics.stage_rhs import StageRHS
from source.solvers.vpm.stabilization.manager import StabilizationManager


@contextmanager
def split_subcycling(substeps, correction_limit, record, report_path):
    """Preserve the outer clock/stabilization while repeating the split pair."""
    substeps = operator.index(substeps)
    if substeps < 1:
        raise ValueError("Require positive VPM substeps")
    original_outer, original_pair = VPMSolver.advance, EvolutionStepper._apply_coupled_update
    original_rk, original_rhs = RungeKutta.advance, StageRHS.evaluate
    original_diffusion, original_phase = EvolutionStepper._apply_viscous_diffusion, StabilizationManager.run_phase
    active = {}

    def outer(solver, *, defer_output=False):
        if active:
            raise RuntimeError("Nested VPM advance is outside this experiment")
        if (solver.flow_model != "DNS" or solver.viscous_scheme != "GBD"
                or solver.precision != "f32" or solver.integrator.name != "RK2"
                or solver.axisymmetric_axis >= 0 or solver.vlm_solver is not None
                or solver.panel_solver is None or solver.n_sources
                or solver.panel_solver.coupling_scope != "vpm_boundary_condition"
                or solver.time_step_size != .05 or not defer_output):
            raise ValueError("Require the deferred 3D laminar RK2/GBD cube at outer dt=0.05")
        assert all(getattr(solver.physics, name, None) is None for name in (
            "body_velocity", "body_velocity_field", "body_velocity_gradient", "body_velocity_gradient_field",
            "velocity_override", "velocity_override_gradient"))
        start = (float(solver.time), int(solver.step))
        row = {"start_time": start[0], "start_step": start[1], "outer_dt": .05,
               "vpm_substeps": substeps, "substep_dt": .05 / substeps,
               "particles_at_entry": len(solver.particles), "rk_calls": [], "stages": [],
               "diffusion_calls": [], "stabilization_phases": [], "substep_projection_corrections": []}
        active.update(solver=solver, row=row, substep=None, in_rk=False)
        try:
            answer = original_outer(solver, defer_output=defer_output)
            assert (solver.time, solver.step) == (round(start[0] + .05, 12), start[1] + 1)
            assert len(row["rk_calls"]) == substeps and len(row["stages"]) == 2 * substeps
            times = [call["time"] + c * call["dt"] for call in row["rk_calls"] for c in solver.integrator.tableau.c]
            np.testing.assert_array_equal([stage["time"] for stage in row["stages"]], times)
            assert [stage["index"] for stage in row["stages"]] == [0, 1] * substeps
            assert [call["dt"] for call in row["diffusion_calls"] if call["dt"] > 0] == [.05 / substeps] * substeps
            assert row["stabilization_phases"] == ["pre_evolution", "pre_strength", "post_evolution", "post_step"]
            row.update(accepted_time=float(solver.time), accepted_step=int(solver.step),
                       particles_at_exit=len(solver.particles), schedule_checks_passed=True)
            record["outer_advances"].append(row)
            report_path.write_text(json.dumps(record, indent=2) + "\n")
            return answer
        finally:
            active.clear()

    def pair(stepper, dt):
        if not active:
            return original_pair(stepper, dt)
        solver, row = active["solver"], active["row"]
        assert stepper.solver is solver and dt == .05 and active["substep"] is None
        assert not row["rk_calls"]
        clock = (solver.time, solver.step, solver.time_step_size)
        try:
            for index in range(substeps):
                active["substep"] = index
                original_pair(stepper, dt / substeps)
                value = float(solver.physics.rate_projection_max_correction_ratio)
                assert np.isfinite(value) and value >= 0
                row["substep_projection_corrections"].append(value)
                assert (solver.time, solver.step, solver.time_step_size) == clock
        finally:
            active["substep"] = None
        # Each native pair resets this diagnostic. Retain the complete interval's maximum.
        solver.physics.rate_projection_max_correction_ratio = max(row["substep_projection_corrections"])
        row["interval_projection_maximum"] = float(solver.physics.rate_projection_max_correction_ratio)
        row["split_preserved_outer_clock"] = True

    def rk(integrator, **arguments):
        if not active:
            return original_rk(integrator, **arguments)
        solver, row, index = active["solver"], active["row"], active["substep"]
        assert index is not None and integrator is solver.integrator
        assert arguments["right_hand_side"] is solver.stage_rhs and arguments["time"] == solver.time
        assert arguments["time_step_size"] == .05 / substeps and len(row["rk_calls"]) == index
        start = float(solver.time) + index * (.05 / substeps)
        call = {"time": start, "dt": float(arguments["time_step_size"]), "count": int(arguments["count"])}
        row["rk_calls"].append(call)
        active["in_rk"] = True
        try:
            return original_rk(integrator, **{**arguments, "time": start})
        finally:
            active["in_rk"] = False

    def rhs(evaluator, state, stage_time, rates):
        if active and active["in_rk"]:
            assert evaluator is active["solver"].stage_rhs and state.time == stage_time
            active["row"]["stages"].append({"time": float(stage_time), "index": int(state.stage_index), "count": int(state.count)})
        return original_rhs(evaluator, state, stage_time, rates)

    def diffusion(stepper, dt):
        if not active:
            return original_diffusion(stepper, dt)
        solver, row = active["solver"], active["row"]
        assert stepper.solver is solver and not active["in_rk"]
        assert dt == 0 or (active["substep"] is not None and dt == .05 / substeps)
        count = len(solver.particles)
        answer = original_diffusion(stepper, dt)
        recovery = solver.physics.last_gbd_moment_recovery
        _validate_gbd_moment_recovery(recovery, correction_limit)
        row["diffusion_calls"].append({"dt": float(dt), "substep": active["substep"],
                                       "accepted_step_before": int(solver.step),
                                       "particles_before": count, "particles_after": len(solver.particles),
                                       "laplacian_substeps": int(solver.physics.last_gbd_diffusion_substeps),
                                       "moment_recovery": recovery, "recovery_checked": True})
        return answer

    def phase(manager, name, profiler=None):
        if active:
            assert manager is active["solver"].stabilization and not active["in_rk"]
            assert active["substep"] is None
            active["row"]["stabilization_phases"].append(name)
        return original_phase(manager, name, profiler=profiler)

    VPMSolver.advance, EvolutionStepper._apply_coupled_update = outer, pair
    RungeKutta.advance, StageRHS.evaluate = rk, rhs
    EvolutionStepper._apply_viscous_diffusion, StabilizationManager.run_phase = diffusion, phase
    try:
        yield
    finally:
        VPMSolver.advance, EvolutionStepper._apply_coupled_update = original_outer, original_pair
        RungeKutta.advance, StageRHS.evaluate = original_rk, original_rhs
        EvolutionStepper._apply_viscous_diffusion, StabilizationManager.run_phase = original_diffusion, original_phase
