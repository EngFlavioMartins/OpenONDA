"""Subcycle only particle RK evolution, preserving the outer hybrid schedule."""

from __future__ import annotations

from contextlib import contextmanager
import json
import operator

import numpy as np

from source.solvers.vpm.core.evolution import EvolutionStepper
from source.solvers.vpm.core.solver import VPMSolver
from source.solvers.vpm.numerics.runge_kutta import RungeKutta
from source.solvers.vpm.physics.stage_rhs import StageRHS
from source.solvers.vpm.stabilization.manager import StabilizationManager


def advance_subcycles(advance, integrator, substeps, arguments):
    """Compose native RK steps with their actual physical start times.

    This operates on position and vector circulation only. It does not call
    the VPM facade, mutate its clock, diffuse, regenerate or renew particles.
    """
    substeps = operator.index(substeps)
    if substeps < 1:
        raise ValueError("Require at least one inviscid substep")
    if substeps == 1:
        return advance(integrator, **arguments)
    start = float(arguments["time"])
    dt = float(arguments["time_step_size"]) / substeps
    for index in range(substeps):
        advance(integrator, **{**arguments, "time": start + index * dt,
                               "time_step_size": dt})


@contextmanager
def inviscid_subcycling(substeps, record, report_path):
    """Scope and audit the single-solver laminar 3D RK2/GBD experiment."""
    if operator.index(substeps) < 1:
        raise ValueError("Require positive inviscid substeps")
    original_outer = VPMSolver.advance
    original_rk = RungeKutta.advance
    original_rhs = StageRHS.evaluate
    original_diffusion = EvolutionStepper._apply_viscous_diffusion
    original_phase = StabilizationManager.run_phase
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
        start = (float(solver.time), int(solver.step))
        row = {"start_time": start[0], "start_step": start[1],
               "outer_dt": float(solver.time_step_size), "inviscid_substeps": substeps,
               "inviscid_dt": float(solver.time_step_size) / substeps,
               "particles_at_entry": len(solver.particles), "rk_calls": [],
               "stages": [], "diffusion_calls": [], "stabilization_phases": []}
        active.update(solver=solver, row=row, in_rk=False)
        try:
            answer = original_outer(solver, defer_output=defer_output)
            assert (solver.time, solver.step) == (round(start[0] + .05, 12), start[1] + 1)
            assert len(row["rk_calls"]) == substeps
            assert len(row["stages"]) == 2 * substeps
            expected_times = [call["time"] + c * call["dt"]
                              for call in row["rk_calls"] for c in solver.integrator.tableau.c]
            np.testing.assert_array_equal([stage["time"] for stage in row["stages"]], expected_times)
            assert [stage["index"] for stage in row["stages"]] == [0, 1] * substeps
            positive = [call["dt"] for call in row["diffusion_calls"] if call["dt"] > 0]
            assert positive == [.05]
            assert all(call["dt"] in (0., .05) for call in row["diffusion_calls"])
            assert row["stabilization_phases"] == ["pre_evolution", "pre_strength", "post_evolution", "post_step"]
            row.update(accepted_time=float(solver.time), accepted_step=int(solver.step),
                       particles_at_exit=len(solver.particles), schedule_checks_passed=True)
            record["outer_advances"].append(row)
            report_path.write_text(json.dumps(record, indent=2) + "\n")
            return answer
        finally:
            active.clear()

    def rk(integrator, **arguments):
        if not active:
            return original_rk(integrator, **arguments)
        solver, row = active["solver"], active["row"]
        assert integrator is solver.integrator and arguments["right_hand_side"] is solver.stage_rhs
        assert arguments["time"] == solver.time and arguments["time_step_size"] == .05
        assert not active["in_rk"] and not row["rk_calls"]
        clock = (solver.time, solver.step, solver.time_step_size, len(solver.particles))

        def native(inner_integrator, **inner):
            row["rk_calls"].append({"time": float(inner["time"]),
                                     "dt": float(inner["time_step_size"]), "count": int(inner["count"])})
            return original_rk(inner_integrator, **inner)

        active["in_rk"] = True
        try:
            answer = advance_subcycles(native, integrator, substeps, arguments)
        finally:
            active["in_rk"] = False
        assert (solver.time, solver.step, solver.time_step_size, len(solver.particles)) == clock
        row["rk_preserved_outer_clock_and_population"] = True
        return answer

    def rhs(evaluator, state, stage_time, rates):
        if active and active["in_rk"]:
            assert evaluator is active["solver"].stage_rhs and state.time == stage_time
            active["row"]["stages"].append({"time": float(stage_time),
                                              "index": int(state.stage_index), "count": int(state.count)})
        return original_rhs(evaluator, state, stage_time, rates)

    def diffusion(stepper, dt):
        if active:
            assert stepper.solver is active["solver"] and not active["in_rk"]
            active["row"]["diffusion_calls"].append({"dt": float(dt),
                                                       "accepted_step_before": int(stepper.solver.step)})
        return original_diffusion(stepper, dt)

    def phase(manager, name, profiler=None):
        if active:
            assert manager is active["solver"].stabilization and not active["in_rk"]
            active["row"]["stabilization_phases"].append(name)
        return original_phase(manager, name, profiler=profiler)

    VPMSolver.advance, RungeKutta.advance, StageRHS.evaluate = outer, rk, rhs
    EvolutionStepper._apply_viscous_diffusion = diffusion
    StabilizationManager.run_phase = phase
    try:
        yield
    finally:
        VPMSolver.advance, RungeKutta.advance, StageRHS.evaluate = original_outer, original_rk, original_rhs
        EvolutionStepper._apply_viscous_diffusion = original_diffusion
        StabilizationManager.run_phase = original_phase
