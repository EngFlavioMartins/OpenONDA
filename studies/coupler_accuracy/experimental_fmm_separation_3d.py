"""Change only FMM geometric admissibility in the matched cube trajectory."""

from __future__ import annotations

from contextlib import contextmanager
import json

import numpy as np

from source.solvers.vpm.core.evolution import EvolutionStepper
from source.solvers.vpm.core.solver import VPMSolver
from source.solvers.vpm.physics.induction.fmm import device as fmm_device
from source.solvers.vpm.physics.stage_rhs import StageRHS
from source.solvers.vpm.stabilization.manager import StabilizationManager


@contextmanager
def fmm_separation(separation, record, report_path):
    """Set the coefficient before the fresh solver compiles any FMM kernels."""
    if separation not in (3., 4.5, 6.):
        raise ValueError("Require one of the independently checked FMM factors")
    original_factor = fmm_device._GEOMETRIC_SEPARATION_FACTOR
    assert original_factor == 3.
    original_outer, original_rhs = VPMSolver.advance, StageRHS.evaluate
    original_diffusion, original_phase = EvolutionStepper._apply_viscous_diffusion, StabilizationManager.run_phase
    active = {}

    def outer(solver, *, defer_output=False):
        if active:
            raise RuntimeError("Nested VPM advance is outside this study")
        if (solver.flow_model != "DNS" or solver.viscous_scheme != "GBD"
                or solver.precision != "f32" or solver.integrator.name != "RK2"
                or solver.axisymmetric_axis >= 0 or solver.vlm_solver is not None
                or solver.panel_solver is None or solver.n_sources or not defer_output
                or solver.panel_solver.coupling_scope != "vpm_boundary_condition"
                or solver.time_step_size != .05):
            raise ValueError("Require the original deferred 3D laminar RK2/GBD cube")
        assert isinstance(solver.stage_rhs.induction, fmm_device.FMMInduction)
        assert solver.stage_rhs.induction.stretching_scheme == "TRANSPOSED"
        assert all(getattr(solver.physics, name, None) is None for name in (
            "body_velocity", "body_velocity_field", "body_velocity_gradient", "body_velocity_gradient_field",
            "velocity_override", "velocity_override_gradient"))
        row = {"start_time": float(solver.time), "start_step": int(solver.step), "outer_dt": .05,
               "geometric_separation_factor": separation, "stages": [], "diffusion_calls": [], "stabilization_phases": []}
        active.update(solver=solver, row=row, in_stage=False)
        try:
            answer = original_outer(solver, defer_output=defer_output)
            assert (solver.time, solver.step) == (round(row["start_time"] + .05, 12), row["start_step"] + 1)
            assert [stage["index"] for stage in row["stages"]] == [0, 1]
            np.testing.assert_array_equal([stage["time"] for stage in row["stages"]],
                                          [row["start_time"], row["start_time"] + .05])
            assert row["diffusion_calls"] == [.05]
            assert row["stabilization_phases"] == ["pre_evolution", "pre_strength", "post_evolution", "post_step"]
            row.update(accepted_time=float(solver.time), accepted_step=int(solver.step), schedule_checks_passed=True)
            record["outer_advances"].append(row)
            report_path.write_text(json.dumps(record, indent=2) + "\n")
            return answer
        finally:
            active.clear()

    def rhs(evaluator, state, stage_time, rates):
        if not active:
            return original_rhs(evaluator, state, stage_time, rates)
        solver, row = active["solver"], active["row"]
        assert evaluator is solver.stage_rhs and not active["in_stage"] and state.time == stage_time
        assert separation == fmm_device._GEOMETRIC_SEPARATION_FACTOR
        clock = (solver.time, solver.step, solver.time_step_size, solver.particles.step)
        induction = evaluator.induction
        before = induction.diagnostics.stage_evaluations
        active["in_stage"] = True
        try:
            answer = original_rhs(evaluator, state, stage_time, rates)
            assert induction.diagnostics.stage_evaluations == before + 1
            assert (solver.time, solver.step, solver.time_step_size, solver.particles.step) == clock
            work = induction.workspace
            row["stages"].append({"time": float(stage_time), "index": int(state.stage_index), "count": int(state.count),
                                   "m2l_pairs": int(work._m2l_count[None]), "near_pairs": int(work._near_count[None]),
                                   "p2p_interactions_excluding_self": int(work._p2p_particle_count[None]),
                                   "native_stage_evaluations": 1, "outer_clocks_unchanged": True})
            return answer
        finally:
            active["in_stage"] = False

    def diffusion(stepper, dt):
        if active:
            assert stepper.solver is active["solver"] and not active["in_stage"]
            active["row"]["diffusion_calls"].append(float(dt))
        return original_diffusion(stepper, dt)

    def phase(manager, name, profiler=None):
        if active:
            assert manager is active["solver"].stabilization and not active["in_stage"]
            active["row"]["stabilization_phases"].append(name)
        return original_phase(manager, name, profiler=profiler)

    fmm_device._GEOMETRIC_SEPARATION_FACTOR = float(separation)
    VPMSolver.advance, StageRHS.evaluate = outer, rhs
    EvolutionStepper._apply_viscous_diffusion, StabilizationManager.run_phase = diffusion, phase
    try:
        yield
    finally:
        fmm_device._GEOMETRIC_SEPARATION_FACTOR = original_factor
        VPMSolver.advance, StageRHS.evaluate = original_outer, original_rhs
        EvolutionStepper._apply_viscous_diffusion, StabilizationManager.run_phase = original_diffusion, original_phase
