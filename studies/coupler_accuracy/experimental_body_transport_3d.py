"""Add the accepted source-panel field only to particle RK stage rates."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json

import numpy as np

from source.solvers.vpm.core.evolution import EvolutionStepper
from source.solvers.vpm.core.solver import VPMSolver
from source.solvers.vpm.physics.stage_rhs import StageRHS
from source.solvers.vpm.stabilization.manager import StabilizationManager
from studies.coupler_accuracy.analytical_panel_gradient_3d import source_panel_gradient


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


@contextmanager
def body_transport(enabled, record, report_path):
    """Keep panel updates/shedding and exchange frequency fixed in this trial."""
    original_outer, original_rhs = VPMSolver.advance, StageRHS.evaluate
    original_diffusion, original_phase = EvolutionStepper._apply_viscous_diffusion, StabilizationManager.run_phase
    active = {}
    names = ("body_velocity", "body_velocity_field", "body_velocity_gradient", "body_velocity_gradient_field",
             "velocity_override", "velocity_override_gradient")

    def outer(solver, *, defer_output=False):
        if active:
            raise RuntimeError("Nested VPM advance is outside this experiment")
        panel = solver.panel_solver
        if (solver.flow_model != "DNS" or solver.viscous_scheme != "GBD"
                or solver.precision != "f32" or solver.integrator.name != "RK2"
                or solver.axisymmetric_axis >= 0 or solver.vlm_solver is not None
                or panel is None or solver.n_sources or not defer_output
                or panel.coupling_scope != "vpm_boundary_condition"
                or panel.boundary_condition_type != "NEUMANN" or solver.time_step_size != .05):
            raise ValueError("Require the deferred fully 3D laminar RK2/GBD source-panel cube")
        assert all(getattr(solver.physics, name, None) is None for name in names)
        assert panel.lattice.n_panels == 108 and panel.lattice.n_panels < panel.far_field_min_panels
        vertices = panel.lattice.vertex_position.to_numpy()[:108].astype(float)
        strength = panel.lattice.source_strength.to_numpy()[:108].astype(float)
        panel_hash = digest(strength)
        row = {"start_time": float(solver.time), "start_step": int(solver.step), "outer_dt": .05,
               "enabled": bool(enabled), "panels": 108, "panel_strength_sha256": panel_hash,
               "stages": [], "diffusion_calls": [], "stabilization_phases": []}
        active.update(solver=solver, row=row, vertices=vertices, strength=strength, in_stage=False)
        try:
            answer = original_outer(solver, defer_output=defer_output)
            assert (solver.time, solver.step) == (round(row["start_time"] + .05, 12), row["start_step"] + 1)
            assert [stage["index"] for stage in row["stages"]] == [0, 1]
            np.testing.assert_array_equal([stage["time"] for stage in row["stages"]],
                                          [row["start_time"], row["start_time"] + .05])
            assert row["diffusion_calls"] == [.05]
            assert row["stabilization_phases"] == ["pre_evolution", "pre_strength", "post_evolution", "post_step"]
            assert digest(panel.lattice.source_strength.to_numpy()[:108].astype(float)) == panel_hash
            assert all(getattr(solver.physics, name, None) is None for name in names)
            row.update(accepted_time=float(solver.time), accepted_step=int(solver.step),
                       panel_strength_unchanged_during_vpm_advance=True, stage_hooks_restored=True,
                       schedule_checks_passed=True)
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
        assert all(getattr(solver.physics, name, None) is None for name in names)
        position = solver.physics._download_vector_field(state.position, state.count)
        q = np.abs(position.astype(float)) - .5
        clearance = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(np.max(q, axis=1), 0)
        assert clearance.min() > 1e-5, "Particle-stage source target crosses the unit-cube surface"
        stage = {"time": float(stage_time), "index": int(state.stage_index), "count": int(state.count),
                 "position_sha256": digest(position), "minimum_cube_clearance": float(clearance.min()),
                 "body_velocity_calls": 0, "body_gradient_calls": 0}
        clock = (solver.time, solver.step, solver.time_step_size, solver.particles.step)

        def velocity_field(target_position, target_velocity, count, time):
            assert target_position is state.position and target_velocity is rates.velocity
            assert count == state.count and time == stage_time
            stage["body_velocity_calls"] += 1
            solver.panel_solver.accumulate_induced_velocity_on_field(target_position, target_velocity, count)

        def gradient(points, time):
            assert time == stage_time
            np.testing.assert_array_equal(points, position)
            value = source_panel_gradient(points, active["vertices"], active["strength"])
            stage["body_gradient_calls"] += 1
            stage["body_gradient_frobenius_rms"] = float(np.sqrt(np.mean(np.sum(value * value, axis=(1, 2)))))
            stage["body_gradient_maximum_trace"] = float(np.max(np.abs(np.trace(value, axis1=1, axis2=2))))
            stage["body_gradient_maximum_antisymmetric_entry"] = float(np.max(np.abs(value - value.transpose(0, 2, 1))))
            magnitude = max(float(np.max(np.abs(value))), np.finfo(float).tiny)
            assert stage["body_gradient_maximum_trace"] <= 1e-8 * magnitude
            assert stage["body_gradient_maximum_antisymmetric_entry"] <= 1e-8 * magnitude
            return value

        if enabled:
            solver.physics.body_velocity_field = velocity_field
            solver.physics.body_velocity_gradient = gradient
        active["in_stage"] = True
        try:
            answer = original_rhs(evaluator, state, stage_time, rates)
            assert stage["body_velocity_calls"] == stage["body_gradient_calls"] == int(enabled)
            assert (solver.time, solver.step, solver.time_step_size, solver.particles.step) == clock
            np.testing.assert_array_equal(solver.physics._download_vector_field(state.position, state.count), position)
            row["stages"].append(stage)
            return answer
        finally:
            solver.physics.body_velocity_field = None
            solver.physics.body_velocity_gradient = None
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

    VPMSolver.advance, StageRHS.evaluate = outer, rhs
    EvolutionStepper._apply_viscous_diffusion, StabilizationManager.run_phase = diffusion, phase
    try:
        yield
    finally:
        VPMSolver.advance, StageRHS.evaluate = original_outer, original_rhs
        EvolutionStepper._apply_viscous_diffusion, StabilizationManager.run_phase = original_diffusion, original_phase
