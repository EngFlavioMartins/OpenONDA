"""Reference startup preserves native restart and the steady slip trace."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openonda import fvm
from openonda.cylinder_campaign import cylinder_initial_velocity
from openonda.cylinder_reference_startup import run_reference_cylinder

STARTUP = [1.0, 0.1, 0.0]


def setup(end=0.06):
    return fvm.FVMSetup(
        case_name="reference_startup",
        time=fvm.TimeConfig(time_step_size=0.01, end_time=end),
        backup=fvm.BackupConfig(path="backup", write_at_end=True),
        boundaries=[
            fvm.BoundaryConfig.inlet("inlet", STARTUP),
            fvm.BoundaryConfig.outlet("outlet"),
            fvm.BoundaryConfig.slip("ymin"),
            fvm.BoundaryConfig.slip("ymax"),
            fvm.BoundaryConfig.slip("zmin"),
            fvm.BoundaryConfig.slip("zmax"),
        ],
        initial_velocity=STARTUP,
    )


class Solver:
    def __init__(self, directory, *, end=0.06, fail=False):
        self.setup = self._resolved_setup = setup(end)
        self.solution_dir = directory / "solution"
        self.solution_dir.mkdir(parents=True, exist_ok=True)
        self.parallel = SimpleNamespace(comm=None, is_root=True)
        self.step, self.time = 0, 0.0
        self._restart_loaded = False
        self.events = []
        self.fail = fail
        self.mesh_data = {"n_cells": 2}
        self.positions = np.array([[0.2, 0.1, 0], [0.7, 0.3, 0.1]])
        self.geo_data = {"cell_centre": self.positions}
        self.inlet, self.laterals = None, {}
        self.boundaries = [{"name": name, "velocity_type": "slip"} for name in ("ymin", "ymax")]

    def start_from(self, selection):
        self.events.append(("select", selection))
        path = self.solution_dir / "backup"
        if selection != "initial" and path.is_file():
            self.step = json.loads(path.read_text())["step"]
            self.time = self.step * 0.01
            self._restart_loaded = True
        return self._restart_loaded

    def set_initial_velocity(self, value):
        self.events.append(("seed", value.copy()))

    def get_boundary_face_normal(self, name):
        return np.tile(
            {"inlet": [-1.0, 0, 0], "ymin": [0, -1.0, 0], "ymax": [0, 1.0, 0]}[name], (2, 1)
        )

    def set_dirichlet_velocity_boundary_condition_vec(self, value, name):
        assert name == "inlet"
        self.inlet = value.copy()
        self.events.append(("inlet", self.step, tuple(value[0])))

    def set_normal_velocity_tangential_gradient_boundary_condition(self, normal, tangent, name):
        assert not np.any(tangent)
        self.laterals[name] = normal.copy()
        self.events.append((name, self.step, tuple(normal)))
        patch = next(boundary for boundary in self.boundaries if boundary["name"] == name)
        patch.update(
            velocity_type="normalValueTangentialGradient",
            normal_velocity_field=normal.copy(),
            tangential_gradient_field=tangent.copy(),
            max_removed_tangential_gradient_normal_component=0.0,
        )

    def save_state(self, path):
        self.events.append(("save", self.step))
        Path(path).write_text(json.dumps({"step": self.step}))

    def advance(self):
        self.events.append(("advance", self.step, tuple(self.inlet[0])))
        if self.fail:
            raise RuntimeError("deliberate physical failure")
        expected = (
            self.expected_transverse[self.step]
            if hasattr(self, "expected_transverse")
            else (0.1 if self.step < 2 else 0)
        )
        np.testing.assert_allclose(self.inlet[:, 1], expected, atol=1e-15, rtol=0)
        np.testing.assert_allclose(self.laterals["ymin"], -expected, atol=1e-15, rtol=0)
        np.testing.assert_allclose(self.laterals["ymax"], expected, atol=1e-15, rtol=0)
        for patch in self.boundaries:
            if expected != 0:
                assert patch["velocity_type"] == "normalValueTangentialGradient"
            else:
                assert patch == {"name": patch["name"], "velocity_type": "slip"}
        self.step += 1
        self.time = self.step * 0.01

    def run(self):
        self.events.append(("run",))
        while self.time < self.setup.time.end_time - 1e-12:
            self.advance()
        self.save_state(self.solution_dir / "backup")


def run(solver, **kwargs):
    duration = kwargs.pop("startup_duration", 0.02)
    return run_reference_cylinder(solver, span=0.96, startup_duration=duration, **kwargs)


def smooth_solver(directory, *, end=0.06):
    solver = Solver(directory, end=end)
    solver.expected_transverse = [0.1, 0.1, 0.05, 0.0, 0.0, 0.0]
    return solver


def run_smooth(solver, **kwargs):
    return run(solver, startup_duration=0.04, startup_transition_duration=0.02, **kwargs)


def test_smooth_boundary_uses_accepted_endpoints_and_steady_fast_path(tmp_path):
    solver = smooth_solver(tmp_path)
    run_smooth(solver)
    updates = [event for event in solver.events if event[0] == "inlet"]
    assert [event[1] for event in updates] == [0, 2, 3]
    np.testing.assert_allclose([event[2][1] for event in updates], [0.1, 0.05, 0], atol=1e-15)
    policy = json.loads((solver.solution_dir / "reference_startup.json").read_text())
    assert policy["schema"] == "openonda-reference-cylinder-startup/2"
    assert policy["startup_transition_duration"] == 0.02
    assert policy["boundary_time_evaluation"] == "accepted endpoint"


@pytest.mark.parametrize("first_end", [0.01, 0.02, 0.03, 0.04, 0.05])
def test_smooth_resume_reconstructs_accepted_boundary_then_uses_next_endpoint(tmp_path, first_end):
    run_smooth(smooth_solver(tmp_path, end=first_end))
    resumed = smooth_solver(tmp_path)
    run_smooth(resumed)
    assert not any(event[0] == "seed" for event in resumed.events)
    updates = [event for event in resumed.events if event[0] == "inlet"]
    expected = {0.01: 0.1, 0.02: 0.1, 0.03: 0.05, 0.04: 0.0, 0.05: 0.0}[first_end]
    assert updates[0][1] == round(first_end / 0.01)
    assert updates[0][2][1] == pytest.approx(expected, abs=1e-15)


def test_known_legacy_checkpoint_retains_abrupt_policy_when_factory_adds_smoothing(tmp_path):
    run(Solver(tmp_path, end=0.01))
    path = tmp_path / "solution/reference_startup.json"
    before = path.read_bytes()
    resumed = Solver(tmp_path)
    run(resumed, startup_transition_duration=0.01)
    assert path.read_bytes() == before
    assert [event for event in resumed.events if event[0] == "inlet"] == [
        ("inlet", 1, (1.0, 0.1, 0)),
        ("inlet", 2, (1.0, 0, 0)),
    ]


@pytest.mark.parametrize("defect", ["unknown_schema", "transition", "profile", "clock"])
def test_smooth_schedule_mutations_rejected_before_native_load(tmp_path, defect):
    run_smooth(smooth_solver(tmp_path, end=0.01))
    path = tmp_path / "solution/reference_startup.json"
    policy = json.loads(path.read_text())
    if defect == "unknown_schema":
        policy["schema"] = "openonda-reference-cylinder-startup/3"
    elif defect == "transition":
        policy["startup_transition_duration"] = 0.03
    elif defect == "profile":
        policy["startup_transition_profile"] = "linear"
    else:
        policy["boundary_time_evaluation"] = "step start"
    path.write_text(json.dumps(policy))
    resumed = smooth_solver(tmp_path)
    with pytest.raises(ValueError, match="different forcing policy"):
        run_smooth(resumed)
    assert not resumed.events


@pytest.mark.parametrize("transition", [-0.01, 0.03, float("nan"), float("inf")])
def test_invalid_transition_duration_rejected(tmp_path, transition):
    solver = Solver(tmp_path)
    with pytest.raises(ValueError, match="startup_transition_duration"):
        run(solver, startup_transition_duration=transition)
    assert not solver.events


def test_native_selection_precedes_seed_and_update_occurs_only_at_switch(tmp_path):
    solver = Solver(tmp_path)
    original_advance = solver.advance
    run(solver)
    assert solver.events[0] == ("select", "latest")
    seed = next(event[1] for event in solver.events if event[0] == "seed")
    np.testing.assert_allclose(
        seed, cylinder_initial_velocity(solver.positions, 0.96) + [0, 0.1, 0]
    )
    assert [event for event in solver.events if event[0] == "inlet"] == [
        ("inlet", 0, (1.0, 0.1, 0)),
        ("inlet", 2, (1.0, 0, 0)),
    ]
    assert solver.advance == original_advance
    assert "advance" not in solver.__dict__
    assert ("save", 0) in solver.events
    assert solver.setup.boundaries[2].velocity_type == "slip"


@pytest.mark.parametrize("first_end", [0.01, 0.02, 0.03])
def test_native_resume_before_at_after_switch_does_not_reseed(tmp_path, first_end):
    run(Solver(tmp_path, end=first_end))
    resumed = Solver(tmp_path)
    run(resumed)
    assert not any(event[0] == "seed" for event in resumed.events)
    assert resumed.step == 6
    first_step = round(first_end / 0.01)
    expected = (1, 0.1, 0) if first_step < 2 else (1, 0, 0)
    assert ("inlet", first_step, expected) in resumed.events
    assert resumed.events[0] == ("select", "latest")


def test_zero_native_checkpoint_preserves_initial_field(tmp_path):
    first = Solver(tmp_path, fail=True)
    with pytest.raises(RuntimeError, match="deliberate"):
        run(first)
    assert json.loads((first.solution_dir / "backup").read_text())["step"] == 0
    resumed = Solver(tmp_path)
    run(resumed)
    assert not any(event[0] == "seed" for event in resumed.events)


def test_advance_override_restored_even_after_error(tmp_path):
    solver = Solver(tmp_path, fail=True)
    original_advance = solver.advance
    solver.advance = original_advance
    with pytest.raises(RuntimeError, match="deliberate"):
        run(solver)
    assert solver.__dict__["advance"] == original_advance


@pytest.mark.parametrize("loaded_step, loaded_flag", [(0, True), (1, True), (1, False)])
def test_in_memory_resume_requires_schedule_without_reseeding(tmp_path, loaded_step, loaded_flag):
    solver = Solver(tmp_path)
    solver.step, solver.time = loaded_step, loaded_step * 0.01
    solver._restart_loaded = loaded_flag
    with pytest.raises(ValueError, match="reference_startup.json"):
        run(solver, start_from=None)
    assert not solver.events

    run(Solver(tmp_path, end=0.01))
    run(solver, start_from=None)
    assert not any(event[0] in ("seed", "select") for event in solver.events)


def test_fresh_in_memory_start_is_seeded(tmp_path):
    solver = Solver(tmp_path)
    run(solver, start_from=None)
    assert any(event[0] == "seed" for event in solver.events)


@pytest.mark.parametrize("defect", ["missing", "different", "corrupt"])
def test_resume_requires_immutable_schedule_before_native_load(tmp_path, defect):
    run(Solver(tmp_path, end=0.01))
    path = tmp_path / "solution/reference_startup.json"
    if defect == "missing":
        path.unlink()
    elif defect == "corrupt":
        path.write_text("{broken JSON")
    else:
        data = json.loads(path.read_text())
        data["startup_duration"] = 0.03
        path.write_text(json.dumps(data))
    solver = Solver(tmp_path)
    with pytest.raises(ValueError, match="reference_startup.json") as error:
        run(solver)
    assert "./allrun.sh --fresh" in str(error.value)
    assert not solver.events


@pytest.mark.parametrize("defect", ["initial", "inlet", "lateral", "clock"])
def test_incompatible_factory_policy_rejected(tmp_path, defect):
    solver = Solver(tmp_path)
    if defect == "initial":
        solver.setup.initial_velocity = [1, 0, 0]
    elif defect == "inlet":
        solver.setup.boundaries[0].velocity_value = [1, 0, 0]
    elif defect == "lateral":
        solver.setup.boundaries[2].velocity_type = "freestream"
    else:
        solver.setup.time = replace(solver.setup.time, time_step_size=0.013)
    with pytest.raises(ValueError):
        run(solver)


def test_actual_fvm_restart_crossflow_switch_matches_continuous(tmp_path, monkeypatch):
    """Native CPU checkpoints retain BDF histories while lateral traces change."""
    from source.solvers.fvm.assemble.diffusion import assemble_diffusion_term
    from source.solvers.fvm.mesh.rectilinear import coupling_box_mesh

    def create(directory, end):
        physical = setup(end)
        physical.schemes = fvm.DiscretizationConfig(time_scheme="backward")
        physical.transport = fvm.TransportConfig(kinematic_viscosity=0.01)
        physical.linear = fvm.LinearSolverConfig(
            momentum_solver="spsolve",
            pressure_solver="spsolve",
            pressure_tolerance=1e-13,
            pressure_relative_tolerance=0,
            momentum_tolerance=1e-13,
            momentum_relative_tolerance=0,
        )
        mesh = coupling_box_mesh(
            (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5),
            0.25,
            separate_outer=("inlet", "outlet", "ymin", "ymax", "zmin", "zmax"),
        )
        return fvm.create_fvm_solver(physical, case_dir=directory, mesh=mesh)

    def snapshot(solver):
        return {
            name: getattr(solver, name).copy()
            for name in ("velocity", "kinematic_pressure", "volumetric_face_flux")
        }

    with create(tmp_path / "continuous", 0.06) as continuous:
        run(continuous)
        expected = snapshot(continuous)
        assert continuous.last_diagnostics.n_nonfinite_values == 0
        assert continuous.last_diagnostics.max_continuity_error < 1e-7
        # Steady lateral conditions regain both the original impermeable trace
        # and the original zero diffusion flux (not the zero-normal mixed BC).
        for name in ("ymin", "ymax"):
            normal = continuous.get_boundary_face_normal(name)
            velocity = continuous.get_boundary_face_velocity(name)
            np.testing.assert_allclose(np.einsum("ij,ij->i", normal, velocity), 0, atol=1e-14)
        for component in range(3):
            fluxes = assemble_diffusion_term(
                continuous.velocity[:, component],
                np.zeros((len(continuous.velocity), 3)),
                0.01,
                continuous.mesh_data,
                continuous.geo_data,
                continuous.boundaries,
                vector_field=continuous.velocity,
                component=component,
            )
            for patch in continuous.boundaries:
                if patch["name"] not in ("ymin", "ymax"):
                    continue
                assert patch["velocity_type"] == "slip"
                assert "normal_velocity_field" not in patch
                assert "tangential_gradient_field" not in patch
                assert "max_removed_tangential_gradient_normal_component" not in patch
                start = patch["start_face"]
                indices = slice(start, start + patch["n_faces"])
                for flux in fluxes.values():
                    np.testing.assert_array_equal(flux[indices], 0)
    for end in (0.01, 0.02, 0.03, 0.06):
        with create(tmp_path / "resumed", end) as resumed:
            run(resumed)
            actual = snapshot(resumed)
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], atol=1e-12, rtol=0)

    # Corrupt the native numerical identity without altering the startup
    # sidecar. Admission must still fail before the initial seed is considered.
    backup = tmp_path / "resumed/solution/backup"
    with np.load(backup, allow_pickle=False) as saved:
        stored = {key: saved[key].copy() for key in saved.files}
    metadata = json.loads(str(stored["metadata"]))
    metadata["config_hash"] = "0" * 64
    stored["metadata"] = np.asarray(json.dumps(metadata))
    with backup.open("wb") as stream:
        np.savez(stream, **stored)
    monkeypatch.setattr(
        "openonda.cylinder_reference_startup.initialize_cylinder_perturbation",
        lambda *args, **kwargs: pytest.fail("invalid native restart must not reseed"),
    )
    with (
        create(tmp_path / "resumed", 0.08) as invalid,
        pytest.raises(RuntimeError, match="configuration hash"),
    ):
        run(invalid)


def test_actual_fvm_smooth_pressure_pulse_and_mid_taper_restart(tmp_path):
    """Resolved C2 forcing reduces the native pressure pulse and restarts exactly.

    This small box qualifies the unsteady boundary/pressure response, not the
    cylinder's force coefficient or its shedding amplitude.
    """
    from source.solvers.fvm.mesh.rectilinear import coupling_box_mesh

    def create(directory, end):
        physical = setup(end)
        physical.schemes = fvm.DiscretizationConfig(time_scheme="backward")
        physical.transport = fvm.TransportConfig(kinematic_viscosity=0.01)
        physical.linear = fvm.LinearSolverConfig(
            momentum_solver="spsolve",
            pressure_solver="spsolve",
            pressure_tolerance=1e-13,
            pressure_relative_tolerance=0,
            momentum_tolerance=1e-13,
            momentum_relative_tolerance=0,
        )
        mesh = coupling_box_mesh(
            (-0.48, 0.48, -0.48, 0.48, -0.48, 0.48),
            0.24,
            separate_outer=("inlet", "outlet", "ymin", "ymax", "zmin", "zmax"),
        )
        return fvm.create_fvm_solver(physical, case_dir=directory, mesh=mesh)

    def evolve(directory, end, transition):
        with create(directory, end) as solver:
            trace = []
            original_advance = solver.advance

            def measured_advance():
                result = original_advance()
                pressure = solver.kinematic_pressure[: solver.mesh_data["n_cells"]]
                trace.append(
                    {
                        "time": solver.time,
                        "pressure_rms": float(np.sqrt(np.mean(pressure**2))),
                        "continuity": solver.last_diagnostics.max_continuity_error,
                    }
                )
                return result

            solver.advance = measured_advance
            run(solver, startup_duration=0.2, startup_transition_duration=transition)
            assert solver.advance is measured_advance
            fields = {
                name: getattr(solver, name).copy()
                for name in ("velocity", "kinematic_pressure", "volumetric_face_flux")
            }
            assert all(np.isfinite(field).all() for field in fields.values())
            assert max(row["continuity"] for row in trace) < 1e-7
            return fields, trace

    expected, smooth = evolve(tmp_path / "smooth", 0.24, 0.1)
    _, abrupt = evolve(tmp_path / "abrupt", 0.24, 0)
    resumed_trace = []
    for end in (0.1, 0.14, 0.2, 0.24):
        actual, trace = evolve(tmp_path / "resumed", end, 0.1)
        resumed_trace.extend(trace)
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        [row["pressure_rms"] for row in resumed_trace],
        [row["pressure_rms"] for row in smooth],
        atol=1e-12,
        rtol=0,
    )

    def pulse(rows):
        return max(row["pressure_rms"] for row in rows if row["time"] >= 0.09)

    assert pulse(smooth) < 0.5 * pulse(abrupt)
    report = {
        "scope": "64-cell CPU box, not cylinder force validation",
        "dt": 0.01,
        "startup_duration": 0.2,
        "transition_duration": 0.1,
        "pressure_rms_pulse_smooth": pulse(smooth),
        "pressure_rms_pulse_abrupt": pulse(abrupt),
        "pulse_ratio": pulse(smooth) / pulse(abrupt),
        "smooth_trace": smooth,
        "abrupt_trace": abrupt,
        "restart_max_absolute_difference": {
            name: float(np.max(abs(actual[name] - expected[name]))) for name in expected
        },
    }
    (tmp_path / "pressure_pulse_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
