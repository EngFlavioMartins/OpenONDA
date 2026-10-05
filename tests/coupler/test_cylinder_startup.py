"""Accepted-clock startup forcing survives bounded native continuation."""

from copy import deepcopy
from dataclasses import dataclass, replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from openonda import cylinder_campaign as campaign
import openonda.coupler as coupling
from openonda.cylinder_startup import startup_velocity
from source.solvers.fvm import TimeConfig

STARTUP = (1.0, 0.1, 0.0)
STEADY = (1.0, 0.0, 0.0)


@dataclass(frozen=True)
class Numerics:
    freestream_velocity: tuple = STARTUP
    time_step_size: float = 0.04
    induction: object = None


@dataclass(frozen=True)
class Particles:
    numerics: Numerics
    run: object


class RecordingSolver:
    """Records accepted stages and insists native load matches current policy."""

    def __init__(self, root, particles, policy, *, comm=None, master=True):
        self.solution_dir = root / "solution"
        self.solution_dir.mkdir(parents=True, exist_ok=True)
        self.setup = policy
        self.freestream_velocity = np.asarray(policy.freestream_velocity)
        self._comm, self._is_master = comm, master
        self.events = []
        self.step = 0
        self.end_step = particles.run.steps
        self.positions = np.asarray([[0.2, 0.1, 0.0], [0.7, 0.3, 0.1]])
        self.fvm_solver = SimpleNamespace(
            mesh_data={"n_cells": 2},
            geo_data={"cell_centre": self.positions},
            set_initial_velocity=lambda value: self.events.append(("initial", value.copy())),
        )
        self.vpm_solver = None
        if master:
            self.vpm_solver = SimpleNamespace(
                setup=particles.numerics,
                case=particles,
                _set_freestream_velocity=lambda value: self.events.append(("background", value)),
            )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def initialize(self):
        self.events.append(("initialize", tuple(self.setup.freestream_velocity)))
        self.vorticity_transfer = SimpleNamespace(config=self.setup)

    def load_backup(self, path):
        saved = json.loads((path / "manifest.json").read_text())
        assert saved["config"]["coupler"]["freestream_velocity"] == self.setup.freestream_velocity
        if self._is_master:
            assert list(self.vpm_solver.setup.freestream_velocity) == self.setup.freestream_velocity
        self.step = saved["coupling_step"]
        self.events.append(("load", self.step, tuple(self.setup.freestream_velocity)))
        return self.step

    def run(self, *, start_from, max_coupling_steps, backup_at_stop):
        assert start_from == "initial"
        assert self.step == 0
        self.events.append(("run",))
        return self.solve(
            start_step=0, max_coupling_steps=max_coupling_steps, backup_at_stop=backup_at_stop
        )

    def _advance_vpm(self, step, time_end):
        assert time_end == step * 0.04
        self.events.append(("exchange", step, tuple(self.setup.freestream_velocity)))
        assert self.vorticity_transfer.config is self.setup
        if self._is_master:
            assert tuple(self.vpm_solver.case.numerics.freestream_velocity) == tuple(
                self.setup.freestream_velocity
            )
        return 0.0

    def solve(self, *, start_step, max_coupling_steps, backup_at_stop):
        assert backup_at_stop and start_step == self.step
        assert max_coupling_steps is None or max_coupling_steps > 0
        stop = min(self.end_step, start_step + (max_coupling_steps or self.end_step))
        velocity = tuple(self.setup.freestream_velocity)
        assert self.vorticity_transfer.config is self.setup
        if self._is_master:
            assert tuple(self.vpm_solver.case.numerics.freestream_velocity) == velocity
        self.events.append(("advance", start_step, stop, velocity))
        for step in range(start_step + 1, stop + 1):
            self._advance_vpm(step, step * 0.04)
        self.step = stop
        if self._is_master:
            path = self.solution_dir / "backups"
            path.mkdir(exist_ok=True)
            (path / "manifest.json").write_text(
                json.dumps({"coupling_step": stop, "config": self.setup.to_dict()})
            )
        return stop


@pytest.fixture
def staged_case(tmp_path, monkeypatch):
    flow = SimpleNamespace(time=TimeConfig(time_step_size=0.008, end_time=0.32))
    particles = Particles(
        Numerics(induction=SimpleNamespace(z_min=-0.48, z_max=0.48)),
        SimpleNamespace(steps=8),
    )
    policy = coupling.CouplerSetup(freestream_velocity=list(STARTUP))
    created = []

    def build(**kwargs):
        return flow, particles, policy, object()

    def create(flow, particles, policy, **kwargs):
        solver = RecordingSolver(kwargs["case_dir"], particles, policy)
        created.append(solver)
        return solver

    monkeypatch.setattr(coupling, "create_coupler", create)

    def run(**kwargs):
        return campaign.run_coupled_cylinder(
            build,
            output_root=tmp_path,
            startup_duration=kwargs.pop("startup_duration", 0.12),
            steady_freestream_velocity=kwargs.pop("steady_freestream_velocity", STEADY),
            **kwargs,
        )

    return SimpleNamespace(run=run, created=created, root=tmp_path, build=build, flow=flow)


def test_full_startup_switches_once_without_reinitializing_state(staged_case):
    assert staged_case.run() == 8
    solver = staged_case.created[-1]
    assert [item for item in solver.events if item[0] == "advance"] == [
        ("advance", 0, 3, STARTUP),
        ("advance", 3, 8, STEADY),
    ]
    initial = [item[1] for item in solver.events if item[0] == "initial"]
    assert len(initial) == 1
    np.testing.assert_allclose(
        initial[0], campaign.cylinder_initial_velocity(solver.positions, 0.96) + [0, 0.1, 0]
    )
    assert [item for item in solver.events if item[0] == "initialize"] == [("initialize", STARTUP)]


@pytest.mark.parametrize("first_stop", [1, 3, 5])
def test_bounded_restart_before_at_and_after_switch_preserves_total_cap(staged_case, first_stop):
    assert staged_case.run(max_coupling_steps=first_stop) == first_stop
    assert staged_case.run(max_coupling_steps=2) == first_stop + 2
    resumed = staged_case.created[-1]
    assert ("initialize", STARTUP) in resumed.events
    assert not any(item[0] == "initial" for item in resumed.events)
    expected_loaded_background = STARTUP if first_stop <= 3 else STEADY
    assert ("load", first_stop, expected_loaded_background) in resumed.events
    advanced = [item for item in resumed.events if item[0] == "advance"]
    assert sum(item[2] - item[1] for item in advanced) == 2
    assert all(item[3] == (STARTUP if item[1] < 3 else STEADY) for item in advanced)
    assert staged_case.run() == 8
    # Repeated latest on a completed series must not advance or reset it.
    assert staged_case.run() == 8
    assert not any(item[0] in {"initial", "advance"} for item in staged_case.created[-1].events)


def test_bounded_resume_crossing_switch_uses_remaining_not_full_cap(staged_case):
    assert staged_case.run(max_coupling_steps=2) == 2
    assert staged_case.run(max_coupling_steps=3) == 5
    assert [item for item in staged_case.created[-1].events if item[0] == "advance"] == [
        ("advance", 2, 3, STARTUP),
        ("advance", 3, 5, STEADY),
    ]


def test_exact_switch_checkpoint_with_steady_policy_is_also_admitted(staged_case):
    assert staged_case.run(max_coupling_steps=3) == 3
    path = staged_case.root / "solution/backups/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["config"]["coupler"]["freestream_velocity"] = list(STEADY)
    path.write_text(json.dumps(manifest))
    assert staged_case.run(max_coupling_steps=1) == 4
    assert ("load", 3, STEADY) in staged_case.created[-1].events


@pytest.mark.parametrize("defect", ["missing", "changed", "wrong_stage"])
def test_resume_rejects_unknown_schedule_or_inconsistent_stage_before_load(staged_case, defect):
    assert staged_case.run(max_coupling_steps=1) == 1
    if defect == "missing":
        (staged_case.root / "solution/cylinder_startup.json").unlink()
    elif defect == "changed":
        path = staged_case.root / "solution/cylinder_startup.json"
        saved = json.loads(path.read_text())
        saved["startup_duration"] = 0.2
        path.write_text(json.dumps(saved))
    else:
        path = staged_case.root / "solution/backups/manifest.json"
        saved = json.loads(path.read_text())
        saved["config"]["coupler"]["freestream_velocity"] = list(STEADY)
        path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="startup|schedule"):
        staged_case.run()
    assert not any(item[0] == "load" for item in staged_case.created[-1].events)


def test_explicit_initial_restarts_forcing_from_zero(staged_case):
    assert staged_case.run(max_coupling_steps=5) == 5
    assert staged_case.run(start_from="initial", max_coupling_steps=2) == 2
    assert ("advance", 0, 2, STARTUP) in staged_case.created[-1].events


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"startup_duration": None}, "supplied together"),
        ({"steady_freestream_velocity": None}, "supplied together"),
        ({"startup_duration": 0.13}, "integer number"),
        ({"startup_duration": float("nan")}, "finite"),
        ({"steady_freestream_velocity": (2, 0, 0)}, "startup lattice"),
        ({"steady_freestream_velocity": (1, 0)}, "three finite"),
        ({"max_coupling_steps": 0}, "positive integer"),
        ({"max_coupling_steps": True}, "positive integer"),
    ],
)
def test_invalid_startup_policy_fails_before_factory_construction(staged_case, kwargs, match):
    with pytest.raises(ValueError, match=match):
        staged_case.run(**kwargs)
    assert not staged_case.created


def test_mismatched_physical_end_is_rejected(staged_case):
    staged_case.flow.time = replace(staged_case.flow.time, end_time=0.3)
    with pytest.raises(ValueError, match="matching FVM/VPM"):
        staged_case.run()


def test_worker_uses_broadcast_selection_without_reading_checkpoint(tmp_path, monkeypatch):
    # Test the actual collective-phase wrapper with deterministic root messages.
    schedule = {
        "switch_step": 3,
        "startup_freestream_velocity": list(STARTUP),
        "steady_freestream_velocity": list(STEADY),
        "end_time": 0.32,
        "exchange_time_step": 0.04,
    }
    path = tmp_path / "root-only-checkpoint"
    comm = SimpleNamespace(
        Get_size=lambda: 2,
        Ibarrier=lambda: SimpleNamespace(Test=lambda: True),
        allgather=lambda local: [None, local],
        bcast=lambda value, root: (path, deepcopy(list(STEADY)), schedule),
    )
    particles = Particles(Numerics(), SimpleNamespace(steps=8))
    solver = RecordingSolver(
        tmp_path,
        particles,
        coupling.CouplerSetup(freestream_velocity=list(STARTUP)),
        comm=comm,
        master=False,
    )

    def load(selected):
        assert selected == path
        assert solver.setup.freestream_velocity == list(STEADY)
        solver.step = 5
        return 5

    solver.load_backup = load
    import source.restart

    monkeypatch.setattr(
        source.restart, "select_backup", lambda *args, **kwargs: pytest.fail("worker read")
    )
    assert campaign._run_cylinder_startup(solver, particles, schedule, "latest", None, 1) == 6
    assert ("advance", 5, 6, STEADY) in solver.events
    assert not (solver.solution_dir / "cylinder_startup.json").exists()


def test_smooth_startup_uses_endpoints_in_one_native_solve(staged_case):
    assert staged_case.run(startup_transition_duration=0.08) == 8
    solver = staged_case.created[-1]
    assert len([event for event in solver.events if event[0] == "advance"]) == 1
    expected = [
        ("exchange", step, startup_velocity(step * 0.04, 0.12, 0.08, STARTUP, STEADY))
        for step in range(1, 9)
    ]
    assert [event for event in solver.events if event[0] == "exchange"] == expected
    assert "_advance_vpm" not in solver.__dict__
    policy = json.loads((solver.solution_dir / "cylinder_startup.json").read_text())
    assert policy["schema"] == "openonda-cylinder-startup/2"
    assert policy["transition_step"] == 1
    assert policy["startup_transition_duration"] == 0.08
    # Runtime background mutation stops after the taper, preserving the
    # coupling predictor and ordinary cadence during the developed run.
    assert len([event for event in solver.events if event[0] == "background"]) == 2


@pytest.mark.parametrize("first_stop", [1, 2, 3, 5])
def test_smooth_restart_before_inside_at_and_after_taper(staged_case, first_stop):
    kwargs = {"startup_transition_duration": 0.08}
    assert staged_case.run(max_coupling_steps=first_stop, **kwargs) == first_stop
    sidecar = staged_case.root / "solution/cylinder_startup.json"
    original = sidecar.read_bytes()
    expected_loaded = startup_velocity(first_stop * 0.04, 0.12, 0.08, STARTUP, STEADY)
    saved = json.loads((staged_case.root / "solution/backups/manifest.json").read_text())
    assert saved["config"]["coupler"]["freestream_velocity"] == list(expected_loaded)
    assert staged_case.run(max_coupling_steps=2, **kwargs) == first_stop + 2
    resumed = staged_case.created[-1]
    assert ("load", first_stop, expected_loaded) in resumed.events
    assert len([event for event in resumed.events if event[0] == "advance"]) == 1
    assert not any(event[0] == "initial" for event in resumed.events)
    assert sidecar.read_bytes() == original
    assert [event for event in resumed.events if event[0] == "exchange"] == [
        ("exchange", step, startup_velocity(step * 0.04, 0.12, 0.08, STARTUP, STEADY))
        for step in range(first_stop + 1, first_stop + 3)
    ]


def test_new_factory_adopts_exact_legacy_native_schedule_without_rewriting(staged_case):
    assert staged_case.run(max_coupling_steps=2) == 2
    sidecar = staged_case.root / "solution/cylinder_startup.json"
    original = sidecar.read_bytes()
    assert staged_case.run(startup_transition_duration=0.08, max_coupling_steps=3) == 5
    assert sidecar.read_bytes() == original
    assert [event for event in staged_case.created[-1].events if event[0] == "advance"] == [
        ("advance", 2, 3, STARTUP),
        ("advance", 3, 5, STEADY),
    ]


def test_legacy_adoption_still_rejects_other_policy_changes(staged_case):
    assert staged_case.run(max_coupling_steps=2) == 2
    with pytest.raises(ValueError, match="identical.*schedule"):
        staged_case.run(startup_duration=0.16, startup_transition_duration=0.08)
    assert not any(event[0] == "load" for event in staged_case.created[-1].events)


@pytest.mark.parametrize("defect", ["start_velocity", "different_taper"])
def test_smooth_resume_rejects_wrong_clock_background_or_schedule(staged_case, defect):
    kwargs = {"startup_transition_duration": 0.08}
    assert staged_case.run(max_coupling_steps=2, **kwargs) == 2
    if defect == "start_velocity":
        path = staged_case.root / "solution/backups/manifest.json"
        saved = json.loads(path.read_text())
        saved["config"]["coupler"]["freestream_velocity"] = list(STARTUP)
        path.write_text(json.dumps(saved))
    else:
        kwargs["startup_transition_duration"] = 0.04
    with pytest.raises(ValueError, match="startup|schedule"):
        staged_case.run(**kwargs)
    assert not any(event[0] == "load" for event in staged_case.created[-1].events)


@pytest.mark.parametrize("duration", [-0.04, 0.16, float("nan"), 0.06])
def test_invalid_taper_duration_rejected_before_solver_construction(staged_case, duration):
    with pytest.raises(ValueError, match="startup_transition_duration"):
        staged_case.run(startup_transition_duration=duration)
    assert not staged_case.created


def test_shared_taper_has_no_jump_and_vanishing_endpoint_acceleration():
    assert startup_velocity(1.0, 2.0, 1.0, STARTUP, STEADY) == STARTUP
    assert startup_velocity(2.0, 2.0, 1.0, STARTUP, STEADY) == STEADY
    assert startup_velocity(1.5, 2.0, 1.0, STARTUP, STEADY) == (1.0, 0.05, 0.0)
    # Endpoint changes scale cubically, unlike the original finite jump.
    h = 1e-3
    values = [
        startup_velocity(2.0 - distance, 2.0, 1.0, STARTUP, STEADY)[1] for distance in (h, 2 * h)
    ]
    assert 7.9 < values[1] / values[0] < 8.1
    assert values[0] < 1e-8


def test_smooth_native_failure_restores_case_owned_advance_hook(staged_case, monkeypatch):
    def fail_advance(self, step, time_end):
        raise RuntimeError("native advance failure")

    monkeypatch.setattr(RecordingSolver, "_advance_vpm", fail_advance)
    with pytest.raises(RuntimeError, match="native advance failure"):
        staged_case.run(startup_transition_duration=0.08)
    assert "_advance_vpm" not in staged_case.created[-1].__dict__


def test_smooth_restores_preexisting_instance_advance_override(staged_case, monkeypatch):
    original_enter = RecordingSolver.__enter__

    def enter(self):
        self.previous_advance = self._advance_vpm
        self._advance_vpm = self.previous_advance
        return original_enter(self)

    monkeypatch.setattr(RecordingSolver, "__enter__", enter)
    assert staged_case.run(startup_transition_duration=0.08) == 8
    solver = staged_case.created[-1]
    assert solver.__dict__["_advance_vpm"] is solver.previous_advance


def test_smooth_worker_reconstructs_taper_from_broadcast_without_checkpoint_reads(
    staged_case, tmp_path, monkeypatch
):
    flow, particles, policy, _ = staged_case.build()
    schedule = campaign._cylinder_startup_schedule(flow, particles, policy, 0.12, STEADY, 0.08)
    path = tmp_path / "root-only-checkpoint"
    saved_velocity = list(startup_velocity(0.08, 0.12, 0.08, STARTUP, STEADY))
    comm = SimpleNamespace(
        Get_size=lambda: 2,
        Ibarrier=lambda: SimpleNamespace(Test=lambda: True),
        allgather=lambda local: [None, local],
        bcast=lambda value, root: (path, saved_velocity, schedule),
    )
    solver = RecordingSolver(tmp_path, particles, policy, comm=comm, master=False)

    def load(selected):
        assert selected == path
        assert solver.setup.freestream_velocity == saved_velocity
        solver.step = 2
        return 2

    solver.load_backup = load
    import source.restart

    monkeypatch.setattr(
        source.restart, "select_backup", lambda *args, **kwargs: pytest.fail("worker read")
    )
    assert campaign._run_cylinder_startup(solver, particles, schedule, "latest", None, 2) == 4
    assert [event for event in solver.events if event[0] == "exchange"] == [
        ("exchange", 3, STEADY),
        ("exchange", 4, STEADY),
    ]
    assert "_advance_vpm" not in solver.__dict__
    assert not (solver.solution_dir / "cylinder_startup.json").exists()


def test_smooth_zero_step_native_restart_does_not_seed_again(staged_case):
    kwargs = {"startup_transition_duration": 0.08}
    assert staged_case.run(max_coupling_steps=1, **kwargs) == 1
    path = staged_case.root / "solution/backups/manifest.json"
    saved = json.loads(path.read_text())
    saved["coupling_step"] = 0
    saved["config"]["coupler"]["freestream_velocity"] = list(STARTUP)
    path.write_text(json.dumps(saved))
    assert staged_case.run(max_coupling_steps=1, **kwargs) == 1
    assert not any(event[0] == "initial" for event in staged_case.created[-1].events)
