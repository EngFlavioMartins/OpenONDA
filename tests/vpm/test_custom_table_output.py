"""Framework-owned transactions for declared scientific table schemas."""

import csv
from types import SimpleNamespace

import numpy as np
import pytest

from source.solvers.vpm.config.artifacts import Backup, Samplers
from source.solvers.vpm.io.sampler import OutputEvent, OutputManager
from source.solvers.vpm.io.sampling import EverySteps
from tests._tutorial_helpers import load_tutorial_module


class TableSampler:
    file_name = "measurements"
    schedule = EverySteps(1)
    csv_columns = ("group_id", "strength")

    def sample(self, solver):
        return {"group_id": np.array([0, 1]), "strength": np.array([solver.time, 2.0])}


def _solver(directory, sampler):
    return SimpleNamespace(
        case_dir=directory,
        case=SimpleNamespace(backup=Backup(), samplers=Samplers((sampler,))),
        step=1,
        time=0.1,
        time_step_size=0.1,
        _restart_loaded=False,
    )


def _rows(path):
    with path.open(newline="") as stream:
        return list(csv.reader(stream))


def test_custom_table_rewinds_and_appends_using_native_transactions(tmp_path):
    solver = _solver(tmp_path, TableSampler())
    manager = OutputManager(solver)
    manager.dispatch(OutputEvent.ACCEPTED_STEP)
    solver.step, solver.time = 2, 0.2
    manager.dispatch(OutputEvent.ACCEPTED_STEP)
    path = tmp_path / "samples/measurements.csv"
    assert _rows(path) == [
        ["time", "step", "group_id", "strength"],
        ["0.1", "1", "0", "0.1"],
        ["0.1", "1", "1", "2.0"],
        ["0.2", "2", "0", "0.2"],
        ["0.2", "2", "1", "2.0"],
    ]
    manager.rewind_histories(0.1)
    assert len(_rows(path)) == 3
    assert list(
        (tmp_path / "samples/restart-branches").glob("before-*/measurements.csv.superseded")
    )
    solver._restart_loaded = True
    manager.dispatch(OutputEvent.ACCEPTED_STEP)
    accepted = path.read_bytes()
    assert len(_rows(path)) == 5
    with pytest.raises(RuntimeError, match="duplicate or nonmonotonic"):
        manager.dispatch(OutputEvent.ACCEPTED_STEP)
    assert path.read_bytes() == accepted
    assert not list(path.parent.glob("*.tmp*"))


@pytest.mark.parametrize(
    "columns", [(), ("strength", "strength"), ("time",), ("step",), (1,), "strength"]
)
def test_custom_table_rejects_invalid_declared_columns(tmp_path, columns):
    sampler = TableSampler()
    sampler.csv_columns = columns
    with pytest.raises(RuntimeError, match="csv_columns"):
        OutputManager(_solver(tmp_path, sampler)).dispatch(OutputEvent.ACCEPTED_STEP)
    assert not (tmp_path / "samples/measurements.csv").exists()


@pytest.mark.parametrize(
    "data",
    [
        {"group_id": np.array([0])},
        {"group_id": np.array([0]), "strength": np.array([1.0]), "extra": np.array([2.0])},
    ],
)
def test_custom_table_rejects_results_outside_declared_schema(tmp_path, data):
    sampler = TableSampler()
    sampler.sample = lambda _solver: data
    with pytest.raises(RuntimeError, match="declared csv_columns"):
        OutputManager(_solver(tmp_path, sampler)).dispatch(OutputEvent.ACCEPTED_STEP)
    assert not (tmp_path / "samples/measurements.csv").exists()


def test_custom_table_rejects_changed_persisted_schema_without_rewriting(tmp_path):
    sampler = TableSampler()
    solver = _solver(tmp_path, sampler)
    manager = OutputManager(solver)
    manager.dispatch(OutputEvent.ACCEPTED_STEP)
    path = tmp_path / "samples/measurements.csv"
    accepted = path.read_bytes()
    sampler.csv_columns = ("strength", "group_id")
    solver.step, solver.time = 2, 0.2
    with pytest.raises(RuntimeError, match="schema mismatch"):
        manager.dispatch(OutputEvent.ACCEPTED_STEP)
    assert path.read_bytes() == accepted


def test_ring_tables_preserve_physical_moments_and_widnall_amplitudes(tmp_path):
    diagnostics = load_tutorial_module("vpm/vortex_ring", "assets.ring_diagnostics")
    theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    radius = 1.0 + 0.03 * np.cos(3 * theta)
    position = np.column_stack(
        (0.02 * np.sin(3 * theta), radius * np.cos(theta), radius * np.sin(theta))
    )
    strength = radius[:, None] * np.column_stack(
        (np.zeros_like(theta), -np.sin(theta), np.cos(theta))
    )
    ring = diagnostics.RingDiagnosticsSampler(schedule=EverySteps(1))
    solver = _solver(tmp_path, ring)
    solver.particle_position = position
    solver.particle_vortex_strength = strength
    solver.particle_group_id = np.full(len(theta), 7, dtype=np.int32)
    data = ring.sample(solver)
    expected = ring._sample_group(position, strength)
    np.testing.assert_allclose([data[name][0] for name in ring.csv_columns[1:]], expected)
    assert data["group_id"].tolist() == [7]
    OutputManager(solver).dispatch(OutputEvent.ACCEPTED_STEP)
    assert _rows(tmp_path / "samples/ring_diagnostics.csv")[0] == list(diagnostics.CSV_COLUMNS)

    modes = diagnostics.RingModeDiagnosticsSampler(
        max_mode=3,
        azimuthal_bins=32,
        reference_radius=1.0,
        transverse_origin=(0.0, 0.0),
        schedule=EverySteps(1),
    )
    measured = modes.sample(solver)
    np.testing.assert_allclose(measured["radial_amplitude"], [0.0, 0.0, 0.03], atol=1.0e-15)
    np.testing.assert_allclose(measured["axial_amplitude"], [0.0, 0.0, 0.02], atol=1.0e-15)
    solver.case.samplers = Samplers((modes,))
    OutputManager(solver).dispatch(OutputEvent.ACCEPTED_STEP)
    assert _rows(tmp_path / "samples/ring_modes.csv")[0] == list(diagnostics.MODE_CSV_COLUMNS)
