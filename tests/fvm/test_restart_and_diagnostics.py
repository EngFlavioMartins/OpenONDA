"""The FVM checkpoint must restore the complete transient state."""

from __future__ import annotations

import contextlib
import io
import json

import numpy as np
import pytest

from source.solvers.fvm import (
    BoundaryConfig,
    DiscretizationConfig,
    FVMSetup,
    FVMSolver,
    LinearSolverConfig,
    LineSampler,
    PimpleControl,
    TimeConfig,
    TransportConfig,
)
from source.solvers.fvm.io.checkpoint import decode_state, encode_state
from source.solvers.fvm.mesh.cartesian import structured_box


def _setup() -> FVMSetup:
    return FVMSetup(
        case_name="restart_contract",
        time=TimeConfig.transient(time_step_size=0.01, duration=0.1, output_interval_steps=100),
        schemes=DiscretizationConfig(convection_scheme="upwind", time_scheme="backward"),
        linear=LinearSolverConfig(linear_solver="spsolve"),
        pimple=PimpleControl(n_correctors=2),
        transport=TransportConfig(density=1.0, kinematic_viscosity=0.02),
        boundaries=[
            BoundaryConfig.inlet("xmin", [0.5, 0.0, 0.0]),
            BoundaryConfig.outlet("xmax"),
            BoundaryConfig.wall("ymin"),
            BoundaryConfig.wall("ymax"),
            BoundaryConfig.wall("zmin"),
            BoundaryConfig.wall("zmax"),
        ],
        initial_velocity=[0.2, 0.0, 0.0],
    )


def _solver(case_dir):
    with contextlib.redirect_stdout(io.StringIO()):
        solver = FVMSolver(_setup(), str(case_dir), mesh_data=structured_box(2, 2, 2))
    solver.auto_write = False
    return solver


def test_restart_restores_backward_time_history(tmp_path):
    reference = _solver(tmp_path / "reference")
    interrupted = _solver(tmp_path / "interrupted")
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(3):
            reference.advance()
        for _ in range(2):
            interrupted.advance()

    checkpoint = tmp_path / "restart.npz"
    interrupted.save_state(checkpoint)
    resumed = _solver(tmp_path / "resumed")
    resumed.load_state(checkpoint)
    with contextlib.redirect_stdout(io.StringIO()):
        resumed.advance()

    for field_name in (
        "velocity",
        "kinematic_pressure",
        "volumetric_face_flux",
        "volumetric_face_flux_old",
        "volumetric_face_flux_older",
        "velocity_old",
        "velocity_older",
    ):
        np.testing.assert_allclose(
            getattr(resumed, field_name), getattr(reference, field_name), atol=1e-13
        )
    assert resumed.time == pytest.approx(reference.time)
    assert resumed.step == reference.step


def test_run_manifest_serializes_sampler_configuration(tmp_path):
    setup = _setup()
    setup.samplers = (
        LineSampler(start=[0, 0, 0], end=[1, 0, 0], n_points=3, file_name="centreline"),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        solver = FVMSolver(setup, str(tmp_path), mesh_data=structured_box(2, 2, 2))
    destination = tmp_path / "run_manifest.json"
    solver.write_run_manifest(destination)
    solver.close()

    payload = json.loads(destination.read_text(encoding="utf-8"))
    sampler = payload["configuration"]["samplers"][0]
    assert sampler["type"] == "LineSampler"
    assert sampler["file_name"] == "centreline"
    assert sampler["n_points"] == 3


def test_checkpoint_storage_codec_is_bit_exact_for_history_and_scalars():
    velocity = np.linspace(-2.0, 3.0, 4096, dtype=np.float64).reshape(-1, 1)
    flux = np.linspace(-1.0, 1.0, 4096, dtype=np.float64)
    state = {
        "metadata": np.asarray('{"format_version": 8}'),
        "velocity": velocity,
        "velocity_old": velocity * (1.0 + np.finfo(np.float64).eps),
        "velocity_older": velocity * (1.0 - np.finfo(np.float64).eps),
        "volumetric_face_flux": flux,
        "volumetric_face_flux_old": flux + np.finfo(np.float64).eps,
        "volumetric_face_flux_older": flux - np.finfo(np.float64).eps,
        "step": np.asarray(42, dtype=np.int64),
    }

    stored = encode_state(state)
    restored = decode_state(stored)

    assert set(stored) == {*state, "storage_layout"}
    assert restored["metadata"].shape == ()
    for name, expected in state.items():
        np.testing.assert_array_equal(restored[name], expected)

    direct, compact = io.BytesIO(), io.BytesIO()
    np.savez_compressed(direct, **state)
    np.savez_compressed(compact, **stored)
    assert len(compact.getvalue()) < len(direct.getvalue()) / 2
