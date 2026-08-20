"""Offline :class:`PostProcess` integration: parity, idempotence, archived dt."""

from __future__ import annotations

import contextlib
import csv
import io

import pytest

from source.solvers.FVM import (
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
from source.solvers.FVM.sampling.postprocess import PostProcess

from ._structured_mesh import structured_box


def _config(samplers):
    return FVMSetup(
        case_name="postproc",
        time=TimeConfig.transient(time_step_size=0.01, duration=0.05, output_interval_steps=1),
        schemes=DiscretizationConfig(convection_scheme="upwind", time_scheme="euler_implicit"),
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
        samplers=samplers,
        initial_velocity=[0.2, 0.0, 0.0],
    )


def _line_sampler():
    return LineSampler(
        start=[0.1, 0.5, 0.5], end=[0.9, 0.5, 0.5], n_points=4, file_name="centerline"
    )


def _rows(path):
    with path.open(newline="") as stream:
        return list(csv.reader(stream))


def _build_archive(tmp_path, samplers):
    """Run a short live solve that writes a snapshot every accepted step."""
    solver = FVMSolver(_config(samplers), str(tmp_path), mesh_data=structured_box(3, 3, 3))
    solver.auto_write = True
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(4):
            solver.advance()
        solver.flush_output()


def _make_post(tmp_path, samplers):
    return PostProcess(
        case_dir=str(tmp_path),
        config=_config(samplers),
        samplers=samplers,
        mesh=structured_box(3, 3, 3),
    )


def test_postprocess_replay_matches_live_sampling(tmp_path):
    samplers = (_line_sampler(),)
    _build_archive(tmp_path, samplers)
    live = _rows(tmp_path / "samples" / "centerline.csv")

    with contextlib.redirect_stdout(io.StringIO()):
        _make_post(tmp_path, samplers).run()
    offline = _rows(tmp_path / "samples" / "centerline.csv")

    assert offline[0] == live[0]
    assert len(offline) == len(live)
    assert [row[0] for row in offline] == [row[0] for row in live]


def test_postprocess_rerun_is_idempotent(tmp_path):
    samplers = (_line_sampler(),)
    _build_archive(tmp_path, samplers)
    with contextlib.redirect_stdout(io.StringIO()):
        _make_post(tmp_path, samplers).run()
    first = _rows(tmp_path / "samples" / "centerline.csv")

    with contextlib.redirect_stdout(io.StringIO()):
        _make_post(tmp_path, samplers).run()
    second = _rows(tmp_path / "samples" / "centerline.csv")

    assert first == second
    assert len(second) == 1 + 4 * 4  # header + 4 points x 4 archived steps


def test_postprocess_uses_diagnostics_dt_before_pvd_spacing(tmp_path, monkeypatch):
    """Accepted diagnostic dt is authoritative when PVD frame spacing differs."""
    from source.solvers.FVM.sampling import postprocess as pp_module

    samplers = (_line_sampler(),)
    _build_archive(tmp_path, samplers)
    post = _make_post(tmp_path, samplers)

    # Fabricate a non-uniform archive over the same real snapshots.
    real_frames = post._pvd_frames()
    fake_frames = [
        (0.00, real_frames[0][1], real_frames[0][2]),
        (0.05, real_frames[1][1], real_frames[1][2]),
        (0.06, real_frames[2][1], real_frames[2][2]),
    ]
    monkeypatch.setattr(post, "_pvd_frames", lambda: fake_frames)

    recorded_time_step_size = []
    original_execute = pp_module.FVMSamplerExecutor.execute

    def recording_execute(context, *, strict=False):
        recorded_time_step_size.append(float(context._current_time_step_size))
        return original_execute(context, strict=strict)

    monkeypatch.setattr(pp_module.FVMSamplerExecutor, "execute", staticmethod(recording_execute))

    with contextlib.redirect_stdout(io.StringIO()):
        post.run()

    # The real archive diagnostics record dt=0.01 for every accepted step;
    # fabricated PVD spacing must not replace that physical timestep.
    assert recorded_time_step_size == pytest.approx([0.01, 0.01, 0.01])
