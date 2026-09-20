"""Numerical qualification checks for the isolated cylinder campaign."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/reference_flow"


def load(name):
    spec = importlib.util.spec_from_file_location(
        "cylinder_campaign_" + name, CASE / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_campaign_preserves_domain_and_realized_geometric_ratio(tmp_path, monkeypatch):
    module = load("setup")
    calls = []
    monkeypatch.setattr(
        module.fvm,
        "create_fvm_solver",
        lambda config, **kw: calls.append((config, kw)) or SimpleNamespace(),
    )
    for h in (0.08, 0.08 / np.sqrt(2), 0.04, 0.04 / np.sqrt(2)):
        solver = module.create_solver(
            str(h), h, output_root=tmp_path, cores=2, span_layers=4, lean=True
        )
        assert not solver.auto_write
    sizes = []
    for config, args in calls:
        mesh = args["mesh"]
        sizes.append(mesh.source.effective_cell_size(mesh.source.patch_refinements[0].cell_size))
        assert mesh.domain.bounds == (-8.0, 24.0, -10.0, 10.0, -0.5, 0.5)
        assert len(mesh.levels) == 5
        assert args["require_empty_output"]
        assert not any(isinstance(s, module.fvm.SurfaceSampler) for s in config.samplers)
        assert config.cores == 2
        lines = {sampler.file_name: sampler for sampler in config.samplers}
        assert lines["span_probe"].k == 1
        assert lines["span_probe"].reconstruction == "idw"
        assert lines["centreline"].k == 12
        assert lines["centreline"].reconstruction == "affine"
    assert np.array(sizes[:-1]) / sizes[1:] == pytest.approx(np.sqrt(2))
    assert sizes[-2] == pytest.approx(0.03)


def test_sampling_noise_blocks_spurious_drag_gci():
    module = load("postprocess_grid_study")
    grids = [
        {"cell_size": h, "statistics": {"mean_drag": 1 + h * h, "drag_batch_95_half_width": 0.01}}
        for h in [0.08, 0.04, 0.02]
    ]
    estimate = module._convergence(grids, "mean_drag")
    assert not estimate["available"]
    assert "sampling" in estimate["reason"]
    for grid in grids:
        grid["statistics"]["drag_batch_95_half_width"] = 0.0
    estimate = module._convergence(grids, "mean_drag")
    assert estimate["available"]
    assert estimate["observed_order"] == pytest.approx(2.0)
    assert estimate["fine_grid_gci"] == pytest.approx(1.25 * 0.02**2 / (1 + 0.02**2))


def test_allrun_lists_the_physical_family_without_shell_infrastructure(tmp_path):
    import os
    import shlex
    import subprocess

    stub = tmp_path / "python"
    stub.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$SHELL_TEST_LOG"\nexit 0\n')
    stub.chmod(0o755)
    env = dict(
        os.environ,
        PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
        SHELL_TEST_LOG=str(tmp_path / "calls"),
    )
    subprocess.run(["/bin/bash", str(CASE / "allrun.sh")], env=env, check=True, cwd=tmp_path)
    calls = [shlex.split(line) for line in (tmp_path / "calls").read_text().splitlines()]
    assert len(calls) == 7
    values = [{args[i]: args[i + 1] for i in range(1, len(args) - 1, 2)} for args in calls]
    assert [row["--name"] for row in values] == [
        "xy_coarse",
        "xy_medium",
        "xy_fine",
        "xy_finer",
        "z_eight",
        "span_two",
        "dt_half",
    ]
    assert [float(row["--dx"]) for row in values[:4]] == pytest.approx(
        [0.08, 0.08 / np.sqrt(2), 0.04, 0.04 / np.sqrt(2)]
    )
    assert [int(row["--span-layers"]) for row in values] == [4, 4, 4, 4, 8, 8, 4]
    assert all(float(row["--end-time"]) == 100 for row in values)
    assert all("geometric_xy_100s" in row["--output-root"] for row in values)
    for row in values:
        if row["--name"] == "xy_fine":
            assert float(row["--output-interval"]) == 0.25
            assert float(row["--backup-interval"]) == 0.25
        else:
            assert "--output-interval" not in row
            assert "--backup-interval" not in row
    assert float(values[-1]["--maximum-time-step"]) == 0.002


@pytest.mark.parametrize(
    "lean,output_interval,backup_interval,visualization,checkpoint,auto_write",
    [
        (True, None, None, 2.5, 10.0, False),
        (True, 0.25, 0.25, 0.25, 0.25, True),
        (False, None, None, 2.5, 2.5, True),
    ],
)
def test_visualization_and_restart_cadences_preserve_force_sampling(
    tmp_path,
    monkeypatch,
    lean,
    output_interval,
    backup_interval,
    visualization,
    checkpoint,
    auto_write,
):
    module = load("setup")
    captured = []
    monkeypatch.setattr(
        module.fvm,
        "create_fvm_solver",
        lambda config, **kw: captured.append(config) or SimpleNamespace(auto_write=True),
    )
    solver = module.create_solver(
        "xy_fine",
        0.04,
        output_root=tmp_path,
        span_layers=4,
        lean=lean,
        output_interval=output_interval,
        backup_interval=backup_interval,
    )
    config = captured[0]
    assert config.time.end_time == 100
    assert config.time.output_schedule.every_time == visualization
    assert config.backup.schedule.every_time == checkpoint
    assert config.backup.write_at_end
    assert solver.auto_write is auto_write
    assert config.time.adjustment.maximum_time_step_size == 0.004
    samplers = {sampler.file_name: sampler for sampler in config.samplers}
    assert samplers["forces_history"].schedule.every_time == 0.02
    for name in ("centreline", "span_probe", "transverse_x1", "transverse_x2", "transverse_x4"):
        assert samplers[name].schedule.every_time == 0.1
    if lean:
        assert "midspan" not in samplers
    else:
        assert samplers["midspan"].schedule.every_time == 0.5


def test_fine_output_override_keeps_numerical_restart_configuration(tmp_path, monkeypatch):
    from source.solvers.fvm.io.backup import _setup_dict

    module = load("setup")
    captured = []
    monkeypatch.setattr(
        module.fvm,
        "create_fvm_solver",
        lambda config, **kw: captured.append(config) or SimpleNamespace(auto_write=True),
    )
    for output in ({}, {"output_interval": 0.25, "backup_interval": 0.25}):
        module.create_solver(
            "xy_fine", 0.04, output_root=tmp_path, span_layers=4, lean=True, **output
        )
    assert _setup_dict(captured[0]) == _setup_dict(captured[1])
