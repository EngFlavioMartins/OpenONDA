"""Saved-run initial geometry and explicit headless screenshot ownership."""

import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from tests._tutorial_helpers import load_tutorial_module, tutorial_directory


@pytest.fixture
def scene():
    return load_tutorial_module("vpm/vortex_ring", "assets.plot_vortex_ring_scenes")


def initial_fixture(path, *, precision=np.float32, step=0, time=0.0):
    cloud = SimpleNamespace(
        position=np.arange(6, dtype=np.float64).reshape(2, 3) / 7,
        vortex_strength=np.array([[0, 1 / 3, -1 / 7], [0, -1 / 3, 1 / 7]]),
        core_radius=np.full(2, 0.07),
    )
    with h5py.File(path, "w") as initial:
        initial.create_group("solver").attrs.update(step=step, time=time, n_particles_total=2)
        particles = initial.create_group("particles")
        for name in ("position", "vortex_strength", "core_radius"):
            particles[name] = getattr(cloud, name).astype(precision)
    return cloud


@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_saved_solver_precision_matches_reconstructed_cloud(scene, tmp_path, precision):
    path = tmp_path / "initial.h5"
    cloud = initial_fixture(path, precision=precision)
    before = path.read_bytes()
    scene.validate_initial_cloud(cloud, path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("field", ["position", "vortex_strength", "core_radius"])
def test_same_count_changed_setup_is_rejected(scene, tmp_path, field):
    path = tmp_path / "initial.h5"
    cloud = initial_fixture(path)
    getattr(cloud, field).flat[-1] += 0.01
    with pytest.raises(ValueError, match=field + " differs from the saved step-zero"):
        scene.validate_initial_cloud(cloud, path)


@pytest.mark.parametrize("step,time", [(1, 0.0), (0, 1e-12), (0, float("nan"))])
def test_nonzero_or_invalid_initial_clock_is_rejected(scene, tmp_path, step, time):
    path = tmp_path / "initial.h5"
    cloud = initial_fixture(path, step=step, time=time)
    with pytest.raises(ValueError, match="not the step-zero state"):
        scene.validate_initial_cloud(cloud, path)


def test_missing_initial_backup_is_not_silently_reconstructed(scene, tmp_path):
    with pytest.raises(FileNotFoundError, match="Saved step-zero ring backup required"):
        scene.validate_initial_cloud(SimpleNamespace(), tmp_path / "missing.h5")


@pytest.mark.parametrize("corruption", ["count", "shape", "nan", "core", "dtype"])
def test_invalid_saved_initial_fields_are_rejected(scene, tmp_path, corruption):
    path = tmp_path / "initial.h5"
    cloud = initial_fixture(path)
    with h5py.File(path, "r+") as initial:
        if corruption == "count":
            initial["solver"].attrs["n_particles_total"] = 3
        elif corruption in ("shape", "dtype"):
            del initial["particles/position"]
            initial["particles/position"] = np.zeros(
                (2, 2) if corruption == "shape" else (2, 3),
                dtype=np.float32 if corruption == "shape" else np.int32,
            )
        else:
            initial["particles/core_radius"][0] = np.nan if corruption == "nan" else -1
    with pytest.raises(ValueError):
        scene.validate_initial_cloud(cloud, path)


def test_scene_requires_saved_step_zero_and_matching_configuration(scene, tmp_path):
    declared = scene.initial_configuration()
    configuration = {
        "initial_conditions": declared["initial_conditions"],
        "initial_weak_particle_percent": declared["initial_weak_particle_percent"],
        "numerics": {key: declared[key] for key in ("precision", "random_seed", "particle_kernel")},
    }
    directory = tmp_path / "solution/les_transposed"
    (directory / "vpm").mkdir(parents=True)
    metadata = directory / "vpm_metadata.json"
    metadata.write_text(json.dumps({"configuration": configuration}))
    cloud = initial_fixture(directory / "vpm/vpm_000000.h5")
    assert scene.validate_scene_initial_cloud(cloud, tmp_path) == "saved_step_zero_verified"
    (directory / "vpm/vpm_000000.h5").unlink()
    with pytest.raises(FileNotFoundError, match="Saved step-zero ring backup required"):
        scene.validate_scene_initial_cloud(cloud, tmp_path)
    configuration["initial_weak_particle_percent"] += 1
    metadata.write_text(json.dumps({"configuration": configuration}))
    with pytest.raises(ValueError, match="configuration differs"):
        scene.validate_scene_initial_cloud(cloud, tmp_path)


@pytest.mark.parametrize("status", ["completed", "resolution_lost", "wall_time_limit"])
def test_terminal_lifecycle_is_preserved_in_supported_postprocessing(tmp_path, status):
    postprocess = load_tutorial_module("vpm/vortex_ring", "assets.postprocess")
    variant = "les_transposed"
    metadata = tmp_path / "solution" / variant / "vpm_metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "solver": "VPM",
                "case_name": variant,
                "configuration": {
                    "numerics": {"induction": {"stretching_scheme": "TRANSPOSED"}},
                    "run": {"steps": 10 if status == "completed" else 20},
                },
                "lifecycle": {"status": status},
                "state": {"step": 10, "time": 0.2},
            }
        )
    )
    samples = tmp_path / "samples"
    (samples / variant).mkdir(parents=True)
    (samples / variant / "ring_diagnostics.csv").write_text("time,step\n0.1,5\n")

    summary = postprocess.build_summary(samples, tmp_path / "figures")

    assert summary["runs"][variant]["status"] == status
    assert summary["runs"][variant]["completed_steps"] == 10
    assert summary["runs"][variant]["completed_time"] == pytest.approx(0.2)
    assert summary["longest_sustained_variant"] == variant


@pytest.mark.parametrize(
    "returned,writes,success", [(True, True, True), (False, True, False), (True, False, False)]
)
def test_headless_screenshot_uses_owned_layout_and_checks_publication(
    tmp_path, monkeypatch, returned, writes, success
):
    path = tutorial_directory("vpm/vortex_ring") / "assets/render_vortex_ring.py"
    tree = ast.parse(path.read_text())
    # Execute the real setup/helper definitions, not the later expensive scenes.
    prefix = tree.body[: next(i for i, node in enumerate(tree.body) if isinstance(node, ast.For))]
    view, layout = SimpleNamespace(), object()
    assigned, screenshots = [], []

    def screenshot(output, owner, **kwargs):
        screenshots.append((owner, kwargs))
        if writes:
            Path(output).write_bytes(b"fixture PNG")
        return returned

    simple = SimpleNamespace(
        CreateView=lambda kind: view,
        CreateLayout=lambda name: layout,
        AssignViewToLayout=lambda **kw: assigned.append(kw),
        SaveScreenshot=screenshot,
    )
    for name in (
        "XMLPolyDataReader",
        "Glyph",
        "Show",
        "Hide",
        "ColorBy",
        "GetColorTransferFunction",
    ):
        setattr(simple, name, lambda *a, **kw: None)
    monkeypatch.setitem(sys.modules, "paraview", SimpleNamespace(simple=simple))
    monkeypatch.setitem(sys.modules, "paraview.simple", simple)
    monkeypatch.setattr(sys, "argv", [str(path), str(tmp_path), "[]"])
    # The renderer now takes its common snapshot camera from the scene record.
    scene = {
        "camera_position": [1.0, -2.0, 0.5],
        "camera_focal_point": [0.0, 0.0, 0.1],
        "parallel_scale": 1.25,
        "snapshot_view_size_px": [2500, 1600],
    }
    (tmp_path / "scene.json").write_text(json.dumps(scene))
    namespace = {}
    exec(compile(ast.Module(prefix, type_ignores=[]), str(path), "exec"), namespace)
    assert view.CameraPosition == scene["camera_position"]
    assert view.CameraFocalPoint == scene["camera_focal_point"]
    assert view.CameraParallelScale == scene["parallel_scale"]
    assert view.ViewSize == scene["snapshot_view_size_px"]
    assert assigned == [{"view": view, "layout": layout}]
    if success:
        namespace["save_screenshot"]("render.png", [80, 60])
    else:
        with pytest.raises(RuntimeError, match="failed to save screenshot"):
            namespace["save_screenshot"]("render.png", [80, 60])
    assert screenshots == [(layout, {"ImageResolution": [80, 60]})]
