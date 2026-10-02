"""Saved-run initial geometry and explicit headless screenshot ownership."""

import ast
import hashlib
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


def restored_fixture(scene, root):
    """Model a released plotting bundle with metadata, but no t0 arrays."""
    initial = root / "fixture-initial.h5"
    cloud = initial_fixture(initial)
    declared = scene.initial_configuration()
    configuration = {
        "initial_conditions": declared["initial_conditions"],
        "initial_weak_particle_percent": declared["initial_weak_particle_percent"],
        "numerics": {key: declared[key] for key in ("precision", "random_seed", "particle_kernel")},
    }
    metadata = root / "solution/les_transposed/vpm_metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps({"configuration": configuration}))
    record = {
        "path": metadata.relative_to(root).as_posix(),
        "size": metadata.stat().st_size,
        "sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
    }
    assets = root / "assets"
    (assets / "results").mkdir(parents=True)
    (assets / "results/manifest.json").write_text(json.dumps({"files": [record]}))
    fingerprint = {
        "schema_version": 1,
        "metadata": record,
        "clock": {"step": 0, "time": 0.0},
        "particle_count": 2,
        "initial_configuration": declared,
        "fields": {},
    }
    with h5py.File(initial) as saved:
        for name in ("position", "vortex_strength", "core_radius"):
            array = saved[f"particles/{name}"][:]
            fingerprint["fields"][name] = {
                "dtype": array.dtype.str, "shape": list(array.shape),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
    sidecar = assets / "initial_state_fingerprint.json"
    sidecar.write_text(json.dumps(fingerprint))
    return cloud, sidecar, metadata


def test_legacy_bundle_without_t0_verifies_recorded_fields(scene, tmp_path):
    cloud, sidecar, metadata = restored_fixture(scene, tmp_path)
    before = (sidecar.read_bytes(), metadata.read_bytes())
    assert scene.validate_scene_initial_cloud(cloud, tmp_path) == (
        "recorded_field_fingerprint_verified_reconstruction"
    )
    assert (sidecar.read_bytes(), metadata.read_bytes()) == before


@pytest.mark.parametrize("field", ["position", "vortex_strength", "core_radius"])
def test_legacy_same_count_field_change_rejected(scene, tmp_path, field):
    cloud, _, _ = restored_fixture(scene, tmp_path)
    getattr(cloud, field).flat[-1] += 0.01
    with pytest.raises(ValueError, match="recorded field fingerprint"):
        scene.validate_scene_initial_cloud(cloud, tmp_path)


@pytest.mark.parametrize("change", ["metadata", "recipe", "schema", "clock", "dtype", "shape", "digest", "count", "missing_field"])
def test_legacy_identity_corruption_rejected(scene, tmp_path, change):
    cloud, sidecar, metadata = restored_fixture(scene, tmp_path)
    fingerprint = json.loads(sidecar.read_text())
    if change == "metadata":
        metadata.write_text(metadata.read_text() + "\n")
    elif change == "recipe":
        contents = json.loads(metadata.read_text())
        contents["configuration"]["initial_conditions"][0]["disturbance"]["seed"] += 1
        metadata.write_text(json.dumps(contents))
    elif change in ("schema", "clock", "count"):
        key = {"schema": "schema_version", "clock": "clock", "count": "particle_count"}[change]
        fingerprint[key] = {"step": 1, "time": 0.0} if change == "clock" else 3
    elif change == "missing_field":
        del fingerprint["fields"]["core_radius"]
    else:
        key = "sha256" if change == "digest" else change
        fingerprint["fields"]["position"][key] = {"dtype": "int32", "shape": [6], "sha256": "0" * 64}[key]
    sidecar.write_text(json.dumps(fingerprint))
    with pytest.raises(ValueError):
        scene.validate_scene_initial_cloud(cloud, tmp_path)


def test_existing_t0_is_preferred_over_sidecar(scene, tmp_path):
    cloud, sidecar, _ = restored_fixture(scene, tmp_path)
    sidecar.write_text("invalid sidecar, deliberately unused")
    initial = tmp_path / "solution/les_transposed/vpm/vpm_000000.h5"
    initial.parent.mkdir()
    initial_fixture(initial)
    assert scene.validate_scene_initial_cloud(cloud, tmp_path) == "saved_step_zero_verified"
    with h5py.File(initial, "r+") as saved:
        saved["solver"].attrs["time"] = 1
    with pytest.raises(ValueError, match="not the step-zero"):
        scene.validate_scene_initial_cloud(cloud, tmp_path)


def test_unknown_legacy_run_cannot_reconstruct_without_fingerprint(scene, tmp_path):
    cloud, sidecar, _ = restored_fixture(scene, tmp_path)
    sidecar.unlink()
    with pytest.raises(FileNotFoundError, match="recorded fingerprint"):
        scene.validate_scene_initial_cloud(cloud, tmp_path)


def test_initial_fingerprint_is_copied_with_normal_tutorial_resources(tmp_path):
    from openonda.tutorials import materialize_tutorial

    output = materialize_tutorial("vpm/vortex_ring", tmp_path)
    relative = "assets/initial_state_fingerprint.json"
    assert (output / relative).read_bytes() == (tutorial_directory("vpm/vortex_ring") / relative).read_bytes()


def test_completed_lifecycle_is_not_mislabeled_running():
    stability = load_tutorial_module("vpm/vortex_ring", "assets.plot_vortex_ring_stability")
    assert stability.STATUS_MARKERS["completed"] == ">"
    assert stability.STATUS_MARKERS["resolution_lost"] == "X"


@pytest.mark.parametrize("returned,writes,success", [(True, True, True), (False, True, False), (True, False, False)])
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
    for name in ("XMLPolyDataReader", "Glyph", "Show", "Hide", "ColorBy", "GetColorTransferFunction"):
        setattr(simple, name, lambda *a, **kw: None)
    monkeypatch.setitem(sys.modules, "paraview", SimpleNamespace(simple=simple))
    monkeypatch.setitem(sys.modules, "paraview.simple", simple)
    monkeypatch.setattr(sys, "argv", [str(path), str(tmp_path), "[]"])
    namespace = {}
    exec(compile(ast.Module(prefix, type_ignores=[]), str(path), "exec"), namespace)
    assert assigned == [{"view": view, "layout": layout}]
    if success:
        namespace["save_screenshot"]("render.png", [80, 60])
    else:
        with pytest.raises(RuntimeError, match="failed to save screenshot"):
            namespace["save_screenshot"]("render.png", [80, 60])
    assert screenshots == [(layout, {"ImageResolution": [80, 60]})]
