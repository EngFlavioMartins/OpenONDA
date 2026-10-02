"""The flat-plate scene follows genuine accepted backups, not one frozen step."""

import json

import numpy as np
import pytest

from tests._tutorial_helpers import load_tutorial_module

render = load_tutorial_module("vpm/flat_plate", "assets.render_flat_plate")


def test_native_scene_accepts_two_archived_steps_and_fits_each_camera(tmp_path):
    directory = render.SOLUTION_DIR
    earlier = render.read_native_state(
        directory / "vpm/vpm_000160.h5", directory / "vlm/vlm_000160.vtp"
    )
    latest_input, latest_surface = render.native_inputs()
    latest = render.read_native_state(latest_input, latest_surface)
    assert (earlier["step"], latest["step"]) == (160, 197)
    assert (earlier["particle_count"], latest["particle_count"]) == (8960, 11032)
    assert earlier["panel_count"] == latest["panel_count"] == 224
    assert earlier["time"] < latest["time"]
    assert not np.array_equal(earlier["plate_bounds"], latest["plate_bounds"])
    for state in (earlier, latest):
        render.prepare_geometry(state, tmp_path)
        camera = render.camera_for_state(state)
        assert camera["projection"] == "perspective"
        assert 0 < camera["view_angle_degrees"] < 90
        assert np.isfinite(camera["position_m"]).all()
        assert np.isfinite(state["arrow_starts"]).all()
        assert len(state["arrow_starts"]) == 1
        # Both accepted native states fit the explicitly reserved render window.
        position = np.asarray(camera["position_m"])
        focal = np.asarray(camera["focal_point_m"])
        forward = focal - position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, camera["view_up"])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        points = np.vstack(
            (state["position"], state["plate_corners"], state["arrow_starts"], state["arrow_ends"])
        )
        relative = points - position
        depth = relative @ forward
        tangent = np.tan(np.radians(camera["view_angle_degrees"] / 2))
        assert np.all(depth > 0)
        aspect = camera["image_pixels"][0] / camera["image_pixels"][1]
        horizontal = 0.5 + relative @ right / (2 * aspect * depth * tangent)
        vertical = 0.5 + relative @ up / (2 * depth * tangent)
        assert np.min(horizontal) > 0.03
        assert np.max(horizontal) < 0.97
        assert np.min(vertical) > 0.24
        assert np.max(vertical) < 0.94
        assert "\\mathbf{e}_x" in render.plate_velocity_tex(state["kinematic_velocity"])
    assert earlier["arrow_starts"] != latest["arrow_starts"]
    assert (
        render.camera_for_state(earlier)["focal_point_m"]
        != render.camera_for_state(latest)["focal_point_m"]
    )


def test_native_scene_rejects_mismatched_surface():
    directory = render.SOLUTION_DIR
    with pytest.raises(ValueError, match="does not match"):
        render.read_native_state(directory / "vpm/vpm_000197.h5", directory / "vlm/vlm_000160.vtp")


def test_native_scene_rejects_empty_accepted_result(tmp_path, monkeypatch):
    (tmp_path / "vpm").mkdir()
    metadata = json.loads((render.SOLUTION_DIR / "vpm_metadata.json").read_text())
    (tmp_path / "vpm_metadata.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(render, "SOLUTION_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="No jointly published accepted"):
        render.native_inputs()


def test_native_scene_selects_latest_jointly_published_step(tmp_path, monkeypatch):
    source = render.SOLUTION_DIR
    (tmp_path / "vpm").mkdir()
    (tmp_path / "vlm").mkdir()
    (tmp_path / "vpm_metadata.json").symlink_to(source / "vpm_metadata.json")
    for step in (160, 197):
        (tmp_path / "vpm" / f"vpm_{step:06d}.h5").symlink_to(source / "vpm" / f"vpm_{step:06d}.h5")
    (tmp_path / "vlm/vlm_000160.vtp").symlink_to(source / "vlm/vlm_000160.vtp")
    monkeypatch.setattr(render, "SOLUTION_DIR", tmp_path)
    selected, surface = render.native_inputs()
    assert selected.name == "vpm_000160.h5"
    assert surface.name == "vlm_000160.vtp"
