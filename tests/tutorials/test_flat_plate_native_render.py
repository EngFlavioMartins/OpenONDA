"""The authored flat-plate camera fits its particle, surface and motion geometry."""

import numpy as np

from tests._tutorial_helpers import load_tutorial_module

render = load_tutorial_module("vpm/flat_plate", "assets.render_flat_plate")


def test_camera_fits_plate_particles_and_motion_arrow(tmp_path):
    x, y, z = np.meshgrid(np.linspace(0, 3, 6), [-2, 2], [-0.3, 0.3])
    position = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    state = {
        "position": position,
        "omega": np.linspace(0.2, 3, len(position)),
        "plate_bounds": np.array([[0, -2, 0], [1, 2, 0]]),
        "plate_corners": np.array([[0, -2, 0], [1, -2, 0], [1, 2, 0], [0, 2, 0]]),
        "kinematic_velocity": np.array([-10.0, 0, 0]),
    }
    render.prepare_geometry(state, tmp_path)
    camera = render.camera_for_state(state)
    position = np.asarray(camera["position_m"])
    forward = np.asarray(camera["focal_point_m"]) - position
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
    aspect = camera["image_pixels"][0] / camera["image_pixels"][1]
    horizontal = 0.5 + relative @ right / (2 * aspect * depth * tangent)
    vertical = 0.5 + relative @ up / (2 * depth * tangent)
    assert np.all(depth > 0)
    assert horizontal.min() > 0.03 and horizontal.max() < 0.97
    assert vertical.min() > 0.24 and vertical.max() < 0.94
    assert "\\mathbf{e}_x" in render.plate_velocity_tex(state["kinematic_velocity"])
