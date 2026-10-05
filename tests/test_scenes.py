"""Scene rendering preserves configured geometry and removes temporary files."""

from math import erf, exp, pi, sqrt
import sys
from xml.etree import ElementTree

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure
import numpy as np
from PIL import Image
import pytest

from openonda import scenes


def test_axial_gaussian_field_matches_one_particle_analytic_values():
    points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    position = np.zeros((1, 3))
    strength = np.array([[0.0, 0.0, 2.0]])
    sigma = np.ones(1)
    factor = 2 * (erf(1) - 2 / sqrt(pi) * exp(-1)) / (4 * pi)
    velocity = scenes.sample_axial_velocity(points, position, strength, sigma)
    np.testing.assert_allclose(velocity, [[0, factor], [-factor, 0]], atol=1e-14)
    np.testing.assert_allclose(
        scenes.sample_axial_vorticity(points, position, strength, sigma, cutoff=5),
        2 * exp(-1) / pi**1.5,
    )


def test_sphere_depth_and_streamline_depth_preserve_front_surface_and_light():
    position = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    light = np.array([-0.45, 0.65, 0.65])
    original = light.copy()
    canvas, depth = scenes.render_spheres(
        position,
        np.full(2, 0.5),
        np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        np.array([-1.0, -1.0]),
        np.array([1.0, 1.0]),
        21,
        21,
        light,
    )
    np.testing.assert_array_equal(light, original)
    assert depth[10, 10] == 1.5
    assert canvas[10, 10, 2] > canvas[10, 10, 0]
    before = canvas.copy()
    result = scenes.render_streamlines(
        canvas,
        depth,
        np.array([[[-0.2, 0.0, -1.0], [0.2, 0.0, -1.0]]]),
        np.array([[0.0, 1.0, 0.0]]),
        np.array([-1.0, -1.0]),
        np.array([1.0, 1.0]),
        1.0,
    )
    np.testing.assert_array_equal(result, before)


def test_scene_export_saves_geometry_and_updates_state_paths_before_removing_temporary_files(
    tmp_path, monkeypatch
):
    renderer = tmp_path / "renderer.py"
    renderer.write_text(
        "from pathlib import Path\nimport sys\n"
        "work = Path(sys.argv[1])\n"
        "(work / 'render.png').write_bytes(b'rendered')\n"
        "(work / 'scene.pvsm').write_text('<ServerManagerState><Element value=\"' + str(work / 'particles.vtp') + '\"/></ServerManagerState>')\n"
    )
    monkeypatch.setenv("OPENONDA_PARAVIEW_PYTHON", sys.executable)
    builds = []

    def prepare(work):
        builds.append(work)
        (work / "particles.vtp").write_bytes(b"geometry")
        return {
            "renderer": (renderer, [work]),
            "images": {"render.png": "physical_scene.png"},
            "files": {
                "particles.vtp": "geometry/particles.vtp",
                "scene.pvsm": "geometry/scene.pvsm",
            },
            "documents": [],
            "format": "both",
            "metadata": {"scene.json": {"camera": [4, -5, 1]}},
        }

    output = tmp_path / "output with spaces & data"
    scenes.export_scene(output, prepare, dpi=100)
    assert (output / "physical_scene.png").read_bytes() == b"rendered"
    assert (output / "geometry/particles.vtp").read_bytes() == b"geometry"
    assert ElementTree.parse(output / "geometry/scene.pvsm").find("Element").attrib["value"] == str(
        output / "geometry/particles.vtp"
    )
    assert (output / "scene.json").is_file()
    assert not builds[0].exists()


def test_scene_build_failure_removes_temporary_files(tmp_path):
    builds = []

    def prepare(work):
        builds.append(work)
        (work / "partial.vtp").write_bytes(b"partial")
        raise ValueError("physical fixture failure")

    with pytest.raises(ValueError, match="physical fixture"):
        scenes.export_scene(tmp_path / "figures", prepare, dpi=100)
    assert not builds[0].exists()


def test_gif_clock_rounding_preserves_average_thirty_fps(tmp_path):
    frames = [Image.new("RGB", (2, 2), (index * 7, 0, 0)) for index in range(30)]
    output = scenes.export_animation(frames, tmp_path / "nested/movie.gif", fps=30)
    with Image.open(output) as movie:
        durations = []
        for index in range(movie.n_frames):
            movie.seek(index)
            durations.append(movie.info["duration"])
    assert set(durations) == {30, 40}
    assert sum(durations) == 1000


def test_figure_frame_preserves_selected_canvas_and_identical_authored_pixels():
    frames = []
    for canvas_type in (FigureCanvasAgg, FigureCanvasSVG):
        figure = Figure(figsize=(2.0, 1.0), dpi=100)
        canvas = canvas_type(figure)
        axis = figure.add_axes([0.1, 0.1, 0.8, 0.8])
        axis.plot([0.0, 1.0], [0.0, 1.0], color="red", linewidth=3.0)
        axis.set_axis_off()
        frame = scenes.figure_frame(figure).convert("RGB")
        assert figure.canvas is canvas
        assert figure.dpi == 100
        assert frame.size == (200, 100)
        pixels = np.asarray(frame)
        assert np.count_nonzero((pixels[..., 0] > 200) & (pixels[..., 1] < 50)) > 200
        np.testing.assert_array_equal(pixels[0, 0], [255, 255, 255])
        frames.append(pixels)
    np.testing.assert_array_equal(frames[0], frames[1])
