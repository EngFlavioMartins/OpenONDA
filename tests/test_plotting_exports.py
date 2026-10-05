"""Figure export checks preserve the solver's numerical arrays."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from openonda import plotting


def test_dual_export_preserves_decimal_basename_and_canvas(tmp_path):
    plotting.set_style()
    fig, ax = plt.subplots(figsize=plotting.figure_size("single"))
    x = np.linspace(0, 1, 17)
    (line,) = ax.plot(x, x**2)
    plotting.centered_subplots_adjust(fig, outer=0.2, bottom=0.2, top=0.85)
    before = np.array(line.get_ydata(), copy=True)
    paths = plotting.export_figure(
        fig, tmp_path / "profile_t2.50", figure_format="both", close=False
    )
    assert [path.name for path in paths] == ["profile_t2.50.pdf", "profile_t2.50.png"]
    assert paths[0].read_bytes().startswith(b"%PDF")
    from PIL import Image

    with Image.open(paths[1]) as image:
        assert abs(image.width / image.info["dpi"][0] * 25.4 - 125) < 0.1
    np.testing.assert_array_equal(line.get_ydata(), before)
    assert fig.get_figwidth() * 25.4 == pytest.approx(125)
    plt.close(fig)


def test_single_export_does_not_create_other_format(tmp_path):
    plotting.set_style()
    fig, _ = plt.subplots(figsize=plotting.figure_size("single"))
    plotting.centered_subplots_adjust(fig, outer=0.2, bottom=0.2, top=0.85)
    paths = plotting.export_figure(fig, tmp_path / "curve.png", figure_format="pdf")
    assert paths == (tmp_path / "curve.pdf",)
    assert not (tmp_path / "curve.png").exists()
    with pytest.raises(ValueError, match="Unsupported"):
        plotting.requested_formats("svg")


def test_svg_canvas_export_keeps_backend_and_native_physical_layout(tmp_path):
    from matplotlib.backends.backend_svg import FigureCanvasSVG
    from matplotlib.figure import Figure
    from PIL import Image

    with plt.rc_context():
        plotting.set_style()
        figure = Figure(figsize=plotting.figure_size("wide_short"), dpi=120)
        canvas = FigureCanvasSVG(figure)
        axis = figure.subplots()
        axis.plot([0.0, 1.0], [0.0, 1.0])
        axis.set_xlabel("x / c")
        axis.set_ylabel("y / c")
        plotting.centered_subplots_adjust(figure, outer=0.2, bottom=0.25, top=0.85)
        plotting.prepare_figure(figure)
        plotting.fit_thesis_y_label_margins(figure, axis)
        bounds = axis.get_position().bounds
        dimensions = figure.get_size_inches().copy()
        outputs = plotting.export_figure(
            figure, tmp_path / "svg_canvas", figure_format="both", dpi=120, close=False
        )
        assert figure.canvas is canvas
        assert figure.dpi == 120
        np.testing.assert_array_equal(figure.get_size_inches(), dimensions)
        np.testing.assert_array_equal(axis.get_position().bounds, bounds)
        assert outputs[0].read_bytes().startswith(b"%PDF")
        with Image.open(outputs[1]) as raster:
            millimetres_per_pixel = 25.4 / raster.info["dpi"][0]
            assert abs(raster.width * millimetres_per_pixel - 125) < millimetres_per_pixel


def test_native_renderer_preserves_default_agg_pixels_and_canvas():
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(2.0, 1.0), dpi=100)
    canvas = FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.plot([0.0, 1.0], [0.0, 1.0], color="blue", linewidth=2.0)
    canvas.draw()
    original = np.asarray(canvas.buffer_rgba()).copy()
    renderer = plotting.figure_renderer(figure)
    assert figure.canvas is canvas
    assert figure.dpi == 100
    np.testing.assert_array_equal(np.asarray(renderer.buffer_rgba()), original)


def test_method_styles_distinguish_reference_and_results():
    for name in ("fvm", "vpm", "hybrid"):
        style = plotting.method_style(name)
        assert style["linestyle"] == "-"
        assert style["marker"]
        assert style["color"] != plotting.REFERENCE_GRAY
    for name in ("reference", "reference_secondary"):
        assert plotting.method_style(name)["linestyle"] in ("--", ":")
    modified = plotting.method_style("vpm")
    modified["color"] = "red"
    assert plotting.method_style("vpm")["color"] == "#41A6C4"
    assert "marker" not in plotting.method_style("vpm", markers=False)


def test_sequential_map_lightness_is_monotonic():
    rgb = matplotlib.colormaps[plotting.COLORMAPS["field_speed"]](np.linspace(0, 1, 256))[:, :3]
    luminance = rgb @ np.array([0.2126, 0.7152, 0.0722])
    assert np.all(np.diff(luminance) > 0)
