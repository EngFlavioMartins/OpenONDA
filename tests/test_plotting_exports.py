"""Publication export contracts independent of any solver's numerical arrays."""

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
    fig, _ = plt.subplots()
    paths = plotting.export_figure(fig, tmp_path / "curve.png", figure_format="pdf")
    assert paths == (tmp_path / "curve.pdf",)
    assert not (tmp_path / "curve.png").exists()
    with pytest.raises(ValueError, match="Unsupported"):
        plotting.requested_formats("svg")


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
