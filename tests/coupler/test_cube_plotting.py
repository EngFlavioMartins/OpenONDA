"""Integrity checks for cube-flow comparison plotting."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

_PLOT_UTIL = Path(__file__).parents[2] / "tutorials/coupled_fvm_vpm/cube_flow/assets/_plotutil.py"
_PLOT_FIELDS = _PLOT_UTIL.with_name("plot_velocity_fields.py")
_ALLPLOT_SCRIPTS = (
    _PLOT_UTIL.with_name("plot_velocity_profiles.py"),
    _PLOT_FIELDS,
    _PLOT_UTIL.with_name("plot_coupling_diagnostics.py"),
)


@pytest.fixture(scope="module")
def plotutil():
    spec = importlib.util.spec_from_file_location("cube_plotutil_test", _PLOT_UTIL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fieldplot():
    spec = importlib.util.spec_from_file_location("cube_fieldplot_test", _PLOT_FIELDS)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_common_times_rejects_neighbouring_vpm_states(plotutil):
    result = plotutil.common_times(
        np.array([0.1, 0.2, 0.3]),
        np.array([0.09, 0.2, 0.31]),
    )
    np.testing.assert_allclose(result, [0.2])


def test_load_line_requires_the_requested_physical_time(plotutil, tmp_path, monkeypatch):
    path = tmp_path / "fvm_centreline.csv"
    path.write_text(
        "time,step,position_x,position_y,position_z,velocity_x,velocity_y,velocity_z,vorticity_x,vorticity_y,vorticity_z,kinematic_pressure\n0.09,9,0,0,0,1,0,0,0,0,0,0\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(plotutil.SOURCES["fvm"], "dir", tmp_path)

    assert plotutil.load_line("fvm", "centreline", 0.1) is None
    frame = plotutil.load_line("fvm", "centreline", 0.09)
    assert frame is not None
    assert frame["time"] == pytest.approx(0.09)


def test_field_plot_has_separate_fvm_and_reference_vpm_comparisons(fieldplot, monkeypatch):
    x, y = np.meshgrid(np.array([-1.5, 1.5]), np.array([-1.5, 1.5]), indexing="ij")
    fields = {
        source: {"x": x, "y": y, "velocity_x": np.ones_like(x) * scale}
        for source, scale in (("fvm", 1.0), ("vpm", 0.9), ("reference", 1.1))
    }
    monkeypatch.setattr(fieldplot.util, "load_slice", lambda source, time: fields[source])
    monkeypatch.setattr(
        fieldplot,
        "_on_grid",
        lambda source, target, key: np.asarray(source[key]),
    )
    figure_names = []

    def record_figure(*args, **kwargs):
        figure_names.append(args[7])
        return 0.0, 0.0, 0.0

    monkeypatch.setattr(fieldplot, "_field_figure", record_figure)
    fieldplot.plot_frame(
        0.1,
        {
            "freestream_speed": 1.0,
            "D": 1.0,
            "box": {"xmin": -1.5, "xmax": 1.5, "ymin": -1.5, "ymax": 1.5},
        },
    )

    assert figure_names == ["velocity_fields", "reference_vpm_fields"]


def test_metadata_is_required_for_selected_samples(plotutil, tmp_path, monkeypatch):
    monkeypatch.setattr(plotutil, "SOLUTION", tmp_path)
    monkeypatch.setattr(plotutil, "SAMPLES", tmp_path / "samples")
    with pytest.raises(FileNotFoundError, match="Refusing to infer"):
        plotutil.metadata()


def test_validation_uses_only_exact_common_samples(plotutil):
    result = plotutil._require_coincident_overlap(
        "Field",
        np.array([0.1, 0.2]),
        np.array([0.09, 0.2]),
    )

    np.testing.assert_allclose(result, [0.2])


def test_validation_accepts_source_specific_startup_prefix(plotutil):
    result = plotutil._require_coincident_overlap(
        "Profile",
        np.array([0.05, 0.10, 0.15]),
        np.array([0.10, 0.15]),
        np.array([0.05, 0.10, 0.15]),
    )

    np.testing.assert_allclose(result, [0.10, 0.15])


def test_publication_style_uses_fixed_width_serif_fonts_and_shared_exports(plotutil):
    assert pytest.approx(12.5) == plotutil.FIGURE_WIDTH_CM
    assert plotutil.figure_size(7.0) == pytest.approx((12.5 / 2.54, 7.0 / 2.54))
    assert plotutil.EXPORT_FORMATS == ("png", "pdf")
    assert mpl.rcParams["font.family"] == ["serif"]
    assert mpl.rcParams["font.serif"][0] == "DejaVu Serif"
    assert mpl.rcParams["mathtext.fontset"] == "dejavuserif"
    assert mpl.rcParams["font.size"] == pytest.approx(10.0)


def test_png_export_preserves_the_exact_physical_canvas(plotutil, tmp_path, monkeypatch):
    dpi = 100
    width, height = plotutil.figure_size(7.0)
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    monkeypatch.setattr(plotutil, "FIGURES", tmp_path)

    out = plotutil.save(fig, "canvas", "png", dpi)
    pixels = plt.imread(out)
    plt.close(fig)

    assert pixels.shape[1] == int(width * dpi)
    assert pixels.shape[0] == int(height * dpi)


@pytest.mark.parametrize("path", _ALLPLOT_SCRIPTS, ids=lambda path: path.stem)
def test_allplot_figures_expose_every_manual_layout_control(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "subplots_adjust"
    ]
    assert calls, f"{path.name} must use explicit manual layout"
    controls = {
        keyword.arg for call in calls for keyword in call.keywords if keyword.arg is not None
    }
    assert controls >= {"left", "right", "bottom", "top", "wspace", "hspace"}

    source = path.read_text(encoding="utf-8")
    assert "tight_layout(" not in source
    assert "constrained_layout=True" not in source
