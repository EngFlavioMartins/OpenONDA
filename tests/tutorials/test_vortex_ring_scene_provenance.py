"""Native observed termination and explicit headless screenshot ownership."""

import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tests._tutorial_helpers import tutorial_directory


@pytest.mark.parametrize("status", ["completed", "resolution_lost", "wall_time_limit"])
def test_terminal_lifecycle_is_preserved_in_supported_postprocessing(tmp_path, status):
    postprocess = __import__("tests.support.vpm.vortex_ring.postprocess", fromlist=["*"])
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


def test_headless_screenshot_uses_owned_layout_and_checks_publication(tmp_path, monkeypatch):
    path = tutorial_directory("vpm/vortex_ring") / "assets/render_vortex_ring.py"
    tree = ast.parse(path.read_text())
    # Execute the real setup/helper definitions, not the later expensive scenes.
    prefix = tree.body[: next(i for i, node in enumerate(tree.body) if isinstance(node, ast.For))]
    view, layout = SimpleNamespace(), object()
    assigned, screenshots = [], []

    def screenshot(output, owner, **kwargs):
        screenshots.append((owner, kwargs))
        Path(output).write_bytes(b"fixture PNG")
        return True

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
    namespace["save_screenshot"]("render.png", [80, 60])
    assert screenshots == [(layout, {"ImageResolution": [80, 60]})]
