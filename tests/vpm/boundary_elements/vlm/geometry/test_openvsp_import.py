import builtins
from pathlib import Path

import numpy as np
import pytest

from source.solvers.VPM.boundary_elements.vlm.config import VLMSetup, VLMSurfaceSetup
import source.solvers.VPM.boundary_elements.vlm.geometry.openvsp_io as openvsp_io
from source.solvers.VPM.boundary_elements.vlm.geometry.openvsp_io import (
    OpenVSPImportConfig,
    load_degengeom_csv,
    load_openvsp_surface,
)
from source.solvers.VPM.boundary_elements.vlm.geometry.surface_io import load_surface, save_surface
from source.solvers.VPM.boundary_elements.vlm.solver.vlm_solver import VLMSolver


def _write_minimal_degengeom_csv(path: Path) -> Path:
    chord_u = [0.0, 0.5, 1.0]
    span_w = [0.0, 0.5, 1.0]
    radii = [0.2, 0.5, 0.8]
    chords = [0.16, 0.12, 0.08]
    x_le = [0.0, 0.02, 0.04]
    z_te = [0.08, 0.04, 0.02]

    lines = [
        "DEGEN_GEOM,OpenONDA synthetic fixture",
        "GEOM,Test Rotor Blade,Type,rotor",
        "PLATE",
        "Num_Pnts,Num_U,Num_W",
        "9,3,3",
        "Name,Type,U,W,X,Y,Z",
    ]
    for u in chord_u:
        for station, w in enumerate(span_w):
            x = x_le[station] + u * chords[station]
            y = radii[station]
            z = u * z_te[station]
            lines.append(f"Test Rotor Blade,rotor,{u},{w},{x:.8f},{y:.8f},{z:.8f}")

    path.write_text("\n".join(lines) + "\n")
    return path


def test_parse_minimal_degengeom_csv_to_aircraft(tmp_path):
    csv_path = _write_minimal_degengeom_csv(tmp_path / "blade_degengeom.csv")

    aircraft = load_degengeom_csv(csv_path)

    assert aircraft.uid == "blade_degengeom"
    assert len(aircraft.wings) == 1
    wing = next(iter(aircraft.wings.values()))
    assert wing.uid == "Test_Rotor_Blade"
    assert len(wing.segments) == 2
    assert aircraft.total_num_panels() == 4
    for segment in wing.segments.values():
        assert segment.area > 0.0
        assert segment.panels_chord == 2
        assert segment.panels_span == 1
        np.testing.assert_allclose(segment.vertices["a"][1], segment.vertices["d"][1])
        np.testing.assert_allclose(segment.vertices["b"][1], segment.vertices["c"][1])


def test_parse_native_openvsp_plate_block(tmp_path):
    csv_path = tmp_path / "native_openvsp_degengeom.csv"
    csv_path.write_text(
        "\n".join(
            [
                "# DEGENERATE GEOMETRY CSV FILE",
                "# NUMBER OF COMPONENTS",
                "1",
                "# DegenGeom Type, Name, SurfNdx, GeomID, MainSurfNdx, SymCopyNdx, FlipNormal",
                "LIFTING_SURFACE,Native Blade,0,ABC,0,0,1",
                "# DegenGeom Type,nXsecs,nPnts/Xsec",
                "SURFACE_NODE,2,3",
                "# x,y,z,u,w",
                "0,0,0,1,0",
                "0,0,0,1,1",
                "0,0,0,1,2",
                "0,1,0,2,0",
                "0,1,0,2,1",
                "0,1,0,2,2",
                "SURFACE_FACE,1,1",
                "# nx,ny,nz,area",
                "0,0,1,1",
                "# DegenGeom Type,nXsecs,nPnts/Xsec",
                "PLATE,3,2",
                "# nx,ny,nz",
                "0,0,1",
                "0,0,1",
                "0,0,1",
                "# x,y,z,zCamber,t,nCamberx,nCambery,nCamberz,u,wTop,wBot,xxCamber,xyCamber,xzCamber",
                "0,0,0,0,0,0,0,1,1,1,0,0,0,0",
                "1,0,0,0,0,0,0,1,1,0,1,1,0,0",
                "0,0.5,0,0,0,0,0,1,1.5,1,0,0,0.5,0",
                "1,0.5,0,0,0,0,0,1,1.5,0,1,1,0.5,0",
                "0,1,0,0,0,0,0,1,2,1,0,0,1,0",
                "1,1,0,0,0,0,0,1,2,0,1,1,1,0",
                "# DegenGeom Type, nXsecs",
                "STICK_FACE,2",
            ]
        )
        + "\n"
    )

    aircraft = load_degengeom_csv(csv_path)

    assert len(aircraft.wings) == 1
    wing = next(iter(aircraft.wings.values()))
    assert wing.uid == "Native_Blade_0"
    assert len(wing.segments) == 2
    assert aircraft.total_num_panels() == 2
    first = next(iter(wing.segments.values()))
    np.testing.assert_allclose(first.vertices["a"], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(first.vertices["b"], [0.0, 0.5, 0.0])
    np.testing.assert_allclose(first.vertices["c"], [1.0, 0.5, 0.0])
    np.testing.assert_allclose(first.vertices["d"], [1.0, 0.0, 0.0])


def test_panel_count_override_and_json_round_trip(tmp_path):
    csv_path = _write_minimal_degengeom_csv(tmp_path / "blade_degengeom.csv")
    config = OpenVSPImportConfig(panels_chord=4, panels_span=6)

    aircraft = load_degengeom_csv(csv_path, config)
    json_path = save_surface(aircraft, str(tmp_path / "blade"))
    reloaded = load_surface(json_path)

    assert reloaded.total_num_panels() == aircraft.total_num_panels()
    assert reloaded.total_num_panels() == 24
    original_segment = next(iter(next(iter(aircraft.wings.values())).segments.values()))
    loaded_segment = next(iter(next(iter(reloaded.wings.values())).segments.values()))
    np.testing.assert_allclose(loaded_segment.vertices["c"], original_segment.vertices["c"])


def test_high_level_loader_saves_json(tmp_path):
    csv_path = _write_minimal_degengeom_csv(tmp_path / "blade_degengeom.csv")
    json_path = tmp_path / "generated" / "blade.json"

    aircraft = load_openvsp_surface(csv_path, save_json=json_path)

    assert aircraft.total_num_panels() == 4
    assert json_path.exists()
    assert load_surface(str(json_path)).total_num_panels() == 4


def test_vlm_solver_accepts_imported_aircraft(tmp_path):
    csv_path = _write_minimal_degengeom_csv(tmp_path / "blade_degengeom.csv")
    aircraft = load_degengeom_csv(csv_path)

    vlm = VLMSolver(
        VLMSetup(
            surfaces=(VLMSurfaceSetup(aircraft, name="openvsp_blade"),),
            max_panels=32,
            linear_solver="SCIPY",
        )
    )

    assert "openvsp_blade" in vlm.surfaces


def test_vsp3_import_without_openvsp_has_clear_error(monkeypatch, tmp_path):
    vsp3_path = tmp_path / "rotor_blade.vsp3"
    vsp3_path.write_text("placeholder")
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "openvsp":
            raise ImportError("openvsp intentionally unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    monkeypatch.setattr(openvsp_io, "_openvsp_python_command", lambda: None)

    with pytest.raises(ImportError, match="Direct .vsp3 import requires"):
        load_openvsp_surface(vsp3_path)
