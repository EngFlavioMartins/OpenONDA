"""
Optional tests for OpenVSP DegenGeom import/export.

These tests are marked ``@pytest.mark.openvsp`` and are automatically
skipped when the OpenVSP Python API is not installed.  Run with::

    conda activate openonda-openvsp
    pytest tests/openvsp/ -v -m openvsp

Tests that parse ``.csv`` DegenGeom files do **not** require OpenVSP.
"""

from pathlib import Path

import pytest

from source.solvers.VPM.boundary_elements.vlm.geometry.openvsp_io import (
    OpenVSPImportConfig,
    _import_openvsp,
    load_degengeom_csv,
    load_openvsp_surface,
)

pytestmark = pytest.mark.openvsp

HERE = Path(__file__).resolve().parent


def _openvsp_available() -> bool:
    try:
        _import_openvsp()
        return True
    except ImportError:
        return False


def _require_openvsp():
    if not _openvsp_available():
        pytest.skip("OpenVSP Python API not available")
    return _import_openvsp()


class TestOpenVSPImport:
    def test_import_openvsp(self):
        vsp = _require_openvsp()

        version = vsp.GetVSPVersion()
        assert isinstance(version, str)
        assert len(version) > 0

    def test_get_version_string(self):
        vsp = _require_openvsp()

        version = vsp.GetVSPVersion()
        assert "OpenVSP" in version


class TestLoadDegenGeomCSV:
    @pytest.fixture
    def sample_csv(self, tmp_path: Path) -> Path:
        csv_file = tmp_path / "test_wing.csv"
        csv_file.write_text(
            "geom,test_wing,wing\n"
            "lifting_surface\n"
            "num_u,num_w\n"
            "2,3\n"
            "X,Y,Z\n"
            "0,0,0\n"
            "1,0,0\n"
            "0,1,0\n"
            "1,1,0\n"
            "0,0,1\n"
            "1,0,1\n"
        )
        return csv_file

    def test_load_degengeom_csv_minimal(self, sample_csv: Path):
        aircraft = load_degengeom_csv(sample_csv)
        assert aircraft is not None
        assert len(aircraft.wings) == 1

    def test_load_degengeom_csv_with_config(self, sample_csv: Path):
        config = OpenVSPImportConfig(length_scale=1.0)
        aircraft = load_degengeom_csv(sample_csv, config)
        assert aircraft is not None
        wing = aircraft.wings[list(aircraft.wings.keys())[0]]
        assert len(wing.segments) >= 1

    def test_load_degengeom_csv_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            load_degengeom_csv("/nonexistent/path.csv")

    def test_load_invalid_csv(self, tmp_path: Path):
        bad_file = tmp_path / "bad.csv"
        bad_file.write_text("not,a,degen,geom,file\n")
        with pytest.raises(ValueError, match="No lifting-surface point tables"):
            load_degengeom_csv(bad_file)


class TestLoadOpenVspSurfaceCSV:
    def test_load_openvsp_surface_csv(self, tmp_path: Path):
        csv_file = tmp_path / "test_surface.csv"
        csv_file.write_text(
            "geom,test_surface,wing\n"
            "lifting_surface\n"
            "num_u,num_w\n"
            "2,2\n"
            "X,Y,Z\n"
            "0,0,0\n"
            "1,0,0\n"
            "0,1,0\n"
            "1,1,0\n"
        )
        aircraft = load_openvsp_surface(csv_file)
        assert aircraft is not None
        assert len(aircraft.wings) == 1


class TestOpenVSPImportConfig:
    def test_default_config(self):
        cfg = OpenVSPImportConfig()
        assert cfg.set_id == "ALL"
        assert cfg.length_scale == 1.0
        assert cfg.preserve_vsp_paneling is True
        assert cfg.target_surface_types == ("wing", "prop", "rotor")

    def test_custom_config(self):
        cfg = OpenVSPImportConfig(
            set_id=1,
            length_scale=0.0254,
            target_surface_types=("wing",),
            preserve_vsp_paneling=False,
        )
        assert cfg.set_id == 1
        assert cfg.length_scale == 0.0254
        assert cfg.target_surface_types == ("wing",)
        assert cfg.preserve_vsp_paneling is False

    def test_with_include_components(self):
        cfg = OpenVSPImportConfig(
            include_components=["blade", "rotor"],
            exclude_components=["hub"],
        )
        assert cfg.include_components == ["blade", "rotor"]
        assert cfg.exclude_components == ["hub"]


@pytest.mark.skipif(not _openvsp_available(), reason="OpenVSP Python API not available")
class TestOpenVspVSP3:
    @pytest.fixture
    def simple_vsp3(self, tmp_path: Path) -> Path:
        vsp = _require_openvsp()

        vsp.ClearVSPModel()
        wing_id = vsp.AddGeom("WING")
        vsp.SetParmVal(wing_id, "TotalSpan", "WingGeom", 2.0)
        vsp.SetParmVal(wing_id, "Root_Chord", "WingGeom", 0.5)
        vsp.SetParmVal(wing_id, "Tip_Chord", "WingGeom", 0.25)
        vsp.Update()
        vsp_file = tmp_path / "test_wing.vsp3"
        vsp.WriteVSPFile(str(vsp_file))
        return vsp_file

    def test_export_degengeom_csv(self, simple_vsp3: Path, tmp_path: Path):
        from source.solvers.VPM.boundary_elements.vlm.geometry.openvsp_io import (
            export_openvsp_degengeom,
        )

        csv_path = tmp_path / "test_wing_degen.csv"
        result = export_openvsp_degengeom(simple_vsp3, csv_path)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_load_openvsp_surface_vsp3(self, simple_vsp3: Path):
        aircraft = load_openvsp_surface(simple_vsp3)
        assert aircraft is not None
        assert len(aircraft.wings) >= 1

    def test_vsp3_to_json_roundtrip(self, simple_vsp3: Path, tmp_path: Path):
        json_path = tmp_path / "roundtrip.json"
        aircraft = load_openvsp_surface(simple_vsp3, save_json=json_path)
        assert json_path.exists()
        assert len(aircraft.wings) >= 1


class TestLazyImport:
    def test_import_error_message(self):
        with pytest.raises(ImportError) as excinfo:
            import builtins

            real_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "openvsp":
                    raise ImportError("Mock: openvsp not available")
                return real_import(name, *args, **kwargs)

            builtins.__import__ = mock_import
            try:
                _import_openvsp()
            finally:
                builtins.__import__ = real_import

        assert "DegenGeom CSV" in str(excinfo.value)
