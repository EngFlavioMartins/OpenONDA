"""Mesh-only delivery must serialize evidence and fail closed on bad oracles."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tutorials.coupled_fvm_vpm.cylinder_shedding_flow.reference_flow import mesh, native


def test_mesh_evidence_serializes_paths_arrays_and_scalars():
    assert mesh._jsonable(
        {"path": Path("wall.stl"), "counts": np.array([1, 2]), "n": np.int64(3)}
    ) == {
        "path": "wall.stl",
        "counts": [1, 2],
        "n": 3,
    }


def test_existing_mesh_is_never_rebuilt_or_overwritten(tmp_path, monkeypatch):
    destination = tmp_path / "coarse"
    destination.mkdir()
    sentinel = destination / "previous.txt"
    sentinel.write_text("preserve")
    monkeypatch.setattr(mesh, "grid_mesh", lambda dx: pytest.fail("must reject before building"))
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        mesh._build("coarse", destination)
    assert sentinel.read_text() == "preserve"


def test_mesh_publication_enforces_solver_conditioning_limit(tmp_path, monkeypatch):
    fake_mesh = {"n_cells": 1, "n_faces": 1}
    monkeypatch.setattr(mesh, "prepare_canonical_surfaces", lambda *_, **__: {})
    monkeypatch.setattr(
        mesh,
        "grid_mesh",
        lambda dx, **_: SimpleNamespace(
            build=lambda **kwargs: (
                kwargs["on_generated"](fake_mesh),
                fake_mesh,
            )[1]
        ),
    )
    monkeypatch.setattr(mesh, "validate_topology", lambda *_: {})
    backed_up = []
    monkeypatch.setattr(mesh, "_save_generated_mesh", lambda *args: backed_up.append(args[1]))
    requested_schemes = []

    def geometry(_mesh, **kwargs):
        requested_schemes.append(kwargs["gradient_scheme"])
        return {}

    monkeypatch.setattr(mesh, "compute_mesh_geometry", geometry)
    monkeypatch.setattr(mesh, "validate_geometry", lambda *_: {"max_lsq_condition": 25.0001})
    monkeypatch.setattr(mesh, "validate_cell_area_closure", lambda *_: {})
    monkeypatch.setattr(mesh, "validate_single_fluid_component", lambda *_: {})
    destination = tmp_path / "coarse"
    with pytest.raises(ValueError, match="max_lsq_condition"):
        mesh._build("coarse", destination)
    assert requested_schemes == ["lsq"]
    assert not destination.exists()
    assert backed_up == [
        tmp_path / "coarse-generated",
        tmp_path / "coarse-generated" / "constrained",
    ]
    assert "generated_not_qualified" in (backed_up[0] / "mesh_backup.json").read_text()


def test_rejected_post_generation_mesh_is_still_backed_up(tmp_path, monkeypatch):
    fake_mesh = {"n_cells": 1, "n_faces": 1}
    monkeypatch.setattr(mesh, "prepare_canonical_surfaces", lambda *_, **__: {})

    def build(*, on_generated):
        on_generated(fake_mesh)
        raise ValueError("projection rejected")

    monkeypatch.setattr(mesh, "grid_mesh", lambda dx, **_: SimpleNamespace(build=build))
    backed_up = []
    monkeypatch.setattr(mesh, "_save_generated_mesh", lambda *args: backed_up.append(args[1]))
    destination = tmp_path / "coarse"
    with pytest.raises(ValueError, match="projection rejected"):
        mesh._build("coarse", destination)
    assert backed_up == [tmp_path / "coarse-generated"]
    assert not destination.exists()


@pytest.mark.parametrize(
    "check_output, code", [("Failed 1 mesh checks.\n", 0), ("missing executable\n", 127)]
)
def test_native_oracle_rejects_quality_failure_even_after_mesher_success(
    tmp_path, monkeypatch, check_output, code
):
    executable = tmp_path / "cartesianMesh"
    executable.write_text("fake pinned executable")
    launcher = tmp_path / "launcher"
    launcher.write_text("fake launcher")
    checkmesh = tmp_path / "checkMesh"
    checkmesh.write_text("fake checkMesh")
    monkeypatch.setattr(native, "EXECUTABLE", executable)
    monkeypatch.setattr(native, "LAUNCHER", launcher)
    monkeypatch.setattr(native, "CHECKMESH", checkmesh)
    monkeypatch.setattr(native, "prepare_canonical_surfaces", lambda *_, **__: {})
    case = tmp_path / "case"

    def run(command, **kwargs):
        if command[0] == "otool":
            return SimpleNamespace(stdout="fake libraries")
        if str(checkmesh) in command:
            assert (case / "system/fvSchemes").is_file()
            assert (case / "system/fvSolution").is_file()
            kwargs["stdout"].write(check_output)
            return SimpleNamespace(returncode=code)
        poly = case / "constant/polyMesh"
        poly.mkdir()
        (poly / "points").write_text("fake mesh")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(native.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="quality was not accepted"):
        native._run_one(case, repeat=1)
    assert (case / "native_manifest.json").is_file()
