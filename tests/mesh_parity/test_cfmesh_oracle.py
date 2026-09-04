"""cfMesh case rendering and reproducibility contracts without cfMesh installed."""

from __future__ import annotations

import json
from pathlib import Path

from tools.mesh_parity.cfmesh_oracle import (
    BoxRefinementSpec,
    ParitySpec,
    PatchRefinementSpec,
    SurfaceSpec,
    executable_metadata,
    render_mesh_dict,
    write_cfmesh_case,
)
from tools.mesh_parity.parity_report import (
    comparison_options_for_stage,
    run_parity,
    run_stage_ladder,
)


def _write_tetra_stl(path: Path) -> None:
    facets = (
        ((0, 0, 0), (0, 1, 0), (1, 0, 0)),
        ((0, 0, 0), (1, 0, 0), (0, 0, 1)),
        ((0, 0, 0), (0, 0, 1), (0, 1, 0)),
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    )
    lines = ["solid body"]
    for facet in facets:
        lines.extend(("  facet normal 0 0 0", "    outer loop"))
        lines.extend(f"      vertex {x} {y} {z}" for x, y, z in facet)
        lines.extend(("    endloop", "  endfacet"))
    lines.append("endsolid body")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _spec(stl: Path) -> ParitySpec:
    return ParitySpec(
        name="tiny_case",
        domain_bounds=(-2.0, 2.0, -2.0, 2.0, -2.0, 2.0),
        domain_patches={
            "xmin": "inlet",
            "xmax": "outlet",
            "ymin": "walls",
            "ymax": "walls",
            "zmin": "front_back",
            "zmax": "front_back",
        },
        surfaces=(SurfaceSpec(stl, "body"),),
        max_cell_size=0.6667,
        boundary_cell_size=0.33335,
        min_cell_size=0.166675,
    )


def test_mesh_dict_preserves_requested_sizes_and_patch_renames(tmp_path):
    stl = tmp_path / "body.stl"
    _write_tetra_stl(stl)
    mesh_dict = render_mesh_dict(_spec(stl), geometry_name="geometry.stl")

    assert "maxCellSize 0.6667;" in mesh_dict
    assert "maxCellSize 0.4;" not in mesh_dict
    assert "newName inlet;" in mesh_dict
    assert "newName body;" in mesh_dict


def test_mesh_dict_maps_box_and_patch_refinements_without_a_second_size_system(tmp_path):
    stl = tmp_path / "body.stl"
    _write_tetra_stl(stl)
    base = _spec(stl)
    spec = ParitySpec(
        name=base.name,
        domain_bounds=base.domain_bounds,
        domain_patches=base.domain_patches,
        surfaces=base.surfaces,
        max_cell_size=base.max_cell_size,
        boundary_cell_size=base.boundary_cell_size,
        min_cell_size=base.min_cell_size,
        box_refinements=(BoxRefinementSpec("near_body", (-1.0, 1.0, -1.5, 1.5, -0.5, 0.5), 0.2),),
        patch_refinements=(PatchRefinementSpec("body", 0.1, 0.3),),
    )

    mesh_dict = render_mesh_dict(spec, geometry_name="geometry.stl")

    assert "objectRefinements" in mesh_dict
    assert "type box;" in mesh_dict
    assert "centre (0.0 0.0 0.0);" in mesh_dict
    assert "lengthX 2.0;" in mesh_dict
    assert "localRefinement" in mesh_dict
    assert "refinementThickness 0.3;" in mesh_dict


def test_case_generation_records_original_stl_hash_and_outer_shell(tmp_path):
    stl = tmp_path / "body.stl"
    _write_tetra_stl(stl)
    directory = tmp_path / "cfmesh"

    geometry, triangles = write_cfmesh_case(_spec(stl), directory)

    assert geometry.is_file()
    assert triangles["body"].shape == (4, 3, 3)
    effective = json.loads((directory / "openonda_effective_config.json").read_text())
    assert effective["surfaces"][0]["sha256"]
    assert 'surfaceFile "constant/triSurface/openonda_parity_geometry.stl";' in (
        directory / "system" / "meshDict"
    ).read_text(encoding="ascii")
    generated = geometry.read_text(encoding="ascii")
    assert "solid body" in generated
    assert "solid inlet" in generated
    assert "solid front_back" in generated


def test_executable_metadata_can_use_an_environment_launcher(tmp_path):
    executable = tmp_path / "cartesianMesh"
    executable.write_text("#!/bin/sh\nprintf 'fake cfMesh help\\n'\n", encoding="ascii")
    executable.chmod(0o755)
    launcher = tmp_path / "openfoam"
    launcher.write_text('#!/bin/sh\nexec "$@"\n', encoding="ascii")
    launcher.chmod(0o755)

    metadata = executable_metadata(executable, launcher)

    assert metadata["version_or_help"] == "fake cfMesh help"
    assert metadata["launcher"]["path"] == str(launcher)
    assert metadata["launcher"]["sha256"]


def test_report_is_explicitly_blocked_when_an_executable_is_not_available(tmp_path):
    stl = tmp_path / "body.stl"
    _write_tetra_stl(stl)

    report = run_parity(
        _spec(stl),
        tmp_path / "report",
        cfmesh_executable=tmp_path / "missing-cartesianMesh",
    )

    assert report["status"] == "blocked"
    assert report["reason"] == "cfmesh_unavailable"
    assert (tmp_path / "report" / "parity_report.json").is_file()
    assert (tmp_path / "report" / "parity_summary.txt").is_file()


def test_stage_ladder_stops_at_the_first_non_pass_stage(tmp_path):
    stl = tmp_path / "body.stl"
    _write_tetra_stl(stl)

    report = run_stage_ladder(
        _spec(stl),
        tmp_path / "ladder",
        cfmesh_executable=tmp_path / "missing-cartesianMesh",
        stages=("templateGeneration", "surfaceTopology"),
    )

    assert report["status"] == "fail"
    assert report["first_bad_stage"] == "templateGeneration"
    assert report["reason"] == "cfmesh_unavailable"
    assert len(report["checkpoints"]) == 1


def test_geometry_profiles_follow_the_measured_stage_envelopes():
    projected = comparison_options_for_stage("patchAssignment")
    optimised = comparison_options_for_stage("edgeExtraction")

    assert projected.centroid_relative_tolerance == 1.0e-3
    assert projected.volume_relative_tolerance == 1.0e-2
    assert optimised.centroid_relative_tolerance == 5.0e-3
    assert optimised.volume_relative_tolerance == 5.0e-2
    wrapper = comparison_options_for_stage("boundaryLayerGeneration")
    assert wrapper.centroid_relative_tolerance == 5.0e-4
    assert wrapper.volume_relative_tolerance == 7.0e-2
    final_optimisation = comparison_options_for_stage("meshOptimisation")
    assert final_optimisation.centroid_relative_tolerance == 2.0e-5
    assert final_optimisation.volume_relative_tolerance == 2.5e-4


def test_blocked_report_still_records_effective_comparison_profile(tmp_path):
    stl = tmp_path / "body.stl"
    _write_tetra_stl(stl)

    report = run_parity(
        _spec(stl),
        tmp_path / "edge-report",
        cfmesh_executable=tmp_path / "missing-cartesianMesh",
        stop_after="edgeExtraction",
    )

    assert report["comparison_profile"] == "cfmesh_surface_optimisation"
    assert report["comparison_options"]["centroid_relative_tolerance"] == 5.0e-3
