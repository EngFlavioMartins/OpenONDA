"""Regression tests for solver-safe curved cut-cell extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from source.solvers.fvm.mesh.adaptive_cartesian import (
    AdaptiveCartesianMesher,
    BoundaryLayerSpec,
)
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.validation import validate_geometry, validate_topology
from tutorials.coupled_fvm_vpm.cylinder_shedding_flow.reference_flow.reference_flow_setup import (
    grid_mesh,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = REPOSITORY_ROOT / "tutorials" / "coupled_fvm_vpm" / "cylinder_shedding_flow"
CYLINDER_STL = CASE_DIR / "assets" / "cylinder_long.stl"


def test_curved_cylinder_mapping_has_no_solid_centres_or_degenerate_faces():
    mesher = AdaptiveCartesianMesher(
        domain=(-1.0, 1.0, -1.0, 1.0, -2.5, 2.5),
        max_cell_size=0.5,
        surface_file=CYLINDER_STL,
        wall_patch_name="cylinder",
        surface_cell_size=0.125,
        surface_may_cross_domain_boundary=True,
    )

    mesh = mesher.build()
    validate_topology(mesh)
    geometry = compute_mesh_geometry(mesh, compute_lsq=False)
    validate_geometry(mesh, geometry)

    centres = np.asarray(geometry["cell_centre"])
    inside = (np.linalg.norm(centres[:, :2], axis=1) < 0.5 - 1.0e-10) & (
        np.abs(centres[:, 2]) < 2.5 - 1.0e-10
    )
    assert not np.any(inside)
    assert np.all(np.asarray(geometry["face_area"]) > 0.0)
    assert np.all(np.asarray(geometry["cell_volume"]) > 0.0)

    wall = next(patch for patch in mesh["boundary"] if patch["name"] == "cylinder")
    first = int(wall["start_face"])
    faces = mesh["faces"][first : first + int(wall["n_faces"])]
    assert {len(face) for face in faces}.issubset({3, 4})
    projection = mesh["mesh_generation"]["surface_projection"]
    assert projection["accepted_points"] == projection["attempted_points"]
    assert projection["rejected_nonpositive_volume"] == 0
    assert projection["rejected_volume_ratio"] == 0


def test_cylinder_boundary_layer_is_complete_and_stitched():
    mesher = AdaptiveCartesianMesher(
        domain=(-1.5, 1.5, -1.5, 1.5, -0.5, 0.5),
        max_cell_size=0.5,
        surface_file=CYLINDER_STL,
        wall_patch_name="cylinder",
        surface_cell_size=0.125,
        surface_may_cross_domain_boundary=True,
        boundary_layer=BoundaryLayerSpec(
            first_cell_height=1.0 / 64.0,
            layers=4,
            growth_ratio=1.1,
            transition_layers=4,
            interface_half_width=0.75,
        ),
    )

    mesh = mesher.build()
    validate_topology(mesh)
    geometry = compute_mesh_geometry(mesh, compute_lsq=False)
    quality = validate_geometry(mesh, geometry)

    generation = mesh["mesh_generation"]["boundary_layer"]
    layer_index = np.asarray(mesh["boundary_layer_index"])
    expected = generation["theta_cells"] * generation["z_cells"]
    assert generation["wall_layers"] == 4
    assert all(np.count_nonzero(layer_index == layer) == expected for layer in range(4))
    assert "__boundary_layer_interface__" not in {
        patch["name"] for patch in mesh["boundary"]
    }
    assert quality["max_non_orthogonality_deg"] < 60.0
    assert quality["max_skewness"] < 0.5
    assert "surface_projection" not in mesh["mesh_generation"]


def test_cylinder_boundary_layer_supports_independent_spanwise_extrusion():
    mesher = AdaptiveCartesianMesher(
        domain=(-1.5, 1.5, -1.5, 1.5, -0.5, 0.5),
        max_cell_size=0.5,
        surface_file=CYLINDER_STL,
        wall_patch_name="cylinder",
        surface_cell_size=0.125,
        surface_may_cross_domain_boundary=True,
        boundary_layer=BoundaryLayerSpec(
            first_cell_height=1.0 / 64.0,
            layers=4,
            growth_ratio=1.1,
            transition_layers=4,
            interface_half_width=0.75,
            spanwise_cell_size=0.25,
        ),
    )

    mesh = mesher.build()
    validate_topology(mesh)
    geometry = compute_mesh_geometry(mesh, compute_lsq=False)
    quality = validate_geometry(mesh, geometry)
    layer = mesh["mesh_generation"]["boundary_layer"]

    assert mesh["mesh_generation"]["spanwise_cells"] == 4
    assert mesh["mesh_generation"]["spanwise_cell_size"] == 0.25
    assert layer["z_cells"] == 4
    assert np.array_equal(
        np.unique(np.asarray(mesh["vertex_position"])[:, 2]),
        np.linspace(-0.5, 0.5, 5),
    )
    assert quality["max_non_orthogonality_deg"] < 60.0
    assert quality["max_skewness"] < 0.5


def test_cylinder_grid_study_uses_the_exact_1_2_4_12_sizes():
    dx = 1.0 / 24.0
    mesher = grid_mesh(dx)
    leaves, _limits, _interface = mesher._study_leaves()

    np.testing.assert_allclose(mesher.surface_cell_size, dx)
    np.testing.assert_allclose(mesher.far_field_cell_size, 12.0 * dx)
    assert set(np.unique(leaves[:, 2])) == {1, 2, 4, 12}


def test_cylinder_tutorial_uses_stl_wall_patch_and_sparse_field_backups():
    reference = (CASE_DIR / "reference_flow" / "reference_flow_setup.py").read_text()
    assert CYLINDER_STL.is_file()
    assert "ExplicitCylinderGridMesher(" in reference
    assert "wall_cell_size=dx" in reference
    assert "near_body_half_width=2.0" in reference
    assert "wake_half_width=4.0" in reference
    assert "FORCE_INTERVAL_TIME = 0.02" in reference
    assert "FIELD_INTERVAL_TIME = 2.5" in reference
    assert 'BoundaryConfig.wall("cylinder")' in reference
    assert "ForceSampler(" in reference
    assert "solution_dir=solution_dir" in reference
    assert "samples_dir=samples_dir" in reference
    assert "ImmersedBody" not in reference
    assert "IBMForceSampler" not in reference
    assert "set_immersed_bodies" not in reference

    coupled = (CASE_DIR / "cylinder_shedding_flow_setup.py").read_text()
    assert 'BoundaryConfig.wall("cylinder")' in coupled
    assert 'BoundaryConfig.slip("zmin")' in coupled
    assert 'BoundaryConfig.slip("zmax")' in coupled
    assert "checkpoint_at_stop=True" in coupled
    assert "ImmersedBody" not in coupled
    assert "IBMForceSampler" not in coupled


def test_cylinder_tutorial_has_the_minimal_user_interface():
    reference_dir = CASE_DIR / "reference_flow"
    reference_run = (reference_dir / "allrun.sh").read_text()
    coupled_run = (CASE_DIR / "allrun.sh").read_text()

    assert "mpirun" not in reference_run
    assert "mpiexec" not in reference_run
    assert "NUMBER_OF_CORES = 6" in (
        reference_dir / "reference_flow_setup.py"
    ).read_text()
    assert "reference_flow_setup.py --dx 0.08333333333333333 --case-name very_coarse" in reference_run
    assert "reference_flow_setup.py --dx 0.041666666666666664 --case-name coarse" in reference_run
    assert "reference_flow_setup.py --dx 0.027777777777777776 --case-name medium" in reference_run
    assert "reference_flow_setup.py --dx 0.020833333333333332 --case-name fine" in reference_run
    assert "allplot.sh" not in reference_run
    assert reference_run.count("python assets/plot_") == 1
    assert "python -u cylinder_shedding_flow_setup.py" in coupled_run

    root_files = {path.name for path in CASE_DIR.iterdir() if path.is_file()}
    reference_files = {path.name for path in reference_dir.iterdir() if path.is_file()}
    assert root_files == {
        "allrun.sh",
        "allclean.sh",
        "allplot.sh",
        "cylinder_shedding_flow_setup.py",
    }
    assert reference_files == {
        "allrun.sh",
        "allclean.sh",
        "allplot.sh",
        "reference_flow_setup.py",
    }
    assert {path.name for path in (CASE_DIR / "assets").iterdir() if path.is_file()} == {
        "cylinder_long.stl",
        "plot_cylinder.py",
        "postprocess.py",
    }
    assert {
        path.name for path in (reference_dir / "assets").iterdir() if path.is_file()
    } == {"plot_grid_study.py", "postprocess.py"}
