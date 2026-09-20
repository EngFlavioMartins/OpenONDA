"""Reference cases own their assets and preserve refinement/output contracts."""

from types import SimpleNamespace

import numpy as np
import pytest

from openonda.tutorial_runner import load_case_module
from openonda.tutorials import materialize_tutorial


@pytest.mark.parametrize("name", ["cube_flow", "cylinder_shedding_flow"])
def test_reference_case_owns_geometry_and_refines_requested_sizes(tmp_path, monkeypatch, name):
    case = materialize_tutorial(f"coupled_fvm_vpm/{name}/reference_flow", tmp_path)
    # Move the case away from every parent tutorial directory.
    standalone = tmp_path / "standalone"
    case.rename(standalone)
    module = load_case_module(standalone)
    captures = []

    def capture(config, **kwargs):
        captures.append((config, kwargs))
        return SimpleNamespace()

    monkeypatch.setattr(module.fvm, "create_fvm_solver", capture)
    for spacing in (0.125, 0.0625):
        module.create_solver(f"spacing-{spacing}", spacing)
    sizes = []
    for config, arguments in captures:
        mesh = arguments["mesh"]
        source = getattr(mesh, "source", mesh)
        assert source.surfaces
        for surface in source.surfaces:
            assert surface.path.is_file()
            assert surface.path.is_relative_to(standalone)
        assert arguments["solution_dir"].is_relative_to(standalone)
        assert arguments["samples_dir"].is_relative_to(standalone)
        assert config.boundaries
        sizes.append([r.cell_size for r in source.patch_refinements])
    np.testing.assert_allclose(np.array(sizes[0]) / sizes[1], 2)


def test_reference_restart_reuses_cached_native_mesh(tmp_path, monkeypatch):
    case = materialize_tutorial("coupled_fvm_vpm/cylinder_shedding_flow/reference_flow", tmp_path)
    module = load_case_module(case)
    captured = []

    def capture(config, **kwargs):
        captured.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(module.fvm, "create_fvm_solver", capture)
    cached_mesh = case / "solution" / "restart" / "fvm" / "mesh.npz"
    cached_mesh.parent.mkdir(parents=True)
    cached_mesh.touch()

    module.create_solver("restart", 0.04, restart_from=case / "solution" / "restart" / "backup")

    assert captured[-1]["mesh"] == cached_mesh


def test_cylinder_coupled_mesh_has_four_direct_resolution_levels(tmp_path):
    """The compact coupled FVM preserves four direct Cartesian sizes."""
    case = materialize_tutorial("coupled_fvm_vpm/cylinder_shedding_flow", tmp_path)
    module = load_case_module(case)
    mesh = module.FVM_MESH
    assert isinstance(mesh, module.msh.CartesianMesher)
    assert len(mesh.refinements) == 1
    assert mesh.refinements[0].name == "nearBody"
    assert len(mesh.patch_refinements) == 1
    assert mesh.patch_refinements[0].patch == "cylinder"
    assert mesh.max_cell_size == pytest.approx(16.0 * module.CELL_SIZE)
    assert mesh.background_cell_size == pytest.approx(16.0 * module.CELL_SIZE)
    assert mesh.cell_size_anchor == pytest.approx(module.CELL_SIZE)
    requested_levels = tuple(module.CELL_SIZE * 2**level for level in range(4))
    assert tuple(
        sorted({mesh.effective_cell_size(requested) for requested in requested_levels})
    ) == pytest.approx(requested_levels)
    generated = mesh.build(stop_after="templateGeneration")
    assert tuple(
        np.unique(np.round(np.asarray(generated["cell_sizes"], dtype=np.float64), 12))
    ) == pytest.approx(requested_levels)
    assert mesh.requested_domain.bounds == module.FVM_BOX
    assert mesh.domain.bounds == pytest.approx((-1.8, 1.8, -1.8, 1.8, -0.6, 0.6))


def test_cylinder_transfer_region_fits_boundary_face_centres(tmp_path):
    """Keep exchange support inside the face-centre box used by the coupler."""
    case = materialize_tutorial("coupled_fvm_vpm/cylinder_shedding_flow", tmp_path)
    module = load_case_module(case)
    half_span = module.FVM_HALF_SPAN - 0.5 * module.SPANWISE_CELL_SIZE
    face_centre_box = np.asarray((*module.FVM_BOX[:4], -half_span, half_span))
    module.COUPLER_SETUP.validate_transfer_region_box(face_centre_box)


def test_cylinder_gbd_grid_is_aligned_and_within_gpu_budget(tmp_path):
    """Keep renewal and GBD aligned without overcommitting the fixed grid."""
    case = materialize_tutorial("coupled_fvm_vpm/cylinder_shedding_flow", tmp_path)
    module = load_case_module(case)
    viscous = module.VPM_CASE.numerics.viscous
    assert viscous.particle_spacing == pytest.approx(module.VPM_PARTICLE_SPACING)
    assert viscous.gbd_grid_spacing == pytest.approx(module.VPM_PARTICLE_SPACING)

    domain = module.VPM_DOMAIN
    padding = viscous.gbd_domain_padding
    spacing = viscous.gbd_grid_spacing
    dimensions = tuple(
        max(5, int(np.ceil((upper - lower + 2.0 * padding * spacing) / spacing)) + 1)
        for lower, upper in zip(domain[::2], domain[1::2], strict=True)
    )
    grid_bytes = int(np.prod(dimensions)) * 32
    assert dimensions == (411, 211, 275)
    assert grid_bytes < 1 << 30
    assert module.VPM_CASE.numerics.bodies[0].diffusion_mask == "cylinder_z"
    assert module.REFERENCE_AREA == module.DIAMETER * module.FVM_RESOLVED_SPAN
    assert module.PANEL_REFERENCE_AREA == module.DIAMETER * module.CYLINDER_LENGTH
