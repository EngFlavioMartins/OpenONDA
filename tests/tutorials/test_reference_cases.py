"""Reference cases own their assets and preserve refinement/output contracts."""

from types import SimpleNamespace

import numpy as np
import pytest

from openonda.tutorial_runner import load_case_module
from openonda.tutorials import materialize_tutorial


def test_cylinder_reference_owns_geometry_and_refines_requested_sizes(tmp_path, monkeypatch):
    case = materialize_tutorial("coupled_fvm_vpm/cylinder_shedding_flow/reference_flow", tmp_path)
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


def test_cube_reference_owns_geometry_and_uses_explicit_spacing(tmp_path, monkeypatch):
    case = materialize_tutorial("coupled_fvm_vpm/cube_flow/reference_flow", tmp_path)
    standalone = tmp_path / "standalone"
    case.rename(standalone)
    module = load_case_module(standalone)
    captured = {}

    def capture(config, **kwargs):
        captured.update(config=config, **kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(module.fvm, "create_fvm_solver", capture)
    module.create_solver("grid_h0045", 0.045)

    mesh = captured["mesh"]
    assert mesh.surfaces[0].path.is_file()
    assert mesh.surfaces[0].path.is_relative_to(standalone)
    assert captured["solution_dir"] == standalone / "solution/grid_h0045"
    assert captured["samples_dir"] == standalone / "samples/grid_h0045"
    assert mesh.patch_refinements[0].cell_size == pytest.approx(0.045)


def test_cylinder_coupled_mesh_is_uniform_at_the_reference_fine_spacing(tmp_path):
    case = materialize_tutorial("coupled_fvm_vpm/cylinder_shedding_flow", tmp_path)
    module = load_case_module(case)
    mesh = module.FVM_MESH
    assert isinstance(mesh, module.msh.CartesianMesher)
    assert pytest.approx(0.04) == module.CELL_SIZE
    assert mesh.max_cell_size == pytest.approx(module.CELL_SIZE)
    assert mesh.background_cell_size == pytest.approx(module.CELL_SIZE)
    assert mesh.boundary_cell_size == pytest.approx(module.CELL_SIZE)
    assert mesh.cell_size_anchor == pytest.approx(module.CELL_SIZE)
    assert mesh.refinements == ()
    assert mesh.patch_refinements == ()
    assert mesh.effective_cell_size(module.CELL_SIZE) == pytest.approx(module.CELL_SIZE)
    assert mesh.requested_domain.bounds == module.FVM_BOX
    assert mesh.domain.bounds == pytest.approx(module.FVM_BOX)
    assert pytest.approx((-1.48, 1.48, -1.48, 1.48, -0.48, 0.48)) == module.FVM_BOX
    assert pytest.approx(24) == module.FVM_RESOLVED_SPAN / module.CELL_SIZE


def test_cylinder_transfer_region_fits_boundary_face_centres(tmp_path):
    """Keep exchange support inside the face-centre box used by the coupler."""
    case = materialize_tutorial("coupled_fvm_vpm/cylinder_shedding_flow", tmp_path)
    module = load_case_module(case)
    half_span = module.FVM_HALF_SPAN - 0.5 * module.CELL_SIZE
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
    assert module.REFERENCE_AREA == module.DIAMETER * module.FVM_RESOLVED_SPAN
