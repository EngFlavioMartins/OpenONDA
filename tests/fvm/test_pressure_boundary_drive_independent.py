"""Independent geometric/partition gates for pressure boundary reconstruction."""

from types import SimpleNamespace

import numpy as np
import pytest

from openonda import fvm
from source.solvers.fvm.core.solver import FVMSolver
from source.solvers.fvm.fields.boundary_reconstruction import boundary_owner_gradient
from source.solvers.fvm.fields.gradients import compute_gauss_gradient, compute_lsq_gradient
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.partition import localize_mesh_and_geometry
from source.solvers.fvm.solve import simple_solver
from tests.support.fvm_mesh import structured_box


JACOBIAN = np.array([[.2, -.7, 1.3], [.6, -.1, -.4], [-.8, 1.2, -.1]])
TRANSFORM = np.array([[1., .3, -.2], [.1, 1.2, .4], [.2, -.1, .8]])


def _skew_box():
    mesh = structured_box(4, 3, 3)
    mesh["vertex_position"] = mesh["vertex_position"] @ TRANSFORM.T
    return mesh, compute_mesh_geometry(mesh)


def _boundary_faces(mesh):
    return np.arange(mesh["n_interior_faces"], mesh["n_faces"])


def _no_collectives(*args, **kwargs):
    raise AssertionError("Boundary-only reconstruction must not communicate")


def test_real_stencil_ignores_poisoned_face_ghosts_and_reuses_geometry_cache():
    mesh, geometry = _skew_box()
    mesh["_parallel_context"] = SimpleNamespace(
        is_partitioned=True, exchange_halo=_no_collectives, global_max=_no_collectives
    )
    faces = _boundary_faces(mesh)
    values = geometry["cell_centre"] @ JACOBIAN.T + [.3, -.2, .1]
    ghosts = np.full((len(faces), 3), np.nan)
    first = boundary_owner_gradient(
        np.concatenate((values, ghosts)), mesh, geometry, faces,
        displacements=geometry["cell_connection_vector"][faces],
    )
    np.testing.assert_allclose(first, np.broadcast_to(JACOBIAN.T, first.shape), atol=2e-13)
    cache = geometry["_boundary_owner_lsq_real"]
    # Cached geometry must consume the new real field, never cached values or
    # the physical ghosts. No hidden collective is needed after halo refresh.
    second = boundary_owner_gradient(
        np.concatenate((-2 * values, np.full_like(ghosts, 1e99))), mesh, geometry, faces,
        displacements=geometry["cell_connection_vector"][faces],
    )
    assert geometry["_boundary_owner_lsq_real"] is cache
    np.testing.assert_allclose(second, -2 * first, atol=3e-13)
    assert len(next(iter(cache.entries.values())).cells) < mesh["n_cells"]


def test_rank_two_stencil_accepts_identifiable_plane_and_rejects_span_extrapolation():
    mesh = structured_box(4, 4, 1)
    geometry = compute_mesh_geometry(mesh)
    faces = _boundary_faces(mesh)
    displacement = geometry["cell_connection_vector"][faces]
    in_plane = faces[np.abs(displacement[:, 2]) < 1e-12]
    span = faces[np.abs(displacement[:, 2]) > 1e-12]
    matrix = JACOBIAN.copy()
    matrix[:, 2] = 0
    values = geometry["cell_centre"] @ matrix.T
    actual = boundary_owner_gradient(
        values, mesh, geometry, in_plane,
        displacements=geometry["cell_connection_vector"][in_plane],
    )
    np.testing.assert_allclose(actual, np.broadcast_to(matrix.T, actual.shape), atol=3e-14)
    with pytest.raises(ValueError, match="cannot resolve extrapolation direction"):
        boundary_owner_gradient(
            values, mesh, geometry, span,
            displacements=geometry["cell_connection_vector"][span],
        )


@pytest.mark.parametrize("components", [1, 3])
@pytest.mark.parametrize("scheme", ["gauss", "compiled_gauss", "lsq"])
def test_native_boundary_gradient_is_affine_exact_at_skew_face_location(components, scheme):
    """Accept the physical face gradient, not only backend agreement."""
    mesh, geometry = _skew_box()
    for patch in mesh["boundary"]:
        patch.update(velocity_type="fixedValue", pressure_type="fixedValue")
    faces = _boundary_faces(mesh)
    points = np.concatenate((geometry["cell_centre"], geometry["face_centre"][faces]))
    field = points @ JACOBIAN.T + [.3, -.2, .1]
    if components == 1:
        field = field[:, 0]
    if scheme == "lsq":
        geometry = compute_mesh_geometry(mesh, gradient_scheme="lsq")
        gradient = compute_lsq_gradient(field, mesh, geometry)
    else:
        if scheme == "compiled_gauss":
            geometry["_operator_backend"] = "numba"
        gradient = compute_gauss_gradient(field, mesh, geometry)
    expected = np.broadcast_to(JACOBIAN.T[:, :components], gradient.shape)
    np.testing.assert_allclose(gradient, expected, rtol=0, atol=5e-13)


def test_real_processor_neighbors_preserve_serial_boundary_gradient_without_collectives():
    mesh, geometry = _skew_box()
    field = geometry["cell_centre"] @ JACOBIAN.T + [.3, -.2, .1]
    expected = boundary_owner_gradient(
        field, mesh, geometry, _boundary_faces(mesh),
        displacements=geometry["cell_connection_vector"][_boundary_faces(mesh)],
    )
    global_gradient = np.zeros((mesh["n_faces"], 3, 3))
    global_gradient[_boundary_faces(mesh)] = expected
    saw_processor_neighbor = False
    for rank in (0, 1):
        local, geo, partition = localize_mesh_and_geometry(mesh, geometry, rank=rank, size=2)
        local["_parallel_context"] = SimpleNamespace(
            is_partitioned=True, exchange_halo=_no_collectives, global_max=_no_collectives
        )
        ids = local["global_face_id"]
        faces = np.flatnonzero(ids >= mesh["n_interior_faces"])
        values = field[partition.local_global_ids]
        actual = boundary_owner_gradient(
            values, local, geo, faces,
            displacements=geo["cell_connection_vector"][faces],
        )
        np.testing.assert_allclose(actual, global_gradient[ids[faces]], atol=3e-13)
        stencil = next(iter(geo["_boundary_owner_lsq_real"].entries.values()))
        owned = len(partition.owned_global_ids)
        saw_processor_neighbor |= bool(np.any(stencil.neighbour >= owned))
    assert saw_processor_neighbor


@pytest.mark.parametrize("diagonal", [False, True], ids=["scalar_variable_D", "diagonal_variable_D"])
def test_skew_linear_pressure_flux_uses_same_variable_inverse_and_no_extra_pressure_data(diagonal):
    mesh, geometry = _skew_box()
    faces = _boundary_faces(mesh)
    owners = mesh["owners"][faces]
    n_cells = mesh["n_cells"]
    normal = geometry["face_area_vector"][faces] / geometry["face_area"][faces, None]
    gradient = np.array([.7, -.4, .2])
    centres = geometry["cell_centre"]
    coefficient = .025 + .003 * centres[:, 0]
    if diagonal:
        coefficient = coefficient[:, None] * [1., 1.7, .6]
        component_coefficient = coefficient
    else:
        component_coefficient = coefficient[:, None]
    # This is the exact projection defined by the same owner inverse used by
    # the boundary conductance. Its spatial variation is deliberate.
    drive_face = geometry["face_centre"][faces] @ JACOBIAN.T + [.3, -.2, .1]
    velocity_face = drive_face - component_coefficient[owners] * gradient
    velocity = np.concatenate((np.zeros((n_cells, 3)), velocity_face))
    pressure = np.concatenate((centres @ gradient, np.full(len(faces), np.nan)))
    grad = np.broadcast_to(gradient, (len(pressure), 3)).copy()
    phi_drive = np.zeros(mesh["n_faces"])
    phi_drive[faces] = np.einsum("fi,fi->f", drive_face, geometry["face_area_vector"][faces])
    for patch in mesh["boundary"]:
        patch.update(pressure_type="fixedFluxPressure", velocity_type="normalValueTangentialGradient")
    simple_solver._update_fixed_flux_pressure_boundaries(
        pressure, velocity, coefficient, mesh, geometry, mesh["boundary"],
        kinematic_pressure_gradient=grad, pressure_free_face_flux=phi_drive,
    )
    np.testing.assert_allclose(pressure[n_cells:], geometry["face_centre"][faces] @ gradient, atol=2e-14)
    for patch in mesh["boundary"]:
        assert "kinematic_pressure_value" not in patch
    # A skew pressure increment represents the physical point difference,
    # including its tangent displacement; it does not alter prescribed flux.
    delta = pressure[n_cells:] - pressure[owners]
    np.testing.assert_allclose(delta, geometry["cell_connection_vector"][faces] @ gradient, atol=2e-14)
    recovered_normal = np.einsum("fi,fi->f", drive_face - component_coefficient[owners] * grad[owners], normal)
    np.testing.assert_allclose(recovered_normal, np.einsum("fi,fi->f", velocity_face, normal), atol=2e-14)


@pytest.mark.parametrize("skew", [False, True], ids=["orthogonal", "skew"])
def test_piso_predictor_freeze_preserves_closed_cavity_projection(tmp_path, monkeypatch, skew):
    """The shared predictor does not change homogeneous pressure closures.

    This cavity has an evolving pressure and velocity, rather than a uniform
    zero-correction state. Rebuilding H/A each PISO iteration provides the
    previous ownership semantics with every other operator held identical.
    """
    def solve(name):
        mesh = structured_box(6, 6, 1)
        if skew:
            mesh["vertex_position"][:, 0] += .3 * mesh["vertex_position"][:, 1]
        setup = fvm.FVMSetup(
            case_name="independent_cavity_predictor_ownership",
            logging=fvm.LoggingConfig(console=False),
            backup=fvm.BackupConfig(schedule=None, write_at_end=False),
            time=fvm.TimeConfig(time_step_size=.002, end_time=.008),
            schemes=fvm.DiscretizationConfig(gradient_scheme="lsq", convection_scheme="upwind"),
            linear=fvm.LinearSolverConfig(linear_solver="spsolve", pressure_solver="spsolve"),
            pimple=fvm.PimpleControl(n_outer_correctors=2, n_correctors=2,
                                    velocity_relaxation=.7, pressure_relaxation=.3),
            transport=fvm.TransportConfig(density=1., kinematic_viscosity=.02),
            boundaries=[
                fvm.BoundaryConfig.wall("xmin"),
                fvm.BoundaryConfig.wall("xmax"),
                fvm.BoundaryConfig.wall("ymin"),
                fvm.BoundaryConfig(name="ymax", velocity_type="fixedValue",
                                   velocity_value=[.5, 0., 0.], pressure_type="zeroGradient"),
                fvm.BoundaryConfig.empty("zmin"),
                fvm.BoundaryConfig.empty("zmax"),
            ],
            initial_velocity=[0., 0., 0.],
        )
        with FVMSolver(setup, str(tmp_path / name), mesh_data=mesh) as solver:
            solver.auto_write = False
            for _ in range(4):
                solver.advance()
                assert solver.last_diagnostics.max_continuity_error < 1e-11
            return (solver.velocity.copy(), solver.kinematic_pressure.copy(),
                    solver.volumetric_face_flux.copy())

    frozen = solve("per_predictor")
    original = simple_solver.assemble_pressure_correction_equation_rhie_chow

    def rebuild_each_correction(*args, **kwargs):
        kwargs["frozen_velocity_h_over_a"] = None
        return original(*args, **kwargs)

    monkeypatch.setattr(simple_solver, "assemble_pressure_correction_equation_rhie_chow",
                        rebuild_each_correction)
    rebuilt = solve("per_correction")
    assert np.linalg.norm(frozen[1]) > 1e-5
    for actual, expected in zip(frozen, rebuilt):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-11)
