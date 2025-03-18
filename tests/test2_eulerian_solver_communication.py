import os
import pytest
import numpy as np
from openONDA.solvers.FVM import fvmModule as fvm
from openONDA.utilities import set_eulerian_module

@pytest.fixture
def setup_solver():
    # Setup the current working directory and OpenFOAM case directory
    current_dir = os.getcwd()
    OF_case_dir = os.path.join(current_dir, "sample_OF_directory")
    # Call the script that prepares the OpenFOAM simulation
    set_eulerian_module(current_dir, OF_case_dir)
    
    # Initialize the solver object
    solver = fvm.pyFoamSolver(["pimpleStepperFoam", "-case", f"{OF_case_dir}"])
    return solver

def test_get_run_time_value(setup_solver):
    solver = setup_solver
    current_time = solver.get_run_time_value()
    assert isinstance(current_time, float), f"Expected float, got {type(current_time)}"
    assert current_time >= 0.0, f"Expected positive value, got {current_time}"

def test_get_time_step(setup_solver):
    solver = setup_solver
    time_step_size = solver.get_time_step()
    assert isinstance(time_step_size, float), f"Expected float, got {type(time_step_size)}"
    assert time_step_size > 0, f"Expected positive value, got {time_step_size}"

def test_get_number_of_cells(setup_solver):
    solver = setup_solver
    n_cells = solver.get_number_of_cells()
    assert isinstance(n_cells, int), f"Expected int, got {type(n_cells)}"
    assert n_cells > 0, f"Expected positive value, got {n_cells}"

def test_get_number_of_boundary_faces(setup_solver):
    solver = setup_solver
    n_boundary_faces = solver.get_number_of_boundary_faces(patchName="numericalBoundary")
    assert isinstance(n_boundary_faces, int), f"Expected int, got {type(n_boundary_faces)}"
    assert n_boundary_faces > 0, f"Expected positive value, got {n_boundary_faces}"

def test_get_cell_volumes(setup_solver):
    solver = setup_solver
    cell_volumes = solver.get_cell_volumes()
    assert isinstance(cell_volumes, np.ndarray), f"Expected np.ndarray, got {type(cell_volumes)}"
    assert cell_volumes.size > 0, "Expected non-empty array for cell volumes"

def test_get_cell_centers(setup_solver):
    solver = setup_solver
    cell_centers = solver.get_cell_center_coordinates()
    assert isinstance(cell_centers, np.ndarray), f"Expected np.ndarray, got {type(cell_centers)}"
    assert cell_centers.shape[0] > 0, "Expected non-empty array for cell centers"

def test_get_boundary_face_centers(setup_solver):
    solver = setup_solver
    boundary_face_centers = solver.get_boundary_face_center_coordinates(patchName="numericalBoundary")
    assert isinstance(boundary_face_centers, np.ndarray), f"Expected np.ndarray, got {type(boundary_face_centers)}"
    assert boundary_face_centers.shape[0] > 0, "Expected non-empty array for boundary face centers"

def test_get_boundary_face_areas(setup_solver):
    solver = setup_solver
    boundary_face_areas = solver.get_boundary_face_areas(patchName="numericalBoundary")
    assert isinstance(boundary_face_areas, np.ndarray), f"Expected np.ndarray, got {type(boundary_face_areas)}"
    assert boundary_face_areas.size > 0, "Expected non-empty array for boundary face areas"

def test_get_boundary_face_normals(setup_solver):
    solver = setup_solver
    boundary_face_normals = solver.get_boundary_face_normals(patchName="numericalBoundary")
    assert isinstance(boundary_face_normals, np.ndarray), f"Expected np.ndarray, got {type(boundary_face_normals)}"
    assert boundary_face_normals.shape[0] > 0, "Expected non-empty array for boundary face normals"

def test_get_boundary_cell_centers(setup_solver):
    solver = setup_solver
    boundary_cell_centers = solver.get_boundary_cell_center_coordinates(patchName="numericalBoundary")
    assert isinstance(boundary_cell_centers, np.ndarray), f"Expected np.ndarray, got {type(boundary_cell_centers)}"
    assert boundary_cell_centers.shape[0] > 0, "Expected non-empty array for boundary cell centers"

def test_get_velocity_field(setup_solver):
    solver = setup_solver
    U = solver.get_velocity_field()
    assert U.size > 0, "Expected non-empty array for velocity field"

def test_get_pressure_field(setup_solver):
    solver = setup_solver
    P = solver.get_pressure_field()
    assert P.size > 0, "Expected non-empty array for pressure field"

def test_get_pressure_gradient_field(setup_solver):
    solver = setup_solver
    dPdx = solver.get_pressure_gradient_field()
    assert dPdx.size > 0, "Expected non-empty array for pressure gradient field"

def test_get_velocity_boundary_field(setup_solver):
    solver = setup_solver
    Ub = solver.get_velocity_boundary_field()
    assert Ub.size > 0, "Expected non-empty array for velocity boundary field"

def test_get_pressure_boundary_field(setup_solver):
    solver = setup_solver
    Pb = solver.get_pressure_boundary_field()
    assert Pb.size > 0, "Expected non-empty array for pressure boundary field"

def test_get_velocity_gradient_boundary_field(setup_solver):
    solver = setup_solver
    dUdxb = solver.get_velocity_gradient_boundary_field()
    assert dUdxb.size > 0, "Expected non-empty array for velocity gradient boundary field"

def test_get_pressure_gradient_boundary_field(setup_solver):
    solver = setup_solver
    dPdxb = solver.get_pressure_gradient_boundary_field()
    assert dPdxb.size > 0, "Expected non-empty array for pressure gradient boundary field"

def test_get_vorticity_field(setup_solver):
    solver = setup_solver
    W = solver.get_vorticity_field()
    assert W.size > 0, "Expected non-empty array for vorticity field"

def test_get_vorticity_boundary_field(setup_solver):
    solver = setup_solver
    Wb = solver.get_vorticity_boundary_field()
    assert Wb.size > 0, "Expected non-empty array for vorticity boundary field"

def test_set_time_step(setup_solver):
    solver = setup_solver
    time_step = np.array([0.01])  # Example time step value
    try:
        solver.set_time_step(time_step)
    except Exception as e:
        pytest.fail(f"set_time_step raised an exception: {e}")

def test_set_dirichlet_velocity_boundary_condition(setup_solver):
    solver = setup_solver
    n_boundary_faces = solver.get_number_of_boundary_faces()
    vxBoundary = np.zeros((n_boundary_faces,1))
    vyBoundary = np.zeros((n_boundary_faces,1))
    vzBoundary = np.zeros((n_boundary_faces,1))
    patch_name = "numericalBoundary"
    try:
        solver.set_dirichlet_velocity_boundary_condition(vxBoundary, vyBoundary, vzBoundary, patch_name)
    except Exception as e:
        pytest.fail(f"set_dirichlet_velocity_boundary_condition raised an exception: {e}")

def test_set_dirichlet_pressure_boundary_condition(setup_solver):
    solver = setup_solver
    pressure_values = np.array([1000.0])  # Example pressure values for Dirichlet boundary
    patch_name = "numericalBoundary"
    try:
        solver.set_dirichlet_pressure_boundary_condition(pressure_values, patch_name)
    except Exception as e:
        pytest.fail(f"set_dirichlet_pressure_boundary_condition raised an exception: {e}")

def test_set_neumann_pressure_boundary_condition(setup_solver):
    solver = setup_solver
    n_boundary_faces = solver.get_number_of_boundary_faces()
    dpdx = np.zeros((n_boundary_faces,1))
    dpdy = np.zeros((n_boundary_faces,1))
    dpdz = np.zeros((n_boundary_faces,1))
    patch_name = "numericalBoundary"
    try:
        solver.set_neumann_pressure_boundary_condition(dpdx, dpdy, dpdz, patch_name)
    except Exception as e:
        pytest.fail(f"set_neumann_pressure_boundary_condition raised an exception: {e}")
        

# === Tests for Correct Mass Flux Python === #

def test_correct_mass_flux_python(setup_solver):
    solver = setup_solver
    patch_name = "numericalBoundary"
    n_faces = solver.get_number_of_boundary_faces(patch_name)
    face_velocity_x = np.random.rand(n_faces)
    face_velocity_y = np.random.rand(n_faces)
    face_velocity_z = np.random.rand(n_faces)

    fx2, fy2, fz2 = solver.correct_mass_flux_python(
        face_velocity_x, face_velocity_y, face_velocity_z, patch_name
    )

    assert fx2.shape == face_velocity_x.shape
    assert fy2.shape == face_velocity_y.shape
    assert fz2.shape == face_velocity_z.shape
    print("test_correct_mass_flux_python() ok")



# === Tests for Correct Normal Pressure Gradient === #

def test_correct_normal_pressure_gradient(setup_solver):
    solver = setup_solver
    patch_name = "numericalBoundary"
    n_faces = solver.get_number_of_boundary_faces(patch_name)
    dpdx = np.random.rand(n_faces)
    dpdy = np.random.rand(n_faces)
    dpdz = np.random.rand(n_faces)

    dpdx_corr, dpdy_corr, dpdz_corr = solver.correct_normal_pressure_gradient(
        dpdx, dpdy, dpdz, patch_name
    )

    assert dpdx_corr.shape == dpdx.shape
    assert dpdy_corr.shape == dpdy.shape
    assert dpdz_corr.shape == dpdz.shape
    print("test_correct_normal_pressure_gradient() ok")


# === Tests for Mesh Centroid === #

def test_get_mesh_centroid(setup_solver):
    solver = setup_solver
    centroid = solver.get_mesh_centroid()

    assert isinstance(centroid, np.ndarray)
    assert centroid.shape == (3,)  # Should be (x, y, z) coordinates
    assert np.all(np.isfinite(centroid))  # Ensure no NaN or Inf values
    print("test_get_mesh_centroid() ok")
    
    
# === Tests for Evolution Functions === #

def test_evolve(setup_solver):
    solver = setup_solver
    solver.evolve()  # Ensure it runs without error
    print("test_evolve() ok")

def test_evolve_mesh(setup_solver):
    solver = setup_solver
    solver.evolve_mesh()  # Ensure mesh evolution runs without error
    print("test_evolve_mesh() ok")

def test_evolve_only_solution(setup_solver):
    solver = setup_solver
    solver.evolve_only_solution()  # Ensure only the solution evolves without error
    print("test_evolve_only_solution() ok")
 
def test_correct_mass_flux(setup_solver):
    solver = setup_solver
    patch_name = "numericalBoundary"
    solver.correct_mass_flux(patch_name)  # Ensure it runs without error
    print("test_correct_mass_flux() ok")