# .pyi ensures that docstrings and function signatures appear correctly while keeping the performance of compiled Cython code. 

# fvmModule.pyi - Stub file for OpenFOAM Python bindings in OpenONDA
from typing import List, Tuple
import numpy as np

class pyFoamSolver:
    """
    Python interface for interacting with OpenFOAM solvers.

    This class provides methods to evolve the simulation, retrieve mesh properties,
    and correct mass fluxes.
    """

    def __init__(self, args: List[str] = ["pimpleStepperFoam"]) -> None: ...
    
    def evolve(self) -> None:
        """Advances the OpenFOAM simulation by one time step, updating the state, fields, and mesh accordingly."""
        ...
    
    def evolve_mesh(self) -> None:
        """Updates the mesh of the OpenFOAM simulation from time step t to t+1."""
        ...

    def evolve_only_solution(self) -> None:
        """Advances the solution of the OpenFOAM simulation without updating the mesh."""
        ...

    def correct_mass_flux(self, patchName: str = "numericalBoundary") -> None:
        """Corrects the mass flux across the specified boundary patch."""
        ...
    
    def get_run_time_value(self) -> float:
        """Get the current flow time in seconds."""
        ...

    def get_time_step(self) -> float:
        """Get the size of the current time step in seconds."""
        ...

    def get_number_of_nodes(self) -> int:
        """Get the number of nodes in the simulation."""
        ...

    def get_number_of_cells(self) -> int:
        """Get the total number of cells in the simulation."""
        ...

    def get_number_of_boundary_nodes(self, patchName: str = "numericalBoundary") -> int:
        """Get the number of boundary nodes for a specified OpenFOAM patch."""
        ...

    def get_number_of_boundary_faces(self, patchName: str = "numericalBoundary") -> int:
        """Get the number of boundary faces for a specified OpenFOAM patch."""
        ...

    def get_node_coordinates(self) -> np.ndarray:
        """Returns the coordinates of the nodes in the simulation."""
        ...

    def get_cell_volumes(self) -> np.ndarray:
        """Returns the volumes of the cells in the simulation."""
        ...

    def get_cell_center_coordinates(self) -> np.ndarray:
        """Returns the coordinates of the cell centers in the simulation."""
        ...

    def get_boundary_face_center_coordinates(self, patchName: str = "numericalBoundary") -> np.ndarray:
        """Returns the coordinates of the boundary face centers for a given patch."""
        ...

    def get_boundary_face_areas(self, patchName: str = "numericalBoundary") -> np.ndarray:
        """Returns the areas of the boundary faces for a given patch."""
        ...

    def get_boundary_face_normals(self, patchName: str = "numericalBoundary") -> np.ndarray:
        """Returns the normals of the boundary faces for a given patch."""
        ...

    def get_velocity_field(self) -> np.ndarray:
        """Returns the velocity field as a NumPy array."""
        ...

    def get_velocity_boundary_field(self, patchName: str = "numericalBoundary") -> np.ndarray:
        """Retrieve the velocity boundary field."""
        ...

    def get_pressure_field(self) -> np.ndarray:
        """Returns the pressure field."""
        ...

    def get_velocity_gradient(self) -> np.ndarray:
        """Retrieve the velocity gradient field."""
        ...

    def get_velocity_gradient_boundary_field(self, patchName: str = "numericalBoundary") -> np.ndarray:
        """Retrieve the velocity gradient boundary field."""
        ...

    def get_pressure_gradient_field(self) -> np.ndarray:
        """Returns the pressure gradient field."""
        ...

    def get_pressure_boundary_field(self, patchName: str = "numericalBoundary") -> np.ndarray:
        """Retrieve the pressure boundary field."""
        ...

    def get_pressure_gradient_boundary_field(self, patchName: str = "numericalBoundary") -> np.ndarray:
        """Retrieve the pressure gradient boundary field."""
        ...

    def get_vorticity_field(self) -> np.ndarray:
        """Retrieve the vorticity gradient field."""
        ...

    def get_vorticity_boundary_field(self, patchName: str = "numericalBoundary") -> np.ndarray:
        """Retrieve the vorticity boundary field."""
        ...

    def set_time_step(self, deltaT: float) -> None:
        """Set the desired time-step size."""
        ...

    def set_dirichlet_velocity_boundary_condition(
        self, vxBoundary: np.ndarray, vyBoundary: np.ndarray, vzBoundary: np.ndarray, patchName: str = "numericalBoundary"
    ) -> None:
        """Set Dirichlet velocity boundary conditions."""
        ...

    def set_dirichlet_pressure_boundary_condition(self, pBoundary: np.ndarray, patchName: str = "numericalBoundary") -> None:
        """Set Dirichlet pressure boundary conditions."""
        ...

    def set_neumann_pressure_boundary_condition(
        self, dpdxBoundary: np.ndarray, dpdyBoundary: np.ndarray, dpdzBoundary: np.ndarray, patchName: str = "numericalBoundary"
    ) -> None:
        """Set Neumann pressure boundary conditions."""
        ...

    def correct_mass_flux_python(
        self, faceVelocityX: np.ndarray, faceVelocityY: np.ndarray, faceVelocityZ: np.ndarray, patchName: str = "numericalBoundary"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Correct the mass flux at each face along the numerical boundary."""
        ...

    def correct_normal_pressure_gradient(
        self, dpdx: np.ndarray, dpdy: np.ndarray, dpdz: np.ndarray, patchName: str = "numericalBoundary"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Correct the normal pressure gradient at each face along the numerical boundary."""
        ...

    def get_total_circulation(self) -> float:
        """Get the total circulation in the finite volume mesh region."""
        ...

    def get_mesh_centroid(self) -> np.ndarray:
        """Calculate the centroid of the mesh."""
        ...
