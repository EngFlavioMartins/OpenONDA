# ==================================================
# Module Imports and Declarations
# ==================================================

# To build this file use: python setup.py build_ext --inplace

__all__ = [
    'pyFoamSolver'
]

__doc__ = """
fvmModule: Interface for OpenFOAM solvers in OpenONDA.

This module provides Python bindings for interacting with OpenFOAM solvers using the PIMPLE algorithm. It includes methods to access mesh properties, correct mass fluxes, and evolve the simulation.
"""


# ==================================================
# Import Statements
# ==================================================

import os as _os
import shutil as _sh
import ctypes
import sys as _sys

import cython as _cython
import numpy as _np
cimport numpy as _np
from scipy.interpolate import griddata as _griddata

from libc.stdlib cimport malloc, free
from cpython.version cimport PY_MAJOR_VERSION

from .foamSolverWrapper cimport cppFoamSolver

# ==================================================
# Python Wrapper Class
# ==================================================
cdef class pyFoamSolver:

      """
      pyFoamSolver: Python interface for interacting with OpenFOAM solvers.

      This class provides methods to evolve the simulation, retrieve mesh properties,
      and correct mass fluxes.
      """

      cdef cppFoamSolver *cppFoamLib  # Declare as C++ pointer

      # ==================================================
      # Constructor and Destructor:
      # ==================================================

      def __cinit__(self, args=["pimpleStepperFoam"]):
            """
            Initializes the pyFoamSolver object by allocating and initializing
            the corresponding C++ object.

            Parameters
            ----------
            args : list of str, optional
                  A list of command-line arguments to pass to the C++ object. Default is ["pimpleStepperFoam"].

            Returns
            -------
            None
            """
            numArgs = len(args)
            args = [x.encode('utf-8') for x in args]

            cdef char **string_buf = <char**>malloc(numArgs * sizeof(char*))
            if string_buf is NULL:
                  raise MemoryError()

            for i in range(numArgs):
                  string_buf[i] = args[i]

            self.cppFoamLib = new cppFoamSolver(numArgs, string_buf)
            free(string_buf)

      def __dealloc__(self):
            """
            Deallocates the C++ object when the Python object is garbage-collected.

            Returns
            -------
            None
            """
            del self.cppFoamLib


      # ================================================== #
      # Simulation methods:
      # ================================================== #
      def evolve(self):
            """
            Advances the OpenFOAM simulation by one time step, updating the state,
            fields, and mesh accordingly.

            Returns
            -------
            None
            """
            self.cppFoamLib.evolve()

      def evolve_mesh(self):
            """
            Updates the mesh of the OpenFOAM simulation from time step t to t+1.

            Note
            ----
            This function requires a moving mesh problem to be set up in OpenFOAM.

            Returns
            -------
            None
            """
            self.cppFoamLib.evolve_mesh()

      def evolve_only_solution(self):
            """
            Advances the solution of the OpenFOAM simulation without updating the mesh.

            Note
            ----
            This function is used when the mesh remains stationary.

            Returns
            -------
            None
            """
            self.cppFoamLib.evolve_only_solution()

      def correct_mass_flux(self, patchName="numericalBoundary"):
            """
            Corrects the mass flux across the specified boundary patch.

            Parameters
            ----------
            patchName : str, optional
                  Name of the OpenFOAM boundary patch. Default is "numericalBoundary".

            Returns
            -------
            None
            """
            patchName = patchName.encode()
            self.cppFoamLib.correct_mass_flux(patchName)
            

      # ==================================================
      # Simulation Methods
      # ==================================================
      def get_run_time_value(self):
            """
            Get the current flow time in seconds.

            Returns
            -------
            float
                  The current flow time in seconds.
            """
            return self.cppFoamLib.get_run_time_value()


      def get_time_step(self):
            """
            Get the size of the current time step in seconds.

            Returns
            -------
            float
                  The size of the current time step in seconds.
            """
            return self.cppFoamLib.get_time_step()


      def get_number_of_nodes(self):
            """
            Get the number of nodes in the simulation.

            Returns
            -------
            int
                  The number of nodes in the simulation.
            """
            return self.cppFoamLib.get_number_of_nodes()


      def get_number_of_cells(self):
            """
            Get the total number of cells in the simulation.

            Returns
            -------
            int
                  The total number of cells in the simulation.
            """
            return self.cppFoamLib.get_number_of_cells()


      def get_number_of_boundary_nodes(self, patchName="numericalBoundary"):
            """
            Get the number of boundary nodes for a specified OpenFOAM patch.

            Parameters
            ----------
            patchName : str, optional
                  The name of the OpenFOAM boundary patch (default is "numericalBoundary").

            Returns
            -------
            int
                  The number of boundary nodes for the specified patch.
            """
            if isinstance(patchName, str):
                  patchName = patchName.encode('utf-8')

            return self.cppFoamLib.get_number_of_boundary_nodes(patchName)


      def get_number_of_boundary_faces(self, patchName="numericalBoundary"):
            """
            Get the number of boundary faces for a specified OpenFOAM patch.

            Parameters
            ----------
            patchName : str, optional
                  The name of the OpenFOAM boundary patch (default is "numericalBoundary").

            Returns
            -------
            int
                  The number of boundary faces for the specified patch.
            """
            if isinstance(patchName, str):
                  patchName = patchName.encode('utf-8')

            return self.cppFoamLib.get_number_of_boundary_faces(patchName)

      

      # Cell and Node Coordinate Access
      def get_node_coordinates(self):
            """
            Returns the coordinates of the nodes in the simulation.

            Returns
            -------
            ndarray (nNodes,)
                  Numpy array containing the coordinates of the nodes.
            """
            nNodes= self.cppFoamLib.get_number_of_nodes()

            cdef _np.ndarray[_np.float64_t, ndim=1] nodeCoordinates

            nodeCoordinates = _np.zeros(nNodes)
            self.cppFoamLib.get_node_coordinates(&nodeCoordinates[0])
      
            return nodeCoordinates
      
      
      def get_connectivity(self):
            """
            Returns the connectivity of the cells in the simulation.

            Returns
            -------
            ndarray (nCells * 8,)
                  Numpy array containing the connectivity of the cells.
            """
            nCells = self.cppFoamLib.get_number_of_cells()

            cdef _np.ndarray[_np.int64_t, ndim=1] connectivity

            connectivity = _np.zeros(nCells * 8, dtype=_np.float64)
            self.cppFoamLib.get_node_coordinates(<double*>connectivity.data)
      
            return connectivity
      
      def get_cell_volumes(self):
            """
            Returns the volumes of the cells in the simulation.

            Returns
            -------
            ndarray (nCells,)
                  Numpy array containing the volumes of the cells.
            """
            nCells = self.cppFoamLib.get_number_of_cells()

            cdef _np.ndarray[_np.float64_t, ndim=1] cellVolumes

            cellVolumes = _np.zeros(nCells)
            self.cppFoamLib.get_cell_volumes(&cellVolumes[0])
      
            return cellVolumes


      def get_cell_center_coordinates(self):
            """
            Returns the coordinates of the cell centers in the simulation.

            Returns
            -------
            ndarray (nCells, 3)
                  Numpy array containing the coordinates of the cell centers.
            """
            nCells = self.cppFoamLib.get_number_of_cells()

            cdef _np.ndarray[_np.float64_t, ndim=1] cellXYZ

            cellXYZ = _np.zeros(nCells*3)
            self.cppFoamLib.get_cell_center_coordinates(&cellXYZ[0])
            
            cellCenters = cellXYZ.reshape(nCells, 3)
            
            return cellCenters


      # Boundary Data Access
      def get_boundary_node_coordinates(self, patchName="numericalBoundary"):
            """
            Returns the coordinates of the boundary nodes for a given patch.

            Parameters
            ----------
            patchName : str, optional
                  Name of the OpenFOAM boundary patch. Default is "numericalBoundary".

            Returns
            -------
            ndarray (nBoundaryNodes, 3)
                  Numpy array containing the coordinates of the boundary nodes.
            """
            if isinstance(patchName, str):
                  patchName = patchName.encode('utf-8')

            nBoundaryNodes = self.cppFoamLib.get_number_of_boundary_nodes(patchName)

            cdef _np.ndarray[_np.float64_t, ndim=1] nodesXYZ

            nodesXYZ = _np.zeros(nBoundaryNodes*3)
            self.cppFoamLib.get_boundary_node_coordinates(&nodesXYZ[0], patchName)
            
            bondNodesCoordinates = nodesXYZ.reshape(nBoundaryNodes, 3)
            
            return bondNodesCoordinates
      
      def get_boundary_node_normal(self, patchName="numericalBoundary"):
            """
            Returns the normals of the boundary nodes for a given patch.

            Parameters
            ----------
            patchName : str, optional
                  Name of the OpenFOAM boundary patch. Default is "numericalBoundary".

            Returns
            -------
            ndarray (nBoundaryNodes, 3)
                  Numpy array containing the normals of the boundary nodes.
            """
            if isinstance(patchName, str):
                  patchName = patchName.encode('utf-8')
                  
            nBoundaryNodes = self.cppFoamLib.get_number_of_boundary_nodes(patchName)

            cdef _np.ndarray[_np.float64_t, ndim=1] bondNodeNormals

            bondNodeNormals = _np.zeros(nBoundaryNodes*3)
            self.cppFoamLib.get_boundary_face_normals(&bondNodeNormals[0], patchName)
            
            return bondNodeNormals.reshape(nBoundaryNodes, 3)
      
      
      def get_boundary_face_center_coordinates(self, patchName="numericalBoundary"):
            """
            Returns the coordinates of the boundary face centers for a given patch.

            Parameters
            ----------
            patchName : str, optional
                  Name of the OpenFOAM boundary patch. Default is "numericalBoundary".

            Returns
            -------
            ndarray (nBoundaryFaces, 3)
                  Numpy array containing the coordinates of the boundary face centers.
            """
            if isinstance(patchName, str):
                  patchName = patchName.encode('utf-8')

            nBoundaryFaces = self.cppFoamLib.get_number_of_boundary_faces(patchName)

            cdef _np.ndarray[_np.float64_t, ndim=1] faceXYZ

            faceXYZ = _np.zeros(nBoundaryFaces*3)
            self.cppFoamLib.get_boundary_face_center_coordinates(&faceXYZ[0], patchName)
            
            bondFaceCenterCoordinates = faceXYZ.reshape(nBoundaryFaces, 3)
            
            return bondFaceCenterCoordinates

      def get_boundary_face_areas(self, patchName="numericalBoundary"):
            """
            Returns the areas of the boundary faces for a given patch.

            Parameters
            ----------
            patchName : str, optional
                  Name of the OpenFOAM boundary patch. Default is "numericalBoundary".

            Returns
            -------
            ndarray (nBoundaryFaces,)
                  Numpy array containing the areas of the boundary faces.
            """
            if isinstance(patchName, str):
                  patchName = patchName.encode('utf-8')
                  
            nBoundaryFaces = self.cppFoamLib.get_number_of_boundary_faces(patchName)

            cdef _np.ndarray[_np.float64_t, ndim=1] boundaryFaceAreas

            boundaryFaceAreas = _np.zeros(nBoundaryFaces)
            self.cppFoamLib.get_boundary_face_areas(&boundaryFaceAreas[0], patchName)
            
            return boundaryFaceAreas

      def get_boundary_face_normals(self, patchName="numericalBoundary"):
            """
            Returns the normals of the boundary faces for a given patch.

            Parameters
            ----------
            patchName : str, optional
                  Name of the OpenFOAM boundary patch. Default is "numericalBoundary".

            Returns
            -------
            ndarray (nBoundaryFaces, 3)
                  Numpy array containing the normals of the boundary faces.
            """
            if isinstance(patchName, str):
                  patchName = patchName.encode('utf-8')
                  
            nBoundaryFaces = self.cppFoamLib.get_number_of_boundary_faces(patchName)

            cdef _np.ndarray[_np.float64_t, ndim=1] bondFaceNormals

            bondFaceNormals = _np.zeros(nBoundaryFaces*3)
            self.cppFoamLib.get_boundary_face_normals(&bondFaceNormals[0], patchName)
            
            return bondFaceNormals.reshape(nBoundaryFaces, 3)

      def get_boundary_cell_center_coordinates(self, patchName="numericalBoundary"):
            """
            Returns the coordinates of the boundary cell centers for a given patch.

            Parameters
            ----------
            patchName : str, optional
                  Name of the OpenFOAM boundary patch. Default is "numericalBoundary".

            Returns
            -------
            ndarray (nBoundaryFaces, 3)
                  Numpy array containing the coordinates of the boundary cell centers.
            """
            if isinstance(patchName, str):
                  patchName = patchName.encode('utf-8')

            nBoundaryFaces = self.cppFoamLib.get_number_of_boundary_faces(patchName)

            cdef _np.ndarray[_np.float64_t, ndim=1] cellXYZ

            cellXYZ = _np.zeros(nBoundaryFaces*3)
            self.cppFoamLib.get_boundary_cell_center_coordinates(&cellXYZ[0], patchName)
            
            bondCellCenterCoordinates = cellXYZ.reshape(nBoundaryFaces, 3)
            
            return bondCellCenterCoordinates


      # Field Access Methods
      def get_velocity_field(self):
            """
            Returns the velocity field as a NumPy array.

            Returns
            -------
            velocity : ndarray (3 * nCells,)
                  Velocity field as a flattened NumPy array (vx, vy, vz).
            """
            cdef _np.ndarray[_np.float64_t, ndim=1] velocity
            num_cells = self.get_number_of_cells()
            velocity = _np.zeros(3 * num_cells, dtype=_np.float64)
            self.cppFoamLib.get_velocity_field(&velocity[0])
            
            return velocity


      
      def get_velocity_boundary_field(self, patchName="numericalBoundary"):
            """
            Retrieve the velocity boundary field.

            Parameters
            ----------
            patchName : str, optional
                  Name of the OpenFOAM boundary patch (default: "numericalBoundary").

            Returns
            -------
            velocity : ndarray (3 * nBoundaryFaces,)
                  Velocity boundary field as a flattened NumPy array (vxb, vyb, vzb).
            """
            if isinstance(patchName, str):
                  patchName = patchName.encode('utf-8')
                  
            cdef _np.ndarray[_np.float64_t, ndim=1] velocity

            num_boundary_faces = self.get_number_of_boundary_faces(patchName)  # Assumes this method exists
            velocity = _np.zeros(3 * num_boundary_faces, dtype=_np.float64)
            self.cppFoamLib.get_velocity_boundary_field(&velocity[0], patchName)
            
            return velocity


      def get_pressure_field(self):
            """
            Returns the pressure field.

            Returns
            -------
            pressure : ndarray (nCells,)
                  Pressure field as a NumPy array.
            """
            cdef _np.ndarray[_np.float64_t, ndim=1] pressure
            
            num_cells = self.get_number_of_cells()
            pressure = _np.zeros(num_cells, dtype=_np.float64)
            self.cppFoamLib.get_pressure_field(&pressure[0])
            
            return pressure

      
      def get_velocity_gradient(self):
            """
            Retrieve the velocity gradient field.

            This version eliminates the need for passing an external ndarray as a parameter
            by managing the storage internally.

            Returns
            -------
            ndarray (9 * nCells,)
                  The velocity gradient field.
            """
            cdef _np.ndarray[_np.float64_t, ndim=1] velocity_gradient
            
            num_cells = self.get_number_of_cells()
            velocity_gradient = _np.zeros(num_cells * 9, dtype=_np.float64)

            # Fetch the velocity gradient field from the C++ library and store it in the array.
            self.cppFoamLib.get_velocity_gradient(&velocity_gradient[0])

            # Return the gradient to the caller
            return velocity_gradient


      def get_velocity_gradient_boundary_field(self, patchName="numericalBoundary"):
            """
            Retrieve the velocity gradient boundary field.

            Parameters
            ----------
            patchName : str, optional
                  Name of the OpenFOAM boundary patch (default: "numericalBoundary").

            Returns
            -------
            velocityGradient : ndarray (9 * nBoundaryFaces,)
                  Velocity gradient boundary field as a flattened NumPy array.
            """
            if isinstance(patchName, str):
                  patchName = patchName.encode('utf-8')
                  
            cdef _np.ndarray[_np.float64_t, ndim=1] velocity_gradient

            num_boundary_faces = self.get_number_of_boundary_faces(patchName)
            velocity_gradient =_np.zeros(9 * num_boundary_faces, dtype=_np.float64)
            self.cppFoamLib.get_velocity_gradient_boundary_field(&velocity_gradient[0], patchName)

            return velocity_gradient
      
      def get_pressure_gradient_field(self):
            """
            Returns the pressure gradient field.

            Returns
            -------
            pressureGradient : ndarray (3 * nCells,)
                  Pressure gradient field as a flattened NumPy array.
            """
            cdef _np.ndarray[_np.float64_t, ndim=1] pressure_gradient
            num_cells = self.get_number_of_cells()
            pressure_gradient =_np.zeros(3 * num_cells, dtype=_np.float64)
            self.cppFoamLib.get_pressure_gradient_field(&pressure_gradient[0])

            return pressure_gradient
      
      def get_pressure_boundary_field(self, patchName="numericalBoundary"):
            """
            Retrieve the pressure boundary field.

            Parameters
            ----------
            patchName : str, optional
                  Name of the OpenFOAM boundary patch (default: "numericalBoundary").

            Returns
            -------
            pressure : ndarray (nBoundaryFaces,)
                  Pressure boundary field as a NumPy array.
            """
            if isinstance(patchName, str):
                  patchName = patchName.encode('utf-8')

            cdef _np.ndarray[_np.float64_t, ndim=1] pressure
            num_boundary_faces = self.get_number_of_boundary_faces(patchName)
            pressure =_np.zeros(num_boundary_faces, dtype=_np.float64)
            self.cppFoamLib.get_pressure_boundary_field(&pressure[0], patchName)

            return pressure

      def get_pressure_gradient_boundary_field(self, patchName="numericalBoundary"):
            """
            Retrieve the pressure gradient boundary field.

            Parameters
            ----------
            patchName : str, optional
                  Name of the OpenFOAM boundary patch (default: "numericalBoundary").

            Returns
            -------
            pressureGradient : ndarray (3 * nBoundaryFaces,)
                  Pressure gradient boundary field as a flattened NumPy array.
            """
            if isinstance(patchName, str):
                  patchName = patchName.encode('utf-8')
                  
            cdef _np.ndarray[_np.float64_t, ndim=1] pressure_gradient
            num_boundary_faces = self.get_number_of_boundary_faces(patchName)
            pressure_gradient =_np.zeros(3 * num_boundary_faces, dtype=_np.float64)
            self.cppFoamLib.get_pressure_gradient_boundary_field(&pressure_gradient[0], patchName)

            return pressure_gradient
      
      
      def get_vorticity_field(self):
            """
            Retrieve the vorticity gradient field.

            This version eliminates the need for passing an external ndarray as a parameter
            by managing the storage internally.

            Returns
            -------
            ndarray (9 * nCells,)
                  The vorticity gradient field.
            """
            cdef _np.ndarray[_np.float64_t, ndim=1] vorticity
            
            num_cells = self.get_number_of_cells()
            vorticity = _np.zeros(num_cells * 3, dtype=_np.float64)

            # Fetch the vorticity gradient field from the C++ library and store it in the array.
            self.cppFoamLib.get_vorticity_field(&vorticity[0])

            # Return the gradient to the caller
            return vorticity


      def get_vorticity_boundary_field(self, patchName="numericalBoundary"):
            """
            Retrieve the vorticity boundary field.

            Parameters
            ----------
            patchName : str, optional
                  Name of the OpenFOAM boundary patch (default: "numericalBoundary").

            Returns
            -------
            vorticity : ndarray (3 * nBoundaryFaces,)
                  Velocity gradient boundary field as a flattened NumPy array.
            """
            if isinstance(patchName, str):
                  patchName = patchName.encode('utf-8')
                  
            cdef _np.ndarray[_np.float64_t, ndim=1] vorticity

            num_boundary_faces = self.get_number_of_boundary_faces(patchName)
            vorticity =_np.zeros(3 * num_boundary_faces, dtype=_np.float64)
            self.cppFoamLib.get_vorticity_boundary_field(&vorticity[0], patchName)

            return vorticity


      # ================================================== #
      # Set Methods (Boundary and Simulation Data)
      # ================================================== #

      def set_time_step(self, deltaT):
            """ 
            Set the desired time-step size.

            Parameters
            ----------
            deltaT : float
                  Desired time-step size in seconds.
            """
            cdef double timeStep
            timeStep = deltaT
            self.cppFoamLib.set_time_step(&timeStep)

      def set_dirichlet_velocity_boundary_condition(self, vxBoundary, vyBoundary, vzBoundary, patchName="numericalBoundary"):      
            """ 
            Set Dirichlet velocity boundary conditions.

            Parameters
            ----------
            vxBoundary : ndarray (3 * nBoundaryFaces,)
                  x-velocity component at the patch "numericalBoundary".
            vyBoundary : ndarray (3 * nBoundaryFaces,)
                  y-velocity component at the patch "numericalBoundary".
            vzBoundary : ndarray (3 * nBoundaryFaces,)
                  z-velocity component at the patch "numericalBoundary".
            patchName : str, optional
                  Name of the OpenFOAM boundary patch (default: "numericalBoundary").
            """
            if isinstance(patchName, str):
                  patchName = patchName.encode('utf-8')
            assert vxBoundary.shape == vyBoundary.shape
            assert vxBoundary.shape == vzBoundary.shape
            nx = vxBoundary.shape[0]
            ny = vyBoundary.shape[0]
            nz = vzBoundary.shape[0]
            vxBoundary = vxBoundary.reshape(nx, 1)
            vyBoundary = vyBoundary.reshape(ny, 1)
            vzBoundary = vzBoundary.reshape(nz, 1)
            cdef _np.ndarray[_np.float64_t, ndim=1] velocityBC
            velocityBC = _np.hstack((vxBoundary, vyBoundary, vzBoundary)).copy().ravel()
            self.cppFoamLib.set_dirichlet_velocity_boundary_condition(&velocityBC[0], patchName)

      def set_dirichlet_pressure_boundary_condition(self, pBoundary, patchName="numericalBoundary"):
            """
            Set Dirichlet pressure boundary conditions.

            Parameters
            ----------
            pBoundary : ndarray (nCells,)
                  Pressure field at the patch "numericalBoundary".
            patchName : str, optional
                  Name of the OpenFOAM boundary patch (default: "numericalBoundary").
            """
            patchName = patchName.encode()
            n = pBoundary.shape[0]
            pBoundary = pBoundary.reshape(n, 1)
            cdef _np.ndarray[_np.float64_t, ndim=1] pressureBC
            pressureBC = pBoundary.copy().ravel()
            self.cppFoamLib.set_dirichlet_pressure_boundary_condition(&pressureBC[0], patchName)

      def set_neumann_pressure_boundary_condition(self, dpdxBoundary, dpdyBoundary, dpdzBoundary, patchName="numericalBoundary"):
            """ 
            Set Neumann pressure boundary conditions.

            Parameters
            ----------
            dpdxBoundary : ndarray (nBoundaryFaces,)
                  dpdx at the patch "numericalBoundary".
            dpdyBoundary : ndarray (nBoundaryFaces,)
                  dpdy at the patch "numericalBoundary".
            dpdzBoundary : ndarray (nBoundaryFaces,)
                  dpdz at the patch "numericalBoundary".
            patchName : str, optional
                  Name of the OpenFOAM boundary patch (default: "numericalBoundary").
            """
            if isinstance(patchName, str):
                  patchName = patchName.encode('utf-8')
            assert dpdxBoundary.shape == dpdyBoundary.shape
            nx = dpdxBoundary.shape[0]
            ny = dpdyBoundary.shape[0]
            nz = dpdzBoundary.shape[0]
            dpdxBoundary = dpdxBoundary.reshape(nx, 1)
            dpdyBoundary = dpdyBoundary.reshape(ny, 1)
            dpdzBoundary = dpdzBoundary.reshape(nz, 1)
            cdef _np.ndarray[_np.float64_t, ndim=1] pressureGradientBC
            pressureGradientBC = _np.hstack((dpdxBoundary, dpdyBoundary, dpdzBoundary)).copy().ravel()
            self.cppFoamLib.set_neumann_pressure_boundary_condition(&pressureGradientBC[0], patchName)


      # ================================================= #
      # Python-based methods
      # ================================================= #

      def correct_mass_flux_python(self, faceVelocityX, faceVelocityY, faceVelocityZ, patchName="numericalBoundary"):
            """
            Correct the mass flux at each face along the numerical boundary.

            Parameters
            ----------
            faceVelocityX : ndarray (3 * nBoundaryFaces,)
                  x-component of the face velocity.
            faceVelocityY : ndarray (3 * nBoundaryFaces,)
                  y-component of the face velocity.
            faceVelocityZ : ndarray (3 * nBoundaryFaces,)
                  z-component of the face velocity.
            patchName : str, optional
                  Name of the OpenFOAM boundary patch (default: "numericalBoundary").

            Returns
            -------
            tuple of ndarrays (3 * nBoundaryFaces,)
                  Corrected face velocities (faceVelocityX2, faceVelocityY2, faceVelocityZ2).
            """
            # Area of each face along the numerical boundary
            faceArea = self.get_boundary_face_areas(patchName)

            # Normal vector of each face along the numerical boundary
            faceNormal = self.get_boundary_face_normals(patchName)

            # Ensure normal vector has unit length
            faceNormal = faceNormal / _np.linalg.norm(faceNormal, axis=1)[:, None]

            # Mass flux at each face along the numerical boundary
            faceFlux = (faceVelocityX * faceNormal[:, 0] + faceVelocityY * faceNormal[:, 1] + faceVelocityZ * faceNormal[:, 2]) * faceArea

            # Net flux along the numerical boundary
            totalFlux = _np.sum(faceFlux)

            # Total flux along the numerical boundary
            totalAbsFlux = _np.sum(_np.abs(faceFlux))

            # Correction for velocity component normal to each face along the numerical boundary
            faceVelocityCorr = (-(_np.abs(faceFlux) / (totalAbsFlux + 1.0e-15)) * totalFlux) / faceArea

            # Apply correction
            faceVelocityX2 = faceVelocityX + faceVelocityCorr * faceNormal[:, 0]
            faceVelocityY2 = faceVelocityY + faceVelocityCorr * faceNormal[:, 1]
            faceVelocityZ2 = faceVelocityZ + faceVelocityCorr * faceNormal[:, 2]

            return faceVelocityX2, faceVelocityY2, faceVelocityZ2


      def correct_normal_pressure_gradient(self, dpdx, dpdy, dpdz, patchName="numericalBoundary"):
            """
            Correct the normal pressure gradient at each face along the numerical boundary.

            Parameters
            ----------
            dpdx : ndarray (nBoundaryFaces,)
                  Pressure gradient in the x direction.
            dpdy : ndarray (nBoundaryFaces,)
                  Pressure gradient in the y direction.
            dpdz : ndarray (nBoundaryFaces,)
                  Pressure gradient in the z direction.
            patchName : str, optional
                  Name of the OpenFOAM boundary patch (default: "numericalBoundary").

            Returns
            -------
            tuple of ndarrays (nBoundaryFaces,)
                  Corrected pressure gradients (dpdx, dpdy, dpdz).
            """
            import numpy as np  # if not already imported as _np
            _np = np

            # If patchName is a str, encode it (needed for your C++ binding)
            if isinstance(patchName, str):
                  patchName = patchName.encode('utf-8')

            # Get the area of each boundary face.
            faceArea = self.get_boundary_face_areas(patchName)

            # Get the face normals.
            faceNormal = self.get_boundary_face_normals(patchName)
            
            # Ensure faceNormal is a 3D array. If only two components are provided,
            # extend to 3D by adding a zero for the z component.
            if faceNormal.shape[1] == 2:
                  faceNormal = _np.hstack([faceNormal, _np.zeros((faceNormal.shape[0], 1))])
            
            # Normalize the face normal vectors.
            faceNormal = faceNormal / _np.linalg.norm(faceNormal, axis=1)[:, None]

            # In 3D there is no unique tangential vector.
            # We define one by projecting a chosen global reference vector onto the tangent plane.
            # Choose [0, 0, 1] as the primary reference vector.
            global_ref = _np.array([0.0, 0.0, 1.0])
            alternative_ref = _np.array([1.0, 0.0, 0.0])  # use if face normal is parallel to global_ref

            # Build a unique tangential vector for each face.
            faceTangent = []
            for i in range(faceNormal.shape[0]):
                  # Check if the face normal is nearly parallel to the global reference.
                  if abs(_np.dot(faceNormal[i], global_ref)) > 0.99:
                        ref = alternative_ref
                  else:
                        ref = global_ref
                  # Project the reference vector onto the plane tangent to faceNormal.
                  tangent = ref - _np.dot(ref, faceNormal[i]) * faceNormal[i]
                  tangent = tangent / _np.linalg.norm(tangent)
                  faceTangent.append(tangent)
            faceTangent = _np.array(faceTangent)

            # Compute the normal pressure gradient at each face.
            dpdn = dpdx * faceNormal[:, 0] + dpdy * faceNormal[:, 1] + dpdz * faceNormal[:, 2]
            dpdn_x = dpdn * faceNormal[:, 0]
            dpdn_y = dpdn * faceNormal[:, 1]
            dpdn_z = dpdn * faceNormal[:, 2]

            # Compute the tangential components (difference between full gradient and its normal projection).
            dpdt_x = dpdx - dpdn_x
            dpdt_y = dpdy - dpdn_y
            dpdt_z = dpdz - dpdn_z
            # dpdt (magnitude of the tangential gradient) is computed for informational purposes.
            dpdt = _np.sqrt(dpdt_x**2 + dpdt_y**2 + dpdt_z**2)

            # Compute the tangential pressure difference along the boundary.
            # Here we take the dot product of the tangential component with our defined tangent vector.
            faceDeltaP = (dpdt_x * faceTangent[:, 0] +
                              dpdt_y * faceTangent[:, 1] +
                              dpdt_z * faceTangent[:, 2]) * faceArea

            # Calculate net and total absolute pressure differences along the boundary.
            totalDeltaP = _np.sum(faceDeltaP)
            totalAbsDeltaP = _np.sum(_np.abs(faceDeltaP))

            # Compute the correction for the velocity component normal to each face.
            facePressureGradientCorr = (-(_np.abs(faceDeltaP) / totalAbsDeltaP) * totalDeltaP) / faceArea

            # Apply the correction along the face normal direction.
            dpdx += facePressureGradientCorr * faceNormal[:, 0]
            dpdy += facePressureGradientCorr * faceNormal[:, 1]
            dpdz += facePressureGradientCorr * faceNormal[:, 2]

            return dpdx, dpdy, dpdz



      def get_total_circulation(self):
            """
            Get the total circulation in the finite volume mesh region.

            Returns
            -------
            float
                  Total circulation within the finite volume mesh region (m²/s).
            """
            gx, gy, gz = self.get_finite_volume_mesh_circulation()
            circulation = _np.sqrt(gx ** 2 + gy ** 2 + gz ** 2).sum()

            return circulation


      def get_mesh_centroid(self):
            """
            Calculate the centroid of the mesh.

            Returns
            -------
            ndarray
                  Coordinates of the mesh centroid (3,).
            """
            cellVolume = self.get_cell_volumes()
            cellCoordinates = self.get_cell_center_coordinates()

            cellVolumeSum = cellVolume.sum()

            centroid = _np.dot(cellVolume, cellCoordinates) / cellVolumeSum

            return centroid
