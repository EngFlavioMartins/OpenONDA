# This module provides a portable way of using operating system dependent functionality.
import os

# The subprocess module allows you to spawn new processes, connect to their input/output/error pipes, and obtain their return codes. This module intends to replace several older modules and functions:
import subprocess

# This is a recursive acronym for “YAML Ain’t Markup Language,” is a human-readable data serialization language. It is often used for configuration files but also for data exchange. 
import yaml

# This module provides regular expression matching operations similar to those found in Perl.
import re

def set_initial_condition(OpenFoamCaseDir, 
                          vx0=None, vy0=None, vz0=None, p=None, 
                          vxBoundary=None, vyBoundary=None, vzBoundary=None, pBoundary=None):
      """
      Sets the initial internal and boundary conditions for an OpenFOAM case.

      Parameters
      ----------
      OpenFoamCaseDir : str
            Directory of the OpenFOAM case.
      vx0, vy0, vz0 : ndarray of shape (numCells,), optional
            Velocity components in the x, y, and z directions for internal cells (m/s).
      p : ndarray of shape (numCells,), optional
            Pressure-over-density for internal cells (m^2/s^2).
      vxBoundary, vyBoundary, vzBoundary : ndarray of shape (numBoundaryFaces,), optional
            Velocity components in the x, y, and z directions for boundary faces (m/s).
      pBoundary : ndarray of shape (numBoundaryFaces,), optional
            Pressure-over-density for boundary faces (m^2/s^2).
      """

      # Internal field conditions
      if vx0 is not None and vy0 is not None and vz0 is not None:
            numCells = len(vx0)
            fileName = os.path.join(OpenFoamCaseDir, "0", "internalFieldU")
            with open(fileName, 'w') as file:
                  file.write(f"nonuniform List<vector>\n{numCells}\n(\n")
                  for i in range(numCells):
                        file.write(f"\t({vx0[i]} {vy0[i]} {vz0[i]})\n")
                  file.write(");\n")

      if p is not None:
            numCells = len(p)
            fileName = os.path.join(OpenFoamCaseDir, "0", "internalFieldP")
            with open(fileName, 'w') as file:
                  file.write(f"nonuniform List<scalar>\n{numCells}\n(\n")
                  for i in range(numCells):
                        file.write(f"\t{p[i]}\n")
                  file.write(");\n")

      # Boundary field conditions
      if vxBoundary is not None and vyBoundary is not None and vzBoundary is not None:
            numBoundaryFaces = len(vxBoundary)
            fileName = os.path.join(OpenFoamCaseDir, "0", "boundaryFieldU")
            with open(fileName, 'w') as file:
                  file.write(f"nonuniform List<vector>\n{numBoundaryFaces}\n(\n")
                  for i in range(numBoundaryFaces):
                        file.write(f"\t({vxBoundary[i]} {vyBoundary[i]} {vzBoundary[i]})\n")
                  file.write(");\n")

      if pBoundary is not None:
            numBoundaryFaces = len(pBoundary)
            fileName = os.path.join(OpenFoamCaseDir, "0", "boundaryFieldP")
            with open(fileName, 'w') as file:
                  file.write(f"nonuniform List<scalar>\n{numBoundaryFaces}\n(\n")
                  for i in range(numBoundaryFaces):
                        file.write(f"\t{pBoundary[i]}\n")
                  file.write(");\n")

      
      
def set_eulerian_module(currentDir, OpenFoamCaseDir):
      """
      Useful script to prepare the OpenFOAM simulation folder:
      
      Parameters:
      -----------
            currentDir: path
                  Current working directory
            OpenFoamCaseDir: path
                  OpenFOAM case directory relatively to currentDir
      """

      #Config the yaml file
      loader = yaml.SafeLoader

      loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))
      
      # -----------

      # Execute code that cleans up the OpenFOAM's directory from previous results:
      cmd = "{}".format(os.path.join(OpenFoamCaseDir, "Allclean.sh"))
      result = subprocess.run(cmd, text=True, capture_output=True)

      print(result.stdout)  # Output of the command
      print(result.stderr)  # Error message, if any

      # Copy backup folders:
      cmd = "cp -r {} {}".format(os.path.join(OpenFoamCaseDir, "0.orig"), os.path.join(OpenFoamCaseDir, "0"))
      
      # Execute
      os.system(cmd)

      # Copy backup folders:
      cmd = "cp -r {} {}".format(os.path.join(OpenFoamCaseDir, "constant/polyMesh.orig"), os.path.join(OpenFoamCaseDir, "constant/polyMesh"))

      # Execute
      os.system(cmd)



def set_boundary_conditions(solver, vx, vy, vz, p, patchName="numericalBoundary"):
      ''' 
      We use it here to determine the Dirichlet boundary condition of an OpenFOAM solver.

      Parameters:
      -----------
            vortexRingRadius: float
                  Radius of the vortex ring, m
            vx, vy, vz: (numCells,) ndarray numpy array  of floaters
                  Velocity components calculated at the cell centers, m/s
            p: (numCells,) ndarray numpy array of floaters
                  Pressure over density, m2/s2

      Returns: 
      -----------
            -----
      '''
      
      # Set pressure boundary condition:
      solver.set_dirichlet_pressure_boundary_condition(p, patchName=patchName)
      
      
      # Correct the velocity at the boundary so that m_dot = 0
      vxBoundaryCorr, vyBoundaryCorr, vzBoundaryCorr = solver.correct_mass_flux_python(vx, vy, vz, patchName=patchName)
      
      
      solver.set_dirichlet_velocity_boundary_condition(vxBoundaryCorr, vyBoundaryCorr, vzBoundaryCorr, patchName=patchName)
      
      
      return 0