"""Resolve tutorial modules through the installed loader contract.

The numbered tutorial folders (``04_flat_plate``, ``06_rotor_flow_PENDING``, ...)
are organizational names that cannot appear in ordinary Python imports.  Test
modules resolve the stable catalog identifier with
:func:`openonda.tutorials.get_tutorial` and import the tutorial's own modules
with :func:`openonda.tutorial_runner.load_case_module`, the same contract the
installed CLI and case launchers use.  No test ever imports a numeric folder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from openonda.tutorial_runner import load_case_module
from openonda.tutorials import get_tutorial

_TUTORIALS = Path(__file__).resolve().parents[1] / "tutorials"


def tutorial_directory(name: str) -> Path:
    """Return the physical tutorial directory for a catalog name."""
    return _TUTORIALS.joinpath(get_tutorial(name).relative_path)


def load_tutorial_module(name: str, module: str = "setup"):
    """Import a tutorial module through the loader contract.

    ``name`` is a catalog identifier such as ``vpm/flat_plate`` or
    ``coupled_fvm_vpm/cylinder_shedding_flow/reference_flow``; ``module`` is the
    module below the tutorial directory (``setup`` by default).
    """
    return load_case_module(tutorial_directory(name), module)


def write_vtu_time_frame(path: Path, time: float) -> Path:
    """Write a minimal raw-appended VTU particle frame carrying the clock.

    The fixture mirrors the solver visualization companion that ParaView peers
    read through ``vpm.pvd``: points are empty and the only meaningful payload
    is the exact ``time``/``TimeValue`` field-data pair used by the audit clock.
    """
    from vtk import vtkPoints, vtkUnstructuredGrid, vtkXMLUnstructuredGridWriter
    from vtk.util.numpy_support import numpy_to_vtk

    points = vtkPoints()
    points.SetData(numpy_to_vtk(np.zeros((0, 3), dtype=np.float32), deep=True))
    grid = vtkUnstructuredGrid()
    grid.SetPoints(points)
    for name in ("time", "TimeValue"):
        array = numpy_to_vtk(np.array([time], dtype=np.float64), deep=True)
        array.SetName(name)
        grid.GetFieldData().AddArray(array)
    writer = vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(Path(path)))
    writer.SetInputData(grid)
    writer.SetDataModeToAppended()
    writer.EncodeAppendedDataOff()
    writer.Write()
    return Path(path)
