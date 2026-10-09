"""Locate tutorial cases and import their modules as their launchers run them.

A tutorial launcher runs ``python setup.py`` or ``python -m assets.<module>``
from the case directory, so the case directory is the first import root and
``assets`` is the case's own package. Every case has an ``assets`` package,
hence loading a module first drops the previously loaded case's ``setup`` and
``assets`` modules and import root.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import re
import shutil
import sys

import numpy as np

TUTORIALS = Path(__file__).resolve().parents[1] / "tutorials"
GENERATED = (
    "solution",
    "solutions",
    "samples",
    "figures",
    "study_results",
    "sensitivity",
    "__pycache__",
)
_case_roots: list[str] = []


def tutorial_directory(name: str) -> Path:
    """Return a case directory from its name without the numbered folder prefix.

    ``vpm/flat_plate`` resolves ``tutorials/vpm/04_flat_plate``; nested cases
    such as ``coupled_fvm_vpm/cube_flow/reference_flow`` resolve part by part.
    """
    directory = TUTORIALS
    for part in name.strip("/").split("/"):
        matches = [
            child
            for child in directory.iterdir()
            if child.is_dir() and re.fullmatch(rf"(?:\d+_)?{re.escape(part)}", child.name)
        ]
        if len(matches) != 1:
            raise ValueError(f"Unknown tutorial {name!r}")
        directory = matches[0]
    return directory


def tutorial_cases() -> list[Path]:
    """Return every tutorial case directory, identified by its ``setup.py``."""
    return sorted(
        path.parent
        for path in TUTORIALS.rglob("setup.py")
        if not set(GENERATED).intersection(path.relative_to(TUTORIALS).parts)
    )


def copy_tutorial(name: str | Path, destination: Path) -> Path:
    """Copy a case's inputs, without generated output, to ``destination``."""
    source = name if isinstance(name, Path) else tutorial_directory(name)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(*GENERATED, "results"))
    return destination


def load_case_module(directory: str | Path, module: str = "setup"):
    """Import ``module`` of a case directory as its launchers would."""
    root = str(Path(directory).resolve())
    for loaded in [
        name for name in sys.modules if name == "setup" or name.split(".")[0] == "assets"
    ]:
        del sys.modules[loaded]
    while _case_roots:
        previous = _case_roots.pop()
        if previous in sys.path:
            sys.path.remove(previous)
    sys.path.insert(0, root)
    _case_roots.append(root)
    return importlib.import_module(module)


def load_tutorial_module(name: str, module: str = "setup"):
    """Import a module of a tutorial case, e.g. ``("vpm/flat_plate", "assets.results")``."""
    return load_case_module(tutorial_directory(name), module)


def write_vtu_time_frame(path: Path, time: float) -> Path:
    """Write a minimal raw-appended VTU particle frame carrying the clock.

    The fixture mirrors the solver visualization companion that ParaView peers
    read through ``vpm.pvd``: points are empty and the only meaningful data
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
