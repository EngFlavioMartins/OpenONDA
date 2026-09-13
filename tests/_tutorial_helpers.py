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
