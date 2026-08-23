"""The VPM owns molecular kinematic_viscosity in a coupled VLM+VPM run; mismatch must fail loudly."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from source.solvers.vpm.core.solver import VPMSolver
from source.solvers.vpm.io.logging import Logging


def test_mismatched_vlm_viscosity_is_rejected():
    viscous = SimpleNamespace(scheme="CS", kinematic_viscosity=1.5e-5)
    vlm = SimpleNamespace(kinematic_viscosity=1.0e-2)
    with pytest.raises(ValueError, match="kinematic_viscosity mismatch"):
        VPMSolver._require_consistent_molecular_viscosity(viscous, vlm)


def test_inviscid_vpm_requires_zero_vlm_viscosity():
    viscous = SimpleNamespace(scheme="NONE")
    vlm = SimpleNamespace(kinematic_viscosity=1.0e-5)
    with pytest.raises(ValueError, match="kinematic_viscosity mismatch"):
        VPMSolver._require_consistent_molecular_viscosity(viscous, vlm)


def test_viscous_vpm_requires_an_explicit_molecular_viscosity():
    viscous = SimpleNamespace(scheme="CS", kinematic_viscosity=None)
    vlm = SimpleNamespace(kinematic_viscosity=1.0e-5)
    with pytest.raises(ValueError, match="requires kinematic_viscosity"):
        VPMSolver._require_consistent_molecular_viscosity(viscous, vlm)


def test_matching_viscosities_are_accepted():
    viscous = SimpleNamespace(scheme="CS", kinematic_viscosity=1.5e-5)
    vlm = SimpleNamespace(kinematic_viscosity=1.5e-5)
    VPMSolver._require_consistent_molecular_viscosity(viscous, vlm)


def test_vlm_log_uses_canonical_kinematic_viscosity_field():
    vlm = SimpleNamespace(
        lattice=SimpleNamespace(n_panels=12),
        max_n_panels=16,
        surfaces={},
        dtype="f32",
        linear_solver="SCIPY",
        density=1.225,
        kinematic_viscosity=1.5e-5,
        force=SimpleNamespace(method="KUTTA_JOUKOWSKI"),
    )
    lines = Logging._format_vlm_mesh_lines(vlm)
    assert "  Kinematic Viscosity      : 1.500e-05 m²/s" in lines
