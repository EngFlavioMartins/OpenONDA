"""Repository-wide scientific test taxonomy.

Every collected test receives exactly one primary tier.  Cross-cutting markers
(``slow``, ``gpu``, and ``stochastic``) remain additive.  Keeping the mapping
here makes CI selection inspectable without scattering path rules through the
individual numerical tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

QUALIFICATION_MODULES = frozenset(
    {
        "test_cylinder_reference_tools.py",
        "test_cube_reference_grid_study.py",
        "test_flux_handoff_vpm_integration.py",
        "test_gbd_projected_renewal.py",
        "test_interpolation_qualification.py",
        "test_core_numerical_qualification.py",
        "test_manufactured_gradient_qualification.py",
        "test_panel_linear_solver_convergence.py",
        "test_panel_moving_qualification.py",
        "test_panel_solver_sphere_analytic.py",
    }
)

INTEGRATION_MODULES = frozenset(
    {
        "test_common_m4_viscous_lifecycle.py",
        "test_coupled_backup.py",
        "test_flux_handoff.py",
        "test_fvm_vpm_smoke.py",
        "test_lattice_transfer.py",
        "test_panel_multibody.py",
        "test_panel_particle_coupling.py",
        "test_physical_coupling.py",
        "test_stable_renewal.py",
    }
)

SLOW_MODULES = QUALIFICATION_MODULES | frozenset(
    {
        "test_backup_storage.py",
        "test_common_m4_viscous_lifecycle.py",
        "test_panel_multibody.py",
    }
)

_qualification_reports: list[dict[str, Any]] = []


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the explicit numerical-evidence report destination."""
    parser.addoption(
        "--numerical-report",
        metavar="PATH",
        help="write qualification outcomes and recorded numerical metrics as JSON",
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Apply the declared primary tier and orthogonal execution properties."""
    for item in items:
        path = Path(str(item.path))
        module = path.name
        if "tutorials" in path.parts:
            item.add_marker(pytest.mark.tutorial)
        elif module in QUALIFICATION_MODULES:
            item.add_marker(pytest.mark.qualification)
        elif module in INTEGRATION_MODULES:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)

        if module in SLOW_MODULES:
            item.add_marker(pytest.mark.slow)
        node_name = item.nodeid.lower()
        if "rwm" in node_name or "turbulence" in node_name:
            item.add_marker(pytest.mark.stochastic)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    """Collect qualification outcomes plus metrics supplied with record_property."""
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or item.get_closest_marker("qualification") is None:
        return
    metrics = {
        name: value
        for name, value in report.user_properties
        if isinstance(value, str | int | float | bool | type(None))
    }
    _qualification_reports.append(
        {
            "nodeid": report.nodeid,
            "outcome": report.outcome,
            "duration_seconds": report.duration,
            "metrics": metrics,
        }
    )


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Write a deterministic qualification artifact when explicitly requested."""
    destination = session.config.getoption("--numerical-report")
    if destination is None:
        return
    report_path = Path(destination)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "schema": "openonda-numerical-qualification-v1",
                "tests": sorted(_qualification_reports, key=lambda entry: entry["nodeid"]),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
