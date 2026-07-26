"""Shared VPM test fixtures and CLI options."""

import platform

import pytest

from source.solvers.VPM import Solver, VPMSetup
from source.solvers.VPM.config.backend import reset_taichi_backend
from source.solvers.VPM.config.types import (
    AdvectionConfig,
    StretchingConfig,
    ViscousConfig,
)

# ── CLI options ──────────────────────────────────────────────────────────────


def pytest_addoption(parser):
    parser.addoption(
        "--snapshot-update",
        action="store_true",
        default=False,
        help="Update regression snapshot golden values",
    )


# ── Backend parametrisation ──────────────────────────────────────────────────

# Exercise the native GPU API for the host platform. Other requested backends
# are already covered by backend-chain unit tests.
BACKENDS = ["CPU", "METAL"] if platform.system() == "Darwin" else ["CPU", "CUDA", "VULKAN"]


def pytest_generate_tests(metafunc):
    """Auto-parametrise any test that declares a ``backend`` argument."""
    if "backend" in metafunc.fixturenames:
        metafunc.parametrize("backend", BACKENDS, scope="function")


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="function")
def solver_for_backend(tmp_path, backend):
    """Return a solver factory for one requested backend."""
    reset_taichi_backend()

    def _make_solver(**kwargs):
        config = VPMSetup(
            processing_unit=backend,
            backup_directory=str(tmp_path),
            backup_frequency=0,
            logging_frequency=0,
            **kwargs,
        )
        solver = Solver(setup=config)
        if solver.processing_unit != backend:
            pytest.skip(f"{backend} unavailable; Taichi initialized {solver.processing_unit}")
        return solver

    yield _make_solver
    reset_taichi_backend()


@pytest.fixture(scope="function")
def minimal_solver_config(tmp_path):
    """Return a minimal ``VPMSetup`` that disables all I/O and physics."""
    return {
        "time_step_size": 0.01,
        "processing_unit": "CPU",
        "stretching": StretchingConfig.disabled(),
        "viscous": ViscousConfig(scheme="NONE"),
        "advection": AdvectionConfig(scheme="NONE"),
        "backup_frequency": 0,
        "logging_frequency": 0,
        "backup_directory": str(tmp_path),
    }
