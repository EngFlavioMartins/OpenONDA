"""Shared VPM test fixtures and CLI options."""

import pytest

from source.solvers.VPM import Solver, SolverConfig
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

# Ordered: CPU first (most reliable), then CUDA, then Vulkan.
# Tests that fail on a backend are warned and skipped via the
# ``solver_for_backend`` fixture.
BACKENDS = ["CPU", "CUDA", "VULKAN"]


def pytest_generate_tests(metafunc):
    """Auto-parametrise any test that declares a ``backend`` argument."""
    if "backend" in metafunc.fixturenames:
        metafunc.parametrize("backend", BACKENDS, scope="function")


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="function")
def solver_for_backend(tmp_path, backend):
    """
    Factory fixture: returns a callable ``make_solver(**kwargs)`` that
    creates a :class:`Solver` initialised on the requested ``backend``.

    ``reset_taichi_backend()`` is called before *and* after the test so
    that switching between CPU / CUDA / Vulkan in the same pytest process
    does not leak GPU memory or stale Taichi kernels.
    """
    reset_taichi_backend()

    def _make_solver(**kwargs):
        config = SolverConfig(
            processing_unit=backend,
            backup_directory=str(tmp_path),
            backup_frequency=0,
            logging_frequency=0,
            **kwargs,
        )
        return Solver(config=config)

    yield _make_solver
    reset_taichi_backend()


@pytest.fixture(scope="function")
def minimal_solver_config(tmp_path):
    """Return a minimal ``SolverConfig`` that disables all I/O and physics."""
    return dict(
        time_step_size=0.01,
        processing_unit="CPU",
        stretching=StretchingConfig.disabled(),
        viscous=ViscousConfig(scheme="NONE"),
        advection=AdvectionConfig(scheme="NONE"),
        backup_frequency=0,
        logging_frequency=0,
        backup_directory=str(tmp_path),
    )
