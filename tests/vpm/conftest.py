"""Shared VPM test fixtures and CLI options."""

import functools
import platform

import pytest

from source.solvers.vpm import VPMSetup, VPMSolver
from source.solvers.vpm.config.types import (
    AdvectionConfig,
    StretchingConfig,
    ViscousConfig,
)
from source.solvers.vpm.runtime.backend import initialize_taichi_backend, reset_taichi_backend

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


@functools.cache
def _backend_available(name: str) -> bool:
    """Probe once per session whether Taichi can initialise ``name`` here.

    An explicit GPU request is strict: ``initialize_taichi_backend`` raises
    rather than falling back, so a missing device has to be detected before the
    solver is constructed or the test reports a failure instead of a skip.
    """
    if name == "CPU":
        return True
    try:
        reset_taichi_backend()
        initialize_taichi_backend(preferred_backend=name)
        return True
    except Exception:
        return False
    finally:
        reset_taichi_backend()


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _skip_unavailable_backend(request):
    """Skip a backend-parametrised test when that backend is absent."""
    callspec = getattr(request.node, "callspec", None)
    backend = callspec.params.get("backend") if callspec else None
    if backend is not None and not _backend_available(backend):
        pytest.skip(f"{backend} backend unavailable on this machine")


@pytest.fixture(scope="function")
def solver_for_backend(tmp_path, backend):
    """Return a solver factory for one requested backend."""
    reset_taichi_backend()

    def _make_solver(**kwargs):
        config = VPMSetup(
            compute_device=backend,
            checkpoint_directory=str(tmp_path),
            checkpoint_interval_steps=0,
            logging_interval_steps=0,
            **kwargs,
        )
        solver = VPMSolver(setup=config)
        if solver.compute_device != backend:
            pytest.skip(f"{backend} unavailable; Taichi initialized {solver.compute_device}")
        return solver

    yield _make_solver
    reset_taichi_backend()


@pytest.fixture(scope="function")
def minimal_solver_config(tmp_path):
    """Return a minimal ``VPMSetup`` that disables all I/O and physics."""
    return {
        "time_step_size": 0.01,
        "compute_device": "CPU",
        "stretching": StretchingConfig.disabled(),
        "viscous": ViscousConfig(scheme="NONE"),
        "advection": AdvectionConfig(scheme="NONE"),
        "checkpoint_interval_steps": 0,
        "logging_interval_steps": 0,
        "checkpoint_directory": str(tmp_path),
    }
