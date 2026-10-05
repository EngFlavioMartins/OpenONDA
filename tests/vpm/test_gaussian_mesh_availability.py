"""Optional dependency validation without a GPU or optional runtime import."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import taichi as ti

from source.solvers.vpm.kernels.base import make_vortex_kernel
from source.solvers.vpm.numerics import ieee
from source.solvers.vpm.physics.induction import slip_slab
from source.solvers.vpm.physics.induction.direct import DirectInduction
from source.solvers.vpm.physics.induction.gaussian_mesh import availability
from source.solvers.vpm.physics.induction.gaussian_mesh.session import GaussianSlabSettings
from source.solvers.vpm.physics.induction.gaussian_tail import _interval


@pytest.fixture
def environment(monkeypatch, tmp_path):
    events = []
    bridge = SimpleNamespace(
        __file__="/installed/site-packages/source/solvers/vpm/numerics/_fenv.so"
    )
    monkeypatch.setattr(ieee, "_bridge", lambda: bridge)
    monkeypatch.setattr(ieee, "require_round_to_nearest", lambda: events.append("rounding"))

    @contextmanager
    def restoring_scope():
        events.append("capture")
        try:
            yield
        finally:
            events.append("restore")

    monkeypatch.setattr(ieee, "ieee_arithmetic", restoring_scope)
    monkeypatch.setattr(_interval, "_platform", lambda: events.append("ieee-check"))
    runtime = SimpleNamespace(
        runtimeGetVersion=lambda: 12000, driverGetVersion=lambda: 12000, getDeviceCount=lambda: 1
    )
    cp = SimpleNamespace(__version__="14.2.0", cuda=SimpleNamespace(runtime=runtime))
    headers = tmp_path / "toolkit/include"
    headers.mkdir(parents=True)
    (headers / "cuda_runtime.h").write_text("// Synthetic header validation fixture\n")

    def find_headers(library):
        assert library == "cudart"
        events.append("headers")
        return str(headers)

    modules = {
        "cupy": cp,
        "cupy_backends.cuda.libs.nvrtc": SimpleNamespace(getVersion=lambda: (12, 0)),
        "cupy.cuda.cufft": SimpleNamespace(PlanNd=lambda *args: None),
        "cuda.pathfinder": SimpleNamespace(find_nvidia_header_directory=find_headers),
    }

    def load(name):
        events.append(("import", name))
        if name not in modules:
            raise ImportError(name)
        return modules[name]

    monkeypatch.setattr(availability, "import_module", load)
    return modules, events


def test_validation_preserves_environment_and_never_allocates_or_selects(environment):
    _, events = environment
    actual = availability.require_gaussian_mesh_runtime()
    assert events[:4] == ["rounding", "capture", "ieee-check", "restore"]
    assert actual["cupy_version"] == "14.2.0"
    assert actual["cuda_runtime_version"] == 12000
    assert actual["cuda_header_directory"].endswith("toolkit/include")
    assert actual["fenv_extension"].startswith("/installed/site-packages/")


def test_missing_cupy_is_explicit_no_fallback(environment):
    modules, _ = environment
    del modules["cupy"]
    with pytest.raises(availability.GaussianMeshUnavailableError, match="optional CuPy/CUDA"):
        availability.require_gaussian_mesh_runtime()


@pytest.mark.parametrize("fault", ["missing_module", "missing_api", "not_found", "stale_path"])
def test_missing_headers_reject_before_cuda_queries_without_environment_changes(
    environment, monkeypatch, tmp_path, fault
):
    import os

    modules, _ = environment
    monkeypatch.setenv("CUDA_PATH", "caller-selected-prefix")
    before = dict(os.environ)
    modules["cupy"].cuda = None  # Header validation must precede even runtime queries.
    if fault == "missing_module":
        del modules["cuda.pathfinder"]
    elif fault == "missing_api":
        modules["cuda.pathfinder"] = SimpleNamespace()
    else:
        result = None if fault == "not_found" else str(tmp_path / "removed-headers")
        modules["cuda.pathfinder"].find_nvidia_header_directory = lambda library: result
    with pytest.raises(availability.GaussianMeshUnavailableError, match="CUDA_PATH"):
        availability.require_gaussian_mesh_runtime()
    assert dict(os.environ) == before


@pytest.mark.parametrize("version", ["13.6.0", "14.1.0", "15.0.0", "unknown"])
def test_unqualified_cupy_rejected_before_cuda_calls(environment, version):
    modules, events = environment
    modules["cupy"].__version__ = version
    modules["cupy"].cuda = None
    with pytest.raises(availability.GaussianMeshUnavailableError, match="CuPy >=14.2,<15"):
        availability.require_gaussian_mesh_runtime()
    assert events[-1] == ("import", "cupy")


@pytest.mark.parametrize("fault", ["runtime", "nvrtc", "devices", "plans"])
def test_incomplete_cuda_installation_rejected(environment, fault):
    modules, _ = environment
    if fault == "runtime":
        modules["cupy"].cuda.runtime.runtimeGetVersion = lambda: 13000
    elif fault == "nvrtc":
        modules["cupy_backends.cuda.libs.nvrtc"].getVersion = lambda: (13, 0)
    elif fault == "devices":
        modules["cupy"].cuda.runtime.getDeviceCount = lambda: 0
    else:
        modules["cupy.cuda.cufft"].PlanNd = None
    with pytest.raises(availability.GaussianMeshUnavailableError, match="No fallback"):
        availability.require_gaussian_mesh_runtime()


def test_missing_guard_precedes_any_optional_gpu_import(environment, monkeypatch):
    _, events = environment

    def missing():
        raise ieee.IEEEEnvironmentUnavailableError("not installed")

    monkeypatch.setattr(ieee, "_bridge", missing)
    with pytest.raises(availability.GaussianMeshUnavailableError, match="installed _fenv"):
        availability.require_gaussian_mesh_runtime()
    assert events == []


def test_failed_ieee_probe_restores_callers_state(environment, monkeypatch):
    _, events = environment

    def fail():
        raise RuntimeError("unsupported IEEE environment")

    monkeypatch.setattr(_interval, "_platform", fail)
    with pytest.raises(availability.GaussianMeshUnavailableError, match="IEEE"):
        availability.require_gaussian_mesh_runtime()
    assert events == ["rounding", "capture", "restore"]


def test_dependency_failure_precedes_rebind_existing_field_close_and_field_allocation(monkeypatch):
    base = DirectInduction()
    slab = slip_slab.SlipSlabInduction(
        base,
        z_min=-0.5,
        z_max=0.5,
        gaussian_mesh_settings=GaussianSlabSettings(backend="cupy_cuda"),
    )
    existing = SimpleNamespace(close=lambda: pytest.fail("old field must remain untouched"))
    slab._mesh_session = existing
    old_binding = slab._mesh_binding
    monkeypatch.setattr(slip_slab, "_mesh_runtime_is_cuda", lambda: True)
    monkeypatch.setattr(base, "bind", lambda *args, **kwargs: pytest.fail("no base mutation"))

    def missing():
        raise availability.GaussianMeshUnavailableError("dependency absent")

    monkeypatch.setattr(slip_slab, "_check_cuda_mesh_installation", missing)
    physics = SimpleNamespace(particle_kernel="GAUSSIAN", accumulator_dtype=ti.f32)
    with pytest.raises(availability.GaussianMeshUnavailableError, match="dependency absent"):
        slab.bind(physics, kernel=make_vortex_kernel("GAUSSIAN"))
    assert slab._mesh_session is existing and slab._mesh_binding is old_binding
