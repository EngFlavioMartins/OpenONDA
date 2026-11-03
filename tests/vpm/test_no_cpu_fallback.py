"""A run meant for a device stops with a message instead of moving to the CPU."""

from types import SimpleNamespace

import pytest

from openonda.vpm.physics.induction import slip_slab
from openonda.vpm.physics.induction.gaussian_mesh.availability import (
    GaussianMeshUnavailableError,
)
from openonda.vpm.physics.induction.gaussian_mesh.execution import PortableGaussianImageFields
from openonda.vpm.runtime import backend


def _cuda_particles_without_cupy(monkeypatch):
    def missing_cuda_mesh():
        raise GaussianMeshUnavailableError("CuPy is not installed")

    monkeypatch.setattr(slip_slab, "_mesh_runtime_is_cuda", lambda: True)
    monkeypatch.setattr(slip_slab, "_check_cuda_mesh_installation", missing_cuda_mesh)
    monkeypatch.setattr(slip_slab, "_check_cpu_mesh_installation", dict)


@pytest.mark.parametrize("setting", ["auto", "cupy_cuda"])
def test_cuda_particle_run_stops_when_the_cuda_slab_images_are_unavailable(monkeypatch, setting):
    _cuda_particles_without_cupy(monkeypatch)
    with pytest.raises(GaussianMeshUnavailableError, match="CuPy is not installed"):
        slip_slab._mesh_execution_backend(setting)


def test_slab_images_run_on_the_cpu_when_requested(monkeypatch):
    _cuda_particles_without_cupy(monkeypatch)
    assert slip_slab._mesh_execution_backend("cpu") == "cpu"


class _ImageFieldsFittingIn(PortableGaussianImageFields):
    """Image fields whose allocation succeeds only on the listed backends."""

    fitting: tuple[str, ...] = ()

    def _construct(self, backend):
        if backend not in self.fitting:
            raise MemoryError(f"{backend} allocation failed")
        return SimpleNamespace(backend=backend, close=lambda: None)


def test_cuda_image_fields_subdivide_into_cuda_blocks_on_gpu_memory_pressure():
    fields_class = type("BlocksFit", (_ImageFieldsFittingIn,), {"fitting": ("cupy_cuda_blocked",)})
    fields = fields_class(execution_backend="cupy_cuda")
    assert fields.backend == "cupy_cuda_blocked"
    assert fields.execution_backend == "cupy_cuda"


def test_cuda_image_fields_stop_when_no_cuda_block_fits_in_gpu_memory():
    fields_class = type("OnlyHostFits", (_ImageFieldsFittingIn,), {"fitting": ("cpu",)})
    with pytest.raises(MemoryError, match="GPU memory"):
        fields_class(execution_backend="cupy_cuda")


@pytest.mark.parametrize("device,precision", [("cpu", "f32"), ("cuda", "f64")])
def test_backend_initialization_stops_when_every_candidate_fails(monkeypatch, device, precision):
    def failing_init(**kwargs):
        raise RuntimeError(f"cannot start {kwargs['arch']}")

    monkeypatch.setattr(backend.ti.lang.impl.get_runtime(), "prog", None)
    monkeypatch.setattr(backend.ti, "init", failing_init)
    monkeypatch.setattr(backend, "_safe_device_memory_for_init", lambda *args: {})
    with pytest.raises(RuntimeError, match=f"compute_device='{device}'"):
        backend.initialize_taichi_backend(device, precision=precision)
