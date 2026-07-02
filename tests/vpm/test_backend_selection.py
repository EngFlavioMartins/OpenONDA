from source.solvers.VPM.config import backend


def _names(chain):
    return [name for _, name in chain]


def test_explicit_vulkan_never_falls_back_to_cuda(monkeypatch):
    monkeypatch.setattr(backend.platform, "system", lambda: "Linux")

    names = _names(backend._build_backend_chain("GPU_VULKAN"))

    assert names[0] == "VULKAN"
    assert "CUDA" not in names
    assert names[-1] == "CPU"


def test_explicit_cuda_never_falls_back_to_vulkan(monkeypatch):
    monkeypatch.setattr(backend.platform, "system", lambda: "Linux")

    names = _names(backend._build_backend_chain("CUDA"))

    assert names[0] == "CUDA"
    assert "VULKAN" not in names
    assert names[-1] == "CPU"
