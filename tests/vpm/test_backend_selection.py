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


def test_macos_f32_prefers_metal(monkeypatch):
    monkeypatch.setattr(backend.platform, "system", lambda: "Darwin")

    names = _names(backend._build_backend_chain("GPU", precision="f32"))

    assert names[0] == "METAL"
    assert names[-1] == "CPU"


def test_macos_f64_skips_metal(monkeypatch, capsys):
    """f64 on macOS must never select Metal: Metal has no fp64 and f64
    kernels abort the process (uncatchably) at SPIRV codegen time."""
    monkeypatch.setattr(backend.platform, "system", lambda: "Darwin")

    names = _names(backend._build_backend_chain("GPU", precision="f64"))

    assert "METAL" not in names
    assert names and all(name == "CPU" for name in names)
    assert "not supported by the Metal backend" in capsys.readouterr().err
