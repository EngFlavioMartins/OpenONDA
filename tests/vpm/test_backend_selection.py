from source.solvers.VPM.config import backend


def _names(chain):
    return [name for _, name in chain]


def test_explicit_vulkan_never_falls_back_to_cuda(monkeypatch):
    monkeypatch.setattr(backend.platform, "system", lambda: "Linux")

    names = _names(backend._build_backend_chain("VULKAN"))

    assert names == ["VULKAN"]


def test_explicit_cuda_never_falls_back_to_vulkan(monkeypatch):
    monkeypatch.setattr(backend.platform, "system", lambda: "Linux")

    names = _names(backend._build_backend_chain("CUDA"))

    assert names == ["CUDA"]


def test_explicit_metal_never_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(backend.platform, "system", lambda: "Darwin")

    names = _names(backend._build_backend_chain("METAL", precision="f32"))

    assert names == ["METAL"]


def test_macos_f32_prefers_metal(monkeypatch):
    monkeypatch.setattr(backend.platform, "system", lambda: "Darwin")

    names = _names(backend._build_backend_chain("AUTO", precision="f32"))

    assert names[0] == "METAL"
    assert names[-1] == "CPU"


def test_macos_auto_f64_skips_metal(monkeypatch):
    monkeypatch.setattr(backend.platform, "system", lambda: "Darwin")

    names = _names(backend._build_backend_chain("AUTO", precision="f64"))

    assert "METAL" not in names
    assert names and all(name == "CPU" for name in names)


def test_explicit_metal_rejects_f64(monkeypatch):
    monkeypatch.setattr(backend.platform, "system", lambda: "Darwin")

    try:
        backend._build_backend_chain("METAL", precision="f64")
    except ValueError as exc:
        assert "not supported" in str(exc)
    else:
        raise AssertionError("explicit Metal f64 must fail")


def test_explicit_backend_is_not_replaced_by_environment(monkeypatch):
    monkeypatch.setenv("OPENONDA_PROCESSING_UNIT", "CPU")
    monkeypatch.setattr(
        backend.ti.lang.impl,
        "get_runtime",
        lambda: type("Runtime", (), {"prog": None})(),
    )
    requested = []

    def build_chain(preferred, precision):
        requested.append((preferred, precision))
        return [(backend.ti.metal, "METAL")]

    monkeypatch.setattr(backend, "_build_backend_chain", build_chain)
    monkeypatch.setattr(backend.ti, "init", lambda **kwargs: None)
    monkeypatch.setattr(backend, "_probe_taichi_backend", lambda: None)
    monkeypatch.setattr(backend.constants_module, "TAICHI_BACKEND", "UNKNOWN")

    selected = backend.initialize_taichi_backend("METAL", precision="f32")

    assert selected == "METAL"
    assert requested == [("METAL", "f32")]
