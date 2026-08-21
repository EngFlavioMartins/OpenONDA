from source.solvers.VPM.runtime import backend


def _names(chain):
    return [name for _, name in chain]


def test_backend_api_lives_in_runtime_module():
    assert callable(backend.initialize_taichi_backend)
    assert callable(backend.reset_taichi_backend)
    assert callable(backend._build_backend_chain)


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

    assert names == ["METAL"]


def test_macos_rejects_vulkan(monkeypatch):
    monkeypatch.setattr(backend.platform, "system", lambda: "Darwin")

    try:
        backend._build_backend_chain("VULKAN", precision="f32")
    except ValueError as exc:
        assert "unavailable on macOS" in str(exc)
    else:
        raise AssertionError("macOS must not silently replace Vulkan with the CPU")


def test_macos_auto_f64_does_not_fall_back_to_cpu(monkeypatch):
    monkeypatch.setattr(backend.platform, "system", lambda: "Darwin")

    try:
        backend._build_backend_chain("AUTO", precision="f64")
    except ValueError as exc:
        assert "request CPU explicitly" in str(exc)
    else:
        raise AssertionError("AUTO f64 must not silently select the CPU")


def test_explicit_metal_rejects_f64(monkeypatch):
    monkeypatch.setattr(backend.platform, "system", lambda: "Darwin")

    try:
        backend._build_backend_chain("METAL", precision="f64")
    except ValueError as exc:
        assert "not supported" in str(exc)
    else:
        raise AssertionError("explicit Metal f64 must fail")


def test_explicit_backend_is_not_replaced_by_environment(monkeypatch):
    monkeypatch.setenv("OPENONDA_COMPUTE_DEVICE", "CPU")
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
    monkeypatch.setattr(
        backend.ti.lang.impl,
        "current_cfg",
        lambda: type("Config", (), {"arch": backend.ti.metal})(),
    )
    monkeypatch.setattr(backend, "_probe_taichi_backend", lambda: None)
    monkeypatch.setattr(backend.constants_module, "TAICHI_BACKEND", "UNKNOWN")

    selected = backend.initialize_taichi_backend("METAL", precision="f32")

    assert selected == "METAL"
    assert requested == [("METAL", "f32")]
