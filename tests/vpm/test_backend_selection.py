"""Cross-platform VPM compute-device selection contracts."""

import pytest

from source.solvers.vpm.runtime import backend


@pytest.mark.parametrize(
    ("system", "has_nvidia", "expected"),
    (
        ("Darwin", False, ["METAL"]),
        ("Linux", False, ["VULKAN", "CUDA"]),
        ("Linux", True, ["CUDA", "VULKAN"]),
        ("Windows", False, ["VULKAN", "CUDA"]),
    ),
)
def test_auto_prioritizes_a_native_gpu_backend(monkeypatch, system, has_nvidia, expected):
    monkeypatch.setattr(backend.platform, "system", lambda: system)
    monkeypatch.setattr(backend, "_has_nvidia_gpu", lambda: has_nvidia)

    assert [name for _, name in backend._build_backend_chain("AUTO")] == expected


def test_auto_skips_gpu_backends_not_qualified_by_the_numerical_method(monkeypatch):
    monkeypatch.setattr(backend.platform, "system", lambda: "Linux")
    monkeypatch.setattr(backend, "_has_nvidia_gpu", lambda: True)
    qualified = frozenset({"AUTO", "CPU", "VULKAN", "METAL"})

    chain = backend._build_backend_chain("AUTO", supported_devices=qualified)

    assert [name for _, name in chain] == ["VULKAN"]
