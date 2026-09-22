"""Cross-platform VPM compute-device selection contracts."""

from types import SimpleNamespace

import pytest

from source.solvers.vpm.runtime import backend


def test_vulkan_memory_probe_does_not_require_a_display(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":unavailable")
    observed = {}

    def run(_command, **kwargs):
        observed.update(kwargs)
        return SimpleNamespace(
            returncode=0,
            stdout="memoryHeaps[0]:\n    size = 16000\n    budget = 5000\n",
        )

    monkeypatch.setattr(backend.subprocess, "run", run)
    assert backend._query_vulkan_budget() == (16000, 5000)
    assert "DISPLAY" not in observed["env"]
    assert backend.os.environ["DISPLAY"] == ":unavailable"


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


def test_integrated_gpu_pool_covers_the_fixed_workspace(monkeypatch):
    gib = 1 << 30
    monkeypatch.setattr(backend.platform, "system", lambda: "Linux")
    monkeypatch.setattr(backend, "_query_vulkan_budget", lambda: (16 * gib, 5 * gib))
    monkeypatch.setattr(backend, "_is_likely_integrated_gpu", lambda: True)

    default = backend._safe_device_memory_for_init(0.5)
    large = backend._safe_device_memory_for_init(0.5, minimum_pool_bytes=3 * gib)

    assert default["device_memory_GB"] == pytest.approx(1.5)
    assert large["device_memory_GB"] == pytest.approx(3.0)


def test_integrated_gpu_pool_leaves_transfer_headroom(monkeypatch):
    gib = 1 << 30
    monkeypatch.setattr(backend.platform, "system", lambda: "Linux")
    monkeypatch.setattr(backend, "_query_vulkan_budget", lambda: (16 * gib, 4 * gib))
    monkeypatch.setattr(backend, "_is_likely_integrated_gpu", lambda: True)

    pool = backend._safe_device_memory_for_init(0.5, minimum_pool_bytes=3 * gib)

    assert pool["device_memory_GB"] == pytest.approx(2.8)
