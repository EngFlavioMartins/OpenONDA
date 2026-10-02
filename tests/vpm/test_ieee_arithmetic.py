"""Optional precompiled host environment: no physical solver configuration."""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from source.solvers.vpm.numerics import ieee


@pytest.fixture
def native_bridge(monkeypatch):
    # A build under a temporary directory can be qualified without installing
    # into the user's environment or making an in-place binary in the repo.
    path = os.environ.get("OPENONDA_TEST_FENV_EXTENSION")
    if path:
        name = "source.solvers.vpm.numerics._fenv"
        spec = importlib.util.spec_from_file_location(name, Path(path).resolve(strict=True))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setitem(sys.modules, name, module)
    try:
        return ieee._bridge()
    except ieee.IEEEEnvironmentUnavailableError:
        pytest.skip("optional compiled _fenv extension not available")


def _probe():
    smallest_normal = np.array([2.0**-1022], dtype=np.float64)
    return (smallest_normal * 0.5)[0].hex()


def test_missing_extension_fails_without_changing_environment(monkeypatch):
    before = _probe()

    def unavailable(_name):
        raise ImportError("injected")

    monkeypatch.setattr(ieee, "import_module", unavailable)
    with pytest.raises(ieee.IEEEEnvironmentUnavailableError), ieee.ieee_arithmetic():
        pytest.fail("unavailable arithmetic guard cannot yield")
    assert _probe() == before


def test_missing_rounding_inspection_fails_without_changing_environment(monkeypatch):
    before = _probe()
    monkeypatch.setattr(ieee, "_bridge", lambda: SimpleNamespace())
    with pytest.raises(ieee.IEEEEnvironmentUnavailableError, match="lacks rounding inspection"):
        ieee.require_round_to_nearest()
    assert _probe() == before


def test_nested_exception_restores_and_tokens_are_single_use(native_bridge):
    before = _probe()
    with ieee.ieee_arithmetic():
        assert _probe() == "0x0.8000000000000p-1022"
        with pytest.raises(ValueError, match="injected"), ieee.ieee_arithmetic():
            assert _probe() == "0x0.8000000000000p-1022"
            raise ValueError("injected")
        assert _probe() == "0x0.8000000000000p-1022"
    assert _probe() == before
    token = native_bridge.capture()
    native_bridge.enter_default(token)
    native_bridge.restore(token)
    native_bridge.restore(token)  # Idempotent finalization, but entry is single use.
    with pytest.raises(RuntimeError, match="single use"):
        native_bridge.enter_default(token)


def test_native_restore_refuses_wrong_thread(native_bridge):
    before = _probe()
    token = native_bridge.capture()
    native_bridge.enter_default(token)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(native_bridge.restore, token)
            with pytest.raises(RuntimeError, match="another thread"):
                future.result()
    finally:
        native_bridge.restore(token)
    assert _probe() == before


@pytest.mark.parametrize("boundary", ["capture", "enter_default", "restore"])
def test_interrupt_after_native_return_preserves_outer_scope(native_bridge, monkeypatch, boundary):
    class InterruptedBridge:
        def __getattr__(self, name):
            def call(*args):
                result = getattr(native_bridge, name)(*args)
                if name == boundary:
                    raise KeyboardInterrupt("injected after native boundary")
                return result
            return call

    before = _probe()
    with ieee.ieee_arithmetic():
        monkeypatch.setattr(ieee, "_bridge", lambda: InterruptedBridge())
        with pytest.raises(KeyboardInterrupt), ieee.ieee_arithmetic():
            assert _probe() == "0x0.8000000000000p-1022"
        assert _probe() == "0x0.8000000000000p-1022"
        monkeypatch.undo()
    assert _probe() == before


def test_native_out_of_order_restoration_is_retryable(native_bridge):
    outer, inner = native_bridge.capture(), native_bridge.capture()
    native_bridge.enter_default(outer)
    native_bridge.enter_default(inner)
    with pytest.raises(RuntimeError, match="reverse entry order"):
        native_bridge.restore(outer)
    native_bridge.restore(inner)
    native_bridge.restore(outer)


def test_scope_restores_actual_taichi_host_mode(native_bridge):
    import taichi as ti

    ti.init(arch=ti.cpu, offline_cache=False, cpu_max_num_threads=1)
    try:
        before = _probe()
        with ieee.ieee_arithmetic():
            assert _probe() == "0x0.8000000000000p-1022"
        assert _probe() == before
        with pytest.raises(ValueError), ieee.ieee_arithmetic():
            raise ValueError("injected")
        assert _probe() == before
    finally:
        ti.reset()
