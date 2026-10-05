"""Actual rounding/exception restoration through the production opaque bridge.

Both shared libraries must be built explicitly in a temporary directory and
supplied by environment variables. This test never compiles or installs code.
The qualification C probe uses public FE constants internally and returns
only portable test labels; no register bits or fenv_t layout are assumed.
"""

from contextlib import contextmanager
import ctypes
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from source.solvers.vpm.numerics import ieee
from source.solvers.vpm.physics.induction.gaussian_tail._interval import _platform


@pytest.fixture
def compiled_bridge(monkeypatch):
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
        pytest.skip("precompiled production _fenv bridge required")


class ModeProbe:
    def __init__(self, library):
        self.library = ctypes.PyDLL(str(library))
        self.library.openonda_modes_enter.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
        self.library.openonda_modes_enter.restype = ctypes.c_int
        self.library.openonda_modes_observe.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.library.openonda_modes_observe.restype = ctypes.c_int
        self.library.openonda_modes_raise_underflow.argtypes = []
        self.library.openonda_modes_raise_underflow.restype = ctypes.c_int
        self.library.openonda_modes_restore.argtypes = [ctypes.c_void_p]
        self.library.openonda_modes_restore.restype = ctypes.c_int

    def observe(self):
        mode, flags = ctypes.c_int(), ctypes.c_int()
        assert self.library.openonda_modes_observe(ctypes.byref(mode), ctypes.byref(flags)) == 0
        return mode.value, flags.value

    def raise_underflow(self):
        assert self.library.openonda_modes_raise_underflow() == 0

    @contextmanager
    def altered(self, mode):
        handle = ctypes.c_void_p()
        status = self.library.openonda_modes_enter(mode, ctypes.byref(handle))
        if status == 2:
            pytest.skip("requested rounding mode unavailable on this platform")
        assert status == 0 and handle.value
        try:
            yield
        finally:
            assert self.library.openonda_modes_restore(handle) == 0


@pytest.fixture
def mode_probe(compiled_bridge):
    del compiled_bridge  # Load bridge before introducing an altered test mode.
    path = os.environ.get("OPENONDA_TEST_FENV_PROBE")
    if not path:
        pytest.skip("separately built qualification mode probe required")
    return ModeProbe(Path(path).resolve(strict=True))


@pytest.mark.parametrize("mode", [0, 1, 2, 3])
@pytest.mark.parametrize("exit_kind", ["normal", "body_error", "nested_error"])
def test_restores_actual_rounding_and_preexisting_exception_flags(mode_probe, mode, exit_kind):
    with mode_probe.altered(mode):
        before = mode_probe.observe()
        assert before[0] == mode
        assert before[1] & 3 == 3  # Portable INVALID and DIVBYZERO test labels.

        def work():
            with ieee.ieee_arithmetic():
                assert mode_probe.observe() == (0, 0)
                _platform()
                mode_probe.raise_underflow()
                outer = mode_probe.observe()
                assert outer[0] == 0 and outer[1] & 24 == 24
                if exit_kind == "nested_error":
                    with pytest.raises(ValueError, match="inner"), ieee.ieee_arithmetic():
                        assert mode_probe.observe() == (0, 0)
                        mode_probe.raise_underflow()
                        raise ValueError("inner")
                    assert mode_probe.observe() == outer
                if exit_kind == "body_error":
                    raise ValueError("body")

        if exit_kind == "body_error":
            with pytest.raises(ValueError, match="body"):
                work()
        else:
            work()
        assert mode_probe.observe() == before


@pytest.mark.parametrize("mode", [1, 2, 3])
def test_production_validation_rejects_each_actual_nonnearest_mode(mode_probe, mode):
    with mode_probe.altered(mode):
        with pytest.raises(RuntimeError, match="round-to-nearest"):
            _platform()
        # The validation probe does not silently change controls to succeed.
        assert mode_probe.observe()[0] == mode
        with ieee.ieee_arithmetic():
            _platform()
        assert mode_probe.observe()[0] == mode


@pytest.mark.parametrize("mode", [0, 1, 2, 3])
def test_rounding_inspection_preserves_modes_and_flags(mode_probe, compiled_bridge, mode):
    with mode_probe.altered(mode):
        before = mode_probe.observe()
        assert compiled_bridge.round_to_nearest() is (mode == 0)
        assert mode_probe.observe() == before
        if mode == 0:
            assert ieee.require_round_to_nearest() is None
        else:
            with pytest.raises(RuntimeError, match="requires round-to-nearest"):
                ieee.require_round_to_nearest()
        assert mode_probe.observe() == before
        with ieee.ieee_arithmetic():
            inner = mode_probe.observe()
            ieee.require_round_to_nearest()
            assert mode_probe.observe() == inner
        assert mode_probe.observe() == before


def test_production_tail_after_taichi_restores_actual_host_environment(mode_probe):
    import taichi as ti

    from source.solvers.vpm.physics.induction.gaussian_tail import (
        prepare_tail_source,
        query_tail_bound,
        validate_source_values,
    )

    # This qualification-only outer capture preserves pytest's initial state;
    # Taichi initialization is deliberately OUTSIDE the production arithmetic
    # scope, whose conditions accepts synchronous host mathematics only.
    with mode_probe.altered(0):
        ti.init(arch=ti.cpu, offline_cache=False, cpu_max_num_threads=1)
        try:
            position = np.array([[0.125, -0.25, 0.125]], dtype=np.float32)
            strength = np.array([[0.01, -0.02, 0.03]], dtype=np.float32)
            radius = np.array([0.04], dtype=np.float32)
            try:
                _platform()
            except RuntimeError:
                needs_scope = True
            else:
                needs_scope = False
            before = mode_probe.observe()
            ieee.require_round_to_nearest()
            assert mode_probe.observe() == before
            with ieee.ieee_arithmetic():
                _platform()
                snapshot = prepare_tail_source(position, strength, radius, z_min=-0.5, z_max=0.5)
                validate_source_values(snapshot, position, strength, radius, z_min=-0.5, z_max=0.5)
                result = query_tail_bound(snapshot, [-1.0, -1.0, -0.5], [1.0, 1.0, 0.5], shells=128)
            assert mode_probe.observe() == before
            assert 0.0 < result.velocity_upper < 1.0e-4
            assert 0.0 < result.gradient_upper < 1.0e-4
            if needs_scope:
                with pytest.raises(RuntimeError):
                    _platform()
            else:
                _platform()
        finally:
            ti.reset()
