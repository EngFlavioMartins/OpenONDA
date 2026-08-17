"""Sandbox-only import shim: fake `taichi` and `numba` so the OpenONDA
experiment scripts can import `source.solvers.VPM...` for pure-numpy helpers
(e.g. `_m4_prime_1d`) without a GPU/JIT toolchain installed.

Nothing in the VPM_LES experiment scripts calls a taichi kernel or an njit
function; they only need the module import to succeed. Any attempt to actually
*use* a shimmed object raises, so a silent wrong-answer path is impossible.
"""

from __future__ import annotations

import sys
import types


class _Unusable:
    """Placeholder that is fine to reference but raises if used."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __call__(self, *a, **k):
        raise RuntimeError(
            f"shimmed taichi/numba object {self._name!r} was actually invoked; "
            "this sandbox has no taichi/numba. Run on the real environment."
        )

    def __getattr__(self, item):
        return _Unusable(f"{self._name}.{item}")

    def __getitem__(self, item):
        return _Unusable(f"{self._name}[{item!r}]")

    def __repr__(self) -> str:
        return f"<shim {self._name}>"


class _Dtype(_Unusable):
    """Usable as a type annotation; must not raise on reference."""

    def __call__(self, *a, **k):
        raise RuntimeError(f"shimmed dtype {self._name!r} invoked")


class _TypeFactory:
    """Callable that yields an annotation placeholder.

    Taichi type constructors appear inside function *annotations*, which Python
    evaluates at import time, so these must not raise. They carry no compute
    semantics, so allowing them cannot produce a wrong numerical answer.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    def __call__(self, *a, **k):
        return _Dtype(f"{self._name}(...)")

    def __getattr__(self, item):
        return _TypeFactory(f"{self._name}.{item}")

    def __repr__(self) -> str:
        return f"<shim type {self._name}>"


def _passthrough_decorator(fn=None, **_kwargs):
    """Behaves as both @dec and @dec(...)."""
    if fn is None:
        return lambda f: f
    return fn


def _install_taichi() -> None:
    ti = types.ModuleType("taichi")

    for name in ("f16", "f32", "f64", "i8", "i16", "i32", "i64", "u8", "u32", "u64"):
        setattr(ti, name, _Dtype(f"ti.{name}"))

    # Decorators used at import time must return the object unchanged.
    ti.func = _passthrough_decorator
    ti.kernel = _passthrough_decorator
    ti.real_func = _passthrough_decorator
    ti.data_oriented = _passthrough_decorator
    ti.pyfunc = _passthrough_decorator

    # Type constructors appear in annotations -> evaluated at import time.
    ti.template = _TypeFactory("ti.template")
    ti.ndarray = _TypeFactory("ti.ndarray")
    ti.any_arr = _TypeFactory("ti.any_arr")

    types_mod = types.ModuleType("taichi.types")
    types_mod.__getattr__ = lambda item: _TypeFactory(f"ti.types.{item}")
    ti.types = types_mod

    def __getattr__(item):  # noqa: N807 - module-level hook
        return _Unusable(f"ti.{item}")

    ti.__getattr__ = __getattr__

    algorithms = types.ModuleType("taichi.algorithms")
    algorithms.__getattr__ = lambda item: _Unusable(f"ti.algorithms.{item}")
    ti.algorithms = algorithms

    math_mod = types.ModuleType("taichi.math")
    math_mod.__getattr__ = lambda item: _Unusable(f"ti.math.{item}")
    ti.math = math_mod

    ti.__SHIMMED__ = True
    sys.modules.setdefault("taichi", ti)
    sys.modules.setdefault("taichi.algorithms", algorithms)
    sys.modules.setdefault("taichi.math", math_mod)
    sys.modules.setdefault("taichi.types", types_mod)


def _install_numba() -> None:
    nb = types.ModuleType("numba")
    nb.njit = _passthrough_decorator
    nb.jit = _passthrough_decorator
    nb.vectorize = _passthrough_decorator
    nb.stencil = _passthrough_decorator
    nb.prange = range
    for name in ("float32", "float64", "int32", "int64", "boolean"):
        setattr(nb, name, _Dtype(f"nb.{name}"))
    nb.__getattr__ = lambda item: _Unusable(f"nb.{item}")
    nb.__SHIMMED__ = True

    types_mod = types.ModuleType("numba.types")
    types_mod.__getattr__ = lambda item: _Dtype(f"nb.types.{item}")
    nb.types = types_mod

    sys.modules.setdefault("numba", nb)
    sys.modules.setdefault("numba.types", types_mod)


try:
    import taichi  # noqa: F401
except ModuleNotFoundError:
    _install_taichi()

try:
    import numba  # noqa: F401
except ModuleNotFoundError:
    _install_numba()
