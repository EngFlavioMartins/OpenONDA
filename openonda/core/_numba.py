"""Numba integration that treats the on-disk compilation cache as optional."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from numba import njit as _numba_njit

_Function = TypeVar("_Function", bound=Callable[..., Any])


def cacheable_njit(*args: Any, **kwargs: Any) -> Any:
    """Return a Numba decorator with a safe uncached fallback.

    Parameters
    ----------
    *args, **kwargs : object
        Arguments accepted by :func:`numba.njit`. ``cache=True`` remains the
        normal path.

    Returns
    -------
    callable
        Numba dispatcher or decorator, matching :func:`numba.njit`.

    Notes
    -----
    Some installed or sandboxed environments make both the package directory
    and the user cache read-only. Numba otherwise raises while importing a
    module containing ``cache=True``. Only that missing-cache-locator error is
    retried with caching disabled; compilation and typing errors still
    propagate unchanged.
    """
    if not kwargs.get("cache"):
        return _numba_njit(*args, **kwargs)

    uncached = dict(kwargs)
    uncached["cache"] = False

    if args and callable(args[0]):
        try:
            return _numba_njit(*args, **kwargs)
        except RuntimeError as error:
            if "no locator available" not in str(error):
                raise
            return _numba_njit(*args, **uncached)

    configured = _numba_njit(*args, **kwargs)

    def decorate(function: _Function) -> Any:
        try:
            return configured(function)
        except RuntimeError as error:
            if "no locator available" not in str(error):
                raise
            return _numba_njit(*args, **uncached)(function)

    return decorate
