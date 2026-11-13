"""Numba integration that warns when the on-disk compilation cache is unavailable."""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any, TypeVar

from numba import njit as _numba_njit

_Function = TypeVar("_Function", bound=Callable[..., Any])
_LOGGER = logging.getLogger("openonda")
_uncached_warning_logged = False


def _compile_uncached(error: RuntimeError) -> None:
    """Re-raise ``error`` unless it is Numba's missing-cache error; warn once."""
    global _uncached_warning_logged
    if "no locator available" not in str(error):
        raise error
    if not _uncached_warning_logged:
        _LOGGER.warning(
            "The Numba cache directory is not writable, so every start recompiles the "
            "Numba kernels; set NUMBA_CACHE_DIR to a writable directory"
        )
        _uncached_warning_logged = True


def cacheable_njit(*args: Any, **kwargs: Any) -> Any:
    """Return a Numba decorator that compiles without the disk cache when it is unavailable.

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
    retried with caching disabled, and the first retry logs a warning on the
    ``openonda`` logger; compilation and typing errors still propagate
    unchanged.
    """
    if not kwargs.get("cache"):
        return _numba_njit(*args, **kwargs)

    uncached = dict(kwargs)
    uncached["cache"] = False

    if args and callable(args[0]):
        try:
            return _numba_njit(*args, **kwargs)
        except RuntimeError as error:
            _compile_uncached(error)
            return _numba_njit(*args, **uncached)

    configured = _numba_njit(*args, **kwargs)

    def decorate(function: _Function) -> Any:
        try:
            return configured(function)
        except RuntimeError as error:
            _compile_uncached(error)
            return _numba_njit(*args, **uncached)(function)

    return decorate
