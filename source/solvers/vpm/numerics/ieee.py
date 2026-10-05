"""Restoring, thread-local host arithmetic scopes for numerical error_bounds.

Taichi may enable host flush-to-zero even with CUDA selected. Interval
arithmetic must not quietly waive its gradual-underflow requirement. This
module uses an optional, installation-time compiled standard-C bridge. There
is no runtime compiler, hard-coded control register, platform struct layout,
or global mode change. The caller must still check its arithmetic assumptions.

Only synchronous host arithmetic belongs inside this scope: no Taichi/GPU
calls, callbacks, async yields or task switches. Normal solver arithmetic
resumes in the exact saved environment, also after an exception.
"""

from contextlib import contextmanager
from importlib import import_module


class IEEEEnvironmentUnavailableError(RuntimeError):
    """The optional compiled arithmetic guard was not installed."""


def _bridge():
    try:
        return import_module("source.solvers.vpm.numerics._fenv")
    except ImportError as error:
        raise IEEEEnvironmentUnavailableError(
            "The optional precompiled floating-point guard is unavailable; "
            "install a build with the _fenv extension before enabling validated "
            "Gaussian tail arithmetic. No floating-point assumptions were relaxed."
        ) from error


def require_round_to_nearest():
    """Validate the current host rounding mode without changing its environment.

    This checks only the standard-C rounding direction. FTZ/DAZ, exception
    flags and all other caller controls remain untouched; gradual underflow
    is neither required nor promised here. Error bound arithmetic still
    requires its separate restoring ``ieee_arithmetic`` scope and checks.
    """
    bridge = _bridge()
    try:
        inspect = bridge.round_to_nearest
    except AttributeError as error:
        raise IEEEEnvironmentUnavailableError(
            "The precompiled floating-point bridge lacks rounding inspection; "
            "rebuild the _fenv extension. No host controls were changed."
        ) from error
    if not inspect():
        raise RuntimeError("Gaussian field host arithmetic requires round-to-nearest")


@contextmanager
def ieee_arithmetic():
    """Enter standard host arithmetic and restore the prior thread environment.

    Nested scopes are supported. Native code owns the active same-thread
    stack and performs mode/stack transitions atomically with respect to
    Python exceptions. Capture changes no mode, and entry occurs only after
    establishing finally. Restoration failure prevents successful publication
    of the caller's tail-bound calculation.
    """
    bridge = _bridge()
    token = bridge.capture()
    try:
        bridge.enter_default(token)
        yield
    finally:
        bridge.restore(token)
