"""Shared fixtures for the coupler test suite."""

from __future__ import annotations

import logging

import pytest


@pytest.fixture(autouse=True)
def _reset_coupler_logger():
    """Detach the 'coupler' logger's handlers after every test.

    ``FVMVPMCoupler._configure_logging`` is a no-op when the logger already
    has handlers (so an embedding application keeps control).  Within one
    pytest process that guard makes tests order-dependent: the first test
    that constructs a coupler pins ``solution/coupler.log`` to ITS case
    directory and every later coupler gets no log file.  Closing the
    handlers between tests restores per-test isolation.
    """
    yield
    logger = logging.getLogger("coupler")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
