"""Study candidate API; implementations live together in :mod:`assets.core`."""
from .core import advance, integrate

advance_one_step = advance
__all__ = ["advance", "advance_one_step", "integrate"]
