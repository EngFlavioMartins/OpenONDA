"""
VLM geometry subpackage: Aircraft, Wing, and WingSegment primitives.
"""

from .aircraft import Aircraft, Wing, WingSegment
from .openvsp_io import load_degengeom_csv

__all__ = [
    "Aircraft",
    "Wing",
    "WingSegment",
    "load_degengeom_csv",
]
