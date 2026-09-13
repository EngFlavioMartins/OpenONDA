"""
VLM geometry subpackage: Aircraft, Wing, and WingSegment primitives.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from .aircraft import Aircraft, Wing, WingSegment
from .openvsp_io import (
    OpenVSPImportConfig,
    export_openvsp_degengeom,
    load_degengeom_csv,
    load_openvsp_surface,
)

__all__ = [
    "Aircraft",
    "Wing",
    "WingSegment",
    "OpenVSPImportConfig",
    "export_openvsp_degengeom",
    "load_degengeom_csv",
    "load_openvsp_surface",
]
