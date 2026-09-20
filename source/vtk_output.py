"""Write VTK XML files with appended binary payloads.

Raw payloads avoid base64 encoding overhead. Atomic replacement keeps readers
from opening a partially written file.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any

try:
    import vtk as _vtk_module
except ImportError:  # Optional visualization dependency.
    _vtk_module = None

vtk: Any = _vtk_module

_COMPRESSOR = {
    "lz4": "SetCompressorTypeToLZ4",
    "lzma": "SetCompressorTypeToLZMA",
    "none": "SetCompressorTypeToNone",
    "zlib": "SetCompressorTypeToZLib",
}

DEFAULT_COMPRESSION = "zlib"


def configure_writer(writer, compression: str = DEFAULT_COMPRESSION) -> None:
    """Put one VTK XML writer into appended-raw mode with the given compressor."""
    if compression not in _COMPRESSOR:
        raise ValueError(f"compression must be one of {', '.join(sorted(_COMPRESSOR))}")
    writer.SetDataModeToAppended()
    writer.EncodeAppendedDataOff()
    getattr(writer, _COMPRESSOR[compression])()


def write_vtk_dataset(
    dataset,
    filename: str | Path,
    *,
    compression: str = DEFAULT_COMPRESSION,
) -> Path:
    """Atomically write any VTK dataset to the XML format matching its type."""
    if vtk is None:
        raise ImportError("VTK output requires the optional visualization dependencies")

    target = Path(filename)
    target.parent.mkdir(parents=True, exist_ok=True)

    writer = vtk.vtkXMLDataSetWriter()
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.stem}.",
        suffix=f".tmp{target.suffix}",
        dir=target.parent,
    )
    os.close(descriptor)
    try:
        writer.SetFileName(temporary)
        writer.SetInputData(dataset)
        configure_writer(writer, compression)
        if writer.Write() != 1:
            raise OSError(f"VTK failed to write {target}")
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
    return target
