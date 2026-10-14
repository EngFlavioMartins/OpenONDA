"""Write VTK XML files and ParaView collections of solution frames.

Raw binary arrays avoid base64 encoding overhead. Atomic replacement keeps
readers from opening a partially written file. A ParaView collection
(``.pvd``) lists the frames of one time series with their solution times;
after a restart :func:`rewind_collection` removes the frames beyond the
restored time.
"""

from __future__ import annotations

from collections.abc import Iterable
import math
import os
from pathlib import Path
import tempfile
from typing import Any
from xml.sax.saxutils import quoteattr

from defusedxml import ElementTree

from .file_output import TIME_TOLERANCE, write_text_atomic

__all__ = [
    "DEFAULT_COMPRESSION",
    "configure_writer",
    "read_collection",
    "rewind_collection",
    "write_collection",
    "write_vtk_dataset",
]

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


def read_collection(path: str | Path) -> list[tuple[float, str]]:
    """Read the frames listed in a ParaView collection.

    Parameters
    ----------
    path : str or pathlib.Path
        ``.pvd`` file. A missing file has no frames.

    Returns
    -------
    list of (float, str)
        ``(time, file)`` pairs in file order; ``time`` is in s and ``file`` is
        the frame path relative to the collection.

    Raises
    ------
    ValueError
        If the file is not valid XML or a dataset lacks a finite
        ``timestep`` or a ``file`` attribute.
    """
    source = Path(path)
    if not source.is_file():
        return []
    try:
        datasets = ElementTree.parse(source).getroot().findall(".//DataSet")
        entries = [(float(item.attrib["timestep"]), item.attrib["file"]) for item in datasets]
    except (ElementTree.ParseError, KeyError, ValueError) as error:
        raise ValueError(f"Invalid ParaView collection {source}") from error
    if not all(math.isfinite(time) for time, _ in entries):
        raise ValueError(f"ParaView collection {source} has a non-finite time")
    return entries


def write_collection(path: str | Path, entries: Iterable[tuple[float, str | Path]]) -> Path:
    """Atomically write a ParaView collection of solution frames.

    Frames are sorted by time and file name. When a file appears more than
    once, its last entry replaces the earlier ones, so a frame written again
    after a restart is listed once. Times are written as ``repr(float(time))``,
    the shortest representation that reads back as the same float.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination ``.pvd`` file. Missing parent directories are created.
    entries : iterable of (float, str or pathlib.Path)
        ``(time, file)`` pairs; ``time`` in s, ``file`` relative to the
        collection.

    Returns
    -------
    pathlib.Path
        The destination path.

    Raises
    ------
    ValueError
        If a time is not finite or a file path is empty or absolute.
    """
    frames: dict[str, float] = {}
    for time, file in entries:
        name = Path(file).as_posix()
        if not str(file) or Path(file).is_absolute():
            raise ValueError(f"Collection frame path {file!r} must be relative")
        time = float(time)
        if not math.isfinite(time):
            raise ValueError(f"Collection frame {name} has a non-finite time")
        frames.pop(name, None)
        frames[name] = time
    ordered = sorted((time, name) for name, time in frames.items())
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
        *(
            f"    <DataSet timestep={quoteattr(repr(time))} file={quoteattr(name)}/>"
            for time, name in ordered
        ),
        "  </Collection>",
        "</VTKFile>",
    ]
    return write_text_atomic(path, "\n".join(lines) + "\n")


def _frame_path(collection: Path, name: str) -> Path:
    """Resolve a frame path, refusing paths outside the collection directory."""
    relative = Path(name)
    if not name or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe frame path {name!r} in {collection}")
    return collection.parent / relative


def rewind_collection(path: str | Path, time: float) -> None:
    """Keep the collection frames up to and including ``time``.

    The frame files beyond ``time`` are deleted, including the pieces of a
    parallel ``.pvtu`` frame. The collection is rewritten only when a frame
    is removed or listed twice.

    Parameters
    ----------
    path : str or pathlib.Path
        ``.pvd`` file. A missing file is left as is.
    time : float
        Restored solution time in s.

    Raises
    ------
    ValueError
        If the collection is invalid or lists a frame outside its directory.
    """
    source = Path(path)
    entries = read_collection(source)
    kept = [(t, name) for t, name in entries if t <= time + TIME_TOLERANCE]
    removed = [name for t, name in entries if t > time + TIME_TOLERANCE]
    kept_names = {name for _, name in kept}
    for name in removed:
        if name in kept_names:
            continue
        frame = _frame_path(source, name)
        if frame.suffix == ".pvtu" and frame.is_file():
            for piece in ElementTree.parse(frame).getroot().findall(".//Piece"):
                _frame_path(frame, piece.attrib.get("Source", "")).unlink(missing_ok=True)
        frame.unlink(missing_ok=True)
    if removed or len(kept_names) != len(kept):
        write_collection(source, kept)
