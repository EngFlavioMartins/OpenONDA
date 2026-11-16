"""Write VTK XML files and ParaView collections of solution frames.

Raw binary arrays avoid base64 encoding overhead. Atomic replacement keeps
readers from opening a partially written file. A ParaView collection
(``.pvd``) lists the frames of one time series with their solution times;
after a restart :func:`rewind_collection` removes the frames beyond the
restored time.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math
from pathlib import Path
import re
from xml.sax.saxutils import quoteattr

from defusedxml import ElementTree
import numpy as np
import vtk

from .clock import time_tolerance
from .file_output import atomic_path, write_text_atomic

__all__ = [
    "DEFAULT_COMPRESSION",
    "add_frame",
    "configure_writer",
    "read_collection",
    "rewind_collection",
    "write_collection",
    "write_surface_frame",
    "write_vtk_dataset",
]

_COMPRESSOR = {
    "lz4": "SetCompressorTypeToLZ4",
    "lzma": "SetCompressorTypeToLZMA",
    "none": "SetCompressorTypeToNone",
    "zlib": "SetCompressorTypeToZLib",
}

DEFAULT_COMPRESSION = "zlib"


def configure_writer(writer: vtk.vtkXMLWriter, compression: str = DEFAULT_COMPRESSION) -> None:
    """Put one VTK XML writer into appended-raw mode with the given compressor."""
    if compression not in _COMPRESSOR:
        raise ValueError(f"compression must be one of {', '.join(sorted(_COMPRESSOR))}")
    writer.SetDataModeToAppended()
    writer.EncodeAppendedDataOff()
    getattr(writer, _COMPRESSOR[compression])()


def write_vtk_dataset(
    dataset: vtk.vtkDataSet,
    filename: str | Path,
    *,
    compression: str = DEFAULT_COMPRESSION,
    uint64_header: bool = False,
) -> Path:
    """Atomically write any VTK dataset to the XML format matching its type.

    ``uint64_header`` selects 64-bit block headers, which arrays above 4 GiB need.
    """
    target = Path(filename)
    with atomic_path(target) as temporary:
        writer = vtk.vtkXMLDataSetWriter()
        writer.SetFileName(str(temporary))
        writer.SetInputData(dataset)
        configure_writer(writer, compression)
        if uint64_header:
            writer.SetHeaderTypeToUInt64()
        if writer.Write() != 1:
            raise OSError(f"VTK failed to write {target}")
    return target


def write_surface_frame(
    path: str | Path,
    positions: Sequence[np.ndarray],
    grid_shape: tuple[int, int],
    point_data: Mapping[str, np.ndarray],
    field_data: Mapping[str, np.ndarray],
    *,
    compression: str = DEFAULT_COMPRESSION,
) -> Path:
    """Atomically write the samples on a plane as a VTK structured grid (``.vts``).

    Parameters
    ----------
    path : str or pathlib.Path
        Destination file.
    positions : sequence of numpy.ndarray
        The x, y and z coordinates of the sample points in m, each with
        ``n_1 * n_2`` values in the order of
        :func:`openonda.core.sampling.surface_points`, the second in-plane
        axis varying fastest.
    grid_shape : tuple of int
        ``(n_1, n_2)`` of :func:`openonda.core.sampling.surface_points`.
    point_data : mapping of str to numpy.ndarray
        Point arrays with one row per sample point in the same order, shape
        ``(n_1 * n_2,)`` or ``(n_1 * n_2, k)``, already in the dtype to store.
    field_data : mapping of str to numpy.ndarray
        Field arrays stored as given, for example the solution time.
    compression : str, default="zlib"
        VTK compressor, see :func:`configure_writer`.

    Returns
    -------
    pathlib.Path
        The destination path.
    """
    import pyvista as pv

    n_first, n_second = grid_shape
    grid = pv.StructuredGrid(
        *(np.asarray(values).reshape((n_first, n_second, 1)) for values in positions)
    )
    for name, values in point_data.items():
        # Sample points are in C order; VTK structured points are in Fortran order.
        grid_values = np.asarray(values).reshape((n_first, n_second, -1)).transpose(1, 0, 2)
        grid.point_data[name] = grid_values.reshape((n_first * n_second, *np.shape(values)[1:]))
    for name, values in field_data.items():
        grid.field_data[name] = values
    return write_vtk_dataset(grid, path, compression=compression)


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


_COLLECTION_HEAD = (
    '<?xml version="1.0"?>\n'
    '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
    "  <Collection>\n"
)
_COLLECTION_TAIL = "  </Collection>\n</VTKFile>\n"


def _collection_line(time: float, name: str) -> str:
    return f"    <DataSet timestep={quoteattr(repr(time))} file={quoteattr(name)}/>\n"


def _ordered_frames(entries: Iterable[tuple[float, str | Path]]) -> list[tuple[float, str]]:
    """Return the ``(time, file)`` pairs of a collection in file order.

    Frames are sorted by time and file name; a file listed more than once
    keeps only its last entry.
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
    return sorted((time, name) for name, time in frames.items())


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
    ordered = _ordered_frames(entries)
    text = _COLLECTION_HEAD + "".join(_collection_line(*frame) for frame in ordered)
    return write_text_atomic(path, text + _COLLECTION_TAIL)


def add_frame(collection: str | Path, time: float, frame: str | Path) -> Path:
    """List one frame in a ParaView collection, replacing an earlier entry for the file.

    The file is the same as :func:`write_collection` would write for the
    listed frames plus this one, and it is replaced atomically. When the
    collection is in the form :func:`write_collection` writes and the frame
    sorts after the last one and is not yet listed, the new line is inserted
    before the closing tags without parsing the frames, so the cost of a frame
    hardly grows with the length of the series. Every other frame reads and
    rewrites the collection whole.

    Parameters
    ----------
    collection : str or pathlib.Path
        ``.pvd`` file. A missing file starts a new collection.
    time : float
        Solution time of the frame in s.
    frame : str or pathlib.Path
        Frame file, relative to the collection directory or an absolute path
        below it.

    Returns
    -------
    pathlib.Path
        The collection path.

    Raises
    ------
    ValueError
        If the collection is invalid, ``time`` is not finite, or an absolute
        ``frame`` lies outside the collection directory.
    """
    source = Path(collection)
    relative = Path(frame)
    if relative.is_absolute():
        relative = relative.resolve().relative_to(source.parent.resolve())
    text = _appended_collection(source, time, relative.as_posix())
    if text is None:
        return write_collection(source, [*read_collection(source), (time, relative)])
    return write_text_atomic(source, text)


# One frame line exactly as _collection_line writes it, for a name and a time
# that need no XML escaping.
_COLLECTION_LINE = re.compile(r'    <DataSet timestep="([^"&<>]*)" file="([^"&<>]*)"/>\n')
_COLLECTION_BODY = re.compile(rf"(?:{_COLLECTION_LINE.pattern})*")


def _appended_collection(source: Path, time: float, name: str) -> str | None:
    """Return the collection text with one frame appended, or ``None`` if it cannot be.

    Appending needs a file in the form :func:`write_collection` writes, a
    finite time, a name without XML escapes, a frame that sorts after the
    last listed one and a file not listed yet. A file in that form is sorted
    and lists each frame once, because only this module writes it.
    """
    time = float(time)
    if not math.isfinite(time) or quoteattr(name) != f'"{name}"':
        return None
    try:
        text = source.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    if not (text.startswith(_COLLECTION_HEAD) and text.endswith(_COLLECTION_TAIL)):
        return None
    body = text[len(_COLLECTION_HEAD) : len(text) - len(_COLLECTION_TAIL)]
    if _COLLECTION_BODY.fullmatch(body) is None or f' file="{name}"/>\n' in body:
        return None
    if body:
        last = _COLLECTION_LINE.fullmatch(body, body.rfind("\n", 0, len(body) - 1) + 1)
        if (time, name) < (float(last.group(1)), last.group(2)):
            return None
    return _COLLECTION_HEAD + body + _collection_line(time, name) + _COLLECTION_TAIL


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
    tolerance = time_tolerance(time)
    kept = [(t, name) for t, name in entries if t <= time + tolerance]
    removed = [name for t, name in entries if t > time + tolerance]
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
