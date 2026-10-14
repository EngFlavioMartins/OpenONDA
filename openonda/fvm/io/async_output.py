"""Bounded background writer for FVM visualization snapshots."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import numpy as np

from openonda.core.vtk_output import read_collection, write_collection

from ..config.output import Output
from .vtk_exporter import VTKExporter


class BufferedVTKWriter:
    """Asynchronous VTK output writer with bounded memory.

    Uses a single background thread to serialise VTU snapshots and update
    the PVD collection file.  At most one snapshot is queued at a time:
    :meth:`submit` blocks until the previous write finishes, which bounds
    the memory used by pending output to one complete field copy.

    The :class:`VTKExporter` is created once on the worker thread.
    """

    def __init__(
        self,
        mesh_data: dict[str, object],
        collection: Path,
        output: Output,
    ) -> None:
        """Allocate a bounded one-worker VTK output queue.

        Parameters
        ----------
        mesh_data : dict
            Native FVM mesh consumed by :class:`VTKExporter`.
        collection : pathlib.Path
            ParaView collection listing the frames.
        output : Output
            Visualization precision and compression.

        Notes
        -----
        The worker thread and VTK objects are created lazily. Each submitted
        field is copied before the caller regains control; ``close()`` or
        ``flush()`` must be called to surface background errors.
        """
        self._mesh_data = mesh_data
        self._collection = Path(collection)
        self._output = output
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fvm-vtk")
        self._pending: Future | None = None
        self._closed = False
        self._lock = Lock()
        # The worker is single-threaded, so these objects are constructed once
        # on that worker and safely reused for every snapshot.
        self._exporter: VTKExporter | None = None
        self._entries: list[tuple[float, str]] | None = None

    def submit(self, filename: Path, time: float, fields: dict[str, np.ndarray]) -> None:
        """Queue one immutable snapshot, waiting for the previous write if needed."""
        snapshot = {
            name: np.ascontiguousarray(np.asarray(values)).copy() for name, values in fields.items()
        }
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot submit output after the buffered writer is closed")
            if self._pending is not None:
                self._pending.result()
            self._pending = self._executor.submit(
                self._write_snapshot, filename, float(time), snapshot
            )

    def _write_snapshot(self, filename: Path, time: float, fields: dict[str, np.ndarray]) -> None:
        if self._exporter is None:
            self._exporter = VTKExporter(self._mesh_data, self._output)
        if self._entries is None:
            self._entries = read_collection(self._collection)
        self._exporter.export(str(filename), fields)
        relative = (
            Path(filename).resolve().relative_to(self._collection.parent.resolve()).as_posix()
        )
        self._entries = [entry for entry in self._entries if entry[1] != relative]
        self._entries.append((time, relative))
        write_collection(self._collection, self._entries)

    def flush(self) -> None:
        """Wait for pending output and propagate any writer exception."""
        with self._lock:
            if self._pending is not None:
                self._pending.result()
                self._pending = None

    def rewind(self, time: float) -> None:
        """Flush pending output and reread the rewound collection at the next frame."""
        del time
        self.flush()
        self._entries = None

    def close(self) -> None:
        """Flush and stop the writer. This method is idempotent."""
        with self._lock:
            if self._closed:
                return
        try:
            self.flush()
        finally:
            with self._lock:
                self._closed = True
            self._executor.shutdown(wait=True)
