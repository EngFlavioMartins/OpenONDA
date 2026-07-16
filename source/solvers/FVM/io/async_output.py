"""Bounded background writer for FVM visualization snapshots."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock

import numpy as np

from .vtk_exporter import PVDManager, VTKExporter


class BufferedVTKWriter:
    """Serialize VTU/PVD output on one worker while bounding queued memory."""

    def __init__(self, mesh_data, pvd_path: str):
        self._mesh_data = mesh_data
        self._pvd_path = pvd_path
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fvm-vtk")
        self._pending: Future | None = None
        self._closed = False
        self._lock = Lock()

    def submit(self, filename: str, time: float, fields: dict[str, np.ndarray]) -> None:
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

    def _write_snapshot(self, filename: str, time: float, fields: dict[str, np.ndarray]) -> None:
        exporter = VTKExporter(self._mesh_data)
        exporter.export(filename, fields)
        PVDManager(self._pvd_path).add_step(time, filename)

    def flush(self) -> None:
        """Wait for pending output and propagate any writer exception."""
        with self._lock:
            if self._pending is not None:
                self._pending.result()
                self._pending = None

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
