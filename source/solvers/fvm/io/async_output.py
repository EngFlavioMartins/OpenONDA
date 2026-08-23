"""Bounded background writer for FVM visualization snapshots."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock

import numpy as np

from ..config.types import OutputConfig
from .vtk_exporter import PVDManager, VTKExporter


class BufferedVTKWriter:
    """Asynchronous VTK output writer with bounded memory.

    Uses a single background thread to serialise VTU snapshots and update
    the PVD collection file.  At most one snapshot is queued at a time:
    :meth:`submit` blocks until the previous write finishes, which bounds
    the memory used by pending output to one complete field copy.

    The writer reuses its :class:`VTKExporter` and :class:`PVDManager`
    across calls — they are created once on the worker thread.

    Examples
    --------
    >>> writer = BufferedVTKWriter(mesh_data, "solution/sim.pvd")
    >>> writer.submit("step_0001.vtu", 0.1, fields)
    >>> writer.flush()
    >>> writer.close()
    """

    def __init__(
        self,
        mesh_data,
        pvd_path: str,
        output: OutputConfig | None = None,
    ):
        self._mesh_data = mesh_data
        self._pvd_path = pvd_path
        self._output = output or OutputConfig()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fvm-vtk")
        self._pending: Future | None = None
        self._closed = False
        self._lock = Lock()
        # The worker is single-threaded, so these objects are constructed once
        # on that worker and safely reused for every snapshot.
        self._exporter: VTKExporter | None = None
        self._pvd: PVDManager | None = None

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
        if self._exporter is None:
            self._exporter = VTKExporter(self._mesh_data, self._output)
        if self._pvd is None:
            self._pvd = PVDManager(self._pvd_path)
        self._exporter.export(filename, fields)
        self._pvd.add_step(time, filename)

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
