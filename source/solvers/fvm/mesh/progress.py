"""Live, solution-local progress logging for FVM mesh generation."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, TextIO


def _clean(value: Any) -> str:
    """Render one compact field value without allowing multiline log records."""
    rendered = str(value).replace("\r", " ").replace("\n", " ")
    return rendered if rendered else "-"


class MesherLog:
    """A line-buffered progress sink owned by one mesh-materialization run."""

    def __init__(self, path: str | Path) -> None:
        """Open a line-buffered meshing log at ``path``.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination text log. Parent directories are created.

        Side Effects
        ------------
        Opens the file in append mode, records a ``START`` event, and owns the
        stream until :meth:`close`.
        """
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existed = self.path.exists() and self.path.stat().st_size > 0
        self._stream: TextIO = self.path.open("a", encoding="utf-8", buffering=1)
        self._started = perf_counter()
        self._closed = False
        if existed:
            self._stream.write("\n")
        timestamp = datetime.now(UTC).isoformat(timespec="seconds")
        self.event("START", "meshing session", started_at=timestamp, log_file=self.path)

    def event(self, status: str, activity: str, **details: Any) -> None:
        """Append and immediately flush one progress record."""
        if self._closed:
            return
        elapsed = perf_counter() - self._started
        suffix = "".join(
            f" | {key}={_clean(value)}" for key, value in details.items() if value is not None
        )
        self._stream.write(f"{elapsed:10.3f}s  {status:<8} {activity}{suffix}\n")
        self._stream.flush()

    def close(self, *, failure: BaseException | None = None) -> None:
        """Finish the session with a durable success or failure record."""
        if self._closed:
            return
        if failure is None:
            self.event("COMPLETE", "meshing session")
        else:
            self.event(
                "FAILED",
                "meshing session",
                error_type=type(failure).__name__,
                error=failure,
            )
        self._closed = True
        self._stream.close()


_ACTIVE_MESHER_LOG: ContextVar[MesherLog | None] = ContextVar(
    "openonda_active_mesher_log", default=None
)


class MeshStage:
    """Context manager that records stage start, duration, details, and failure."""

    def __init__(self, activity: str) -> None:
        """Create a stage timer attached to the active :class:`MesherLog`.

        Parameters
        ----------
        activity : str
            Human-readable stage label used in progress events.

        Notes
        -----
        The stage is a cheap no-op when no mesher log is active and measures
        wall time only after entering the context manager.
        """
        self.activity = activity
        self._logger = _ACTIVE_MESHER_LOG.get()
        self._started = 0.0
        self._details: dict[str, Any] = {}

    def __enter__(self) -> MeshStage:
        self._started = perf_counter()
        if self._logger is not None:
            self._logger.event("START", self.activity)
        return self

    def details(self, **values: Any) -> None:
        """Attach values to the eventual completion record for this stage."""
        self._details.update(values)

    def __exit__(self, error_type, error, _traceback) -> bool:
        if self._logger is not None:
            elapsed = perf_counter() - self._started
            if error is None:
                self._logger.event("DONE", self.activity, seconds=f"{elapsed:.3f}", **self._details)
            else:
                self._logger.event(
                    "FAILED",
                    self.activity,
                    seconds=f"{elapsed:.3f}",
                    error_type=error_type.__name__,
                    error=error,
                    **self._details,
                )
        return False


def mesh_stage(activity: str) -> MeshStage:
    """Return a stage logger; outside a mesher session it is a cheap no-op."""
    return MeshStage(activity)


def mesh_event(activity: str, **details: Any) -> None:
    """Write an instantaneous progress event when a mesher session is active."""
    logger = _ACTIVE_MESHER_LOG.get()
    if logger is not None:
        logger.event("INFO", activity, **details)


@contextmanager
def mesher_log_session(
    path: str | Path | None,
    *,
    announce: bool = False,
) -> Iterator[MesherLog | None]:
    """Activate a live mesher log without adding paths to mesher public APIs."""
    if path is None:
        yield None
        return

    logger = MesherLog(path)
    if announce:
        print(f"Mesher log: {logger.path}", flush=True)
    token = _ACTIVE_MESHER_LOG.set(logger)
    try:
        yield logger
    except BaseException as error:
        logger.close(failure=error)
        raise
    else:
        logger.close()
    finally:
        _ACTIVE_MESHER_LOG.reset(token)


__all__ = ["MeshStage", "MesherLog", "mesh_event", "mesh_stage", "mesher_log_session"]
