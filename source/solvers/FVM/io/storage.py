"""Capacity checks and recoverable append helpers for solver output."""

from __future__ import annotations

import errno
import os
from pathlib import Path
import shutil

_DEFAULT_RESERVE_BYTES = 1 << 30


class InsufficientStorageError(OSError):
    """Raised before an atomic output cannot fit alongside its old version."""

    def __init__(self, path: str | Path, required_bytes: int, free_bytes: int) -> None:
        self.path = Path(path)
        self.required_bytes = int(required_bytes)
        self.free_bytes = int(free_bytes)
        super().__init__(
            errno.ENOSPC,
            (
                f"insufficient free space for atomic output at {self.path}: "
                f"need {self.required_bytes} bytes including reserve, "
                f"have {self.free_bytes} bytes"
            ),
            str(self.path),
        )


def configured_reserve_bytes() -> int:
    """Return the disk reserve retained after a solver write."""
    raw = os.environ.get("FVM_DISK_RESERVE_BYTES")
    if raw is None:
        return _DEFAULT_RESERVE_BYTES
    reserve = int(raw)
    if reserve < 0:
        raise ValueError("FVM_DISK_RESERVE_BYTES must be non-negative")
    return reserve


def require_free_space(
    path: str | Path,
    payload_bytes: int,
    *,
    reserve_bytes: int | None = None,
) -> int:
    """Verify an atomic temporary payload can fit and return current free bytes."""
    destination = Path(path).resolve()
    probe = destination if destination.exists() else destination.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = int(shutil.disk_usage(probe).free)
    reserve = configured_reserve_bytes() if reserve_bytes is None else int(reserve_bytes)
    required = max(0, int(payload_bytes)) + max(0, reserve)
    if free < required:
        raise InsufficientStorageError(destination, required, free)
    return free


def append_line_recoverably(path: str | Path, line: str) -> None:
    """Append one text record and remove a partial tail if the write fails."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a+", encoding="utf-8") as stream:
        stream.seek(0, os.SEEK_END)
        original_size = stream.tell()
        try:
            stream.write(line)
            stream.flush()
        except OSError:
            # Truncation releases blocks and does not need additional capacity;
            # preserve a valid JSONL/CSV prefix for restart and post-processing.
            try:
                stream.seek(original_size)
                stream.truncate()
            except OSError:
                pass
            raise
