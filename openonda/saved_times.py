"""Match saved physical clocks without interpolating or inventing states.

The symmetric tolerance is ``max(1e-12 s, 1e-10 * max(|a|, |b|))``.
It permits accumulated floating-point clock roundoff, not neighbouring output
steps. Ambiguous clocks in one source are rejected rather than rounded/merged.
This module deliberately has no solver, NumPy or ParaView dependencies.
"""

from bisect import bisect_left
from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

CLOCK_RTOL = 1.0e-10
CLOCK_ATOL = 1.0e-12


def same_saved_time(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=CLOCK_RTOL, abs_tol=CLOCK_ATOL)


def _validated_times(values) -> tuple[float, ...]:
    times = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0 for value in times):
        raise ValueError("Saved times must be finite and nonnegative")
    for left, right in zip(times, times[1:], strict=False):
        if right <= left or same_saved_time(left, right):
            raise ValueError("Saved times must be increasing and unambiguous at clock tolerance")
    return times


@dataclass(frozen=True)
class SavedTimeMatch:
    times: tuple[float, ...]
    # One index tuple per source, in the same order as times.
    indices: tuple[tuple[int, ...], ...]


def match_saved_times(*series) -> SavedTimeMatch:
    """Return common clocks and each source's native indices, without resampling."""
    clocks = tuple(_validated_times(values) for values in series)
    if not clocks:
        return SavedTimeMatch((), ())
    common, indices = [], [[] for _ in clocks]
    for base_index, time in enumerate(clocks[0]):
        row = [base_index]
        for other in clocks[1:]:
            insertion = bisect_left(other, time)
            candidates = [
                index for index in (insertion - 1, insertion)
                if 0 <= index < len(other) and same_saved_time(time, other[index])
            ]
            if len(candidates) > 1:
                raise ValueError(f"Ambiguous saved states near t={time:g}")
            if not candidates:
                break
            row.append(candidates[0])
        if len(row) == len(clocks):
            common.append(time)
            for column, index in zip(indices, row, strict=True):
                if column and index == column[-1]:
                    raise ValueError(f"One saved state matches multiple clocks near t={time:g}")
                column.append(index)
    return SavedTimeMatch(tuple(common), tuple(tuple(column) for column in indices))


def read_pvd_times(path: str | Path) -> tuple[float, ...]:
    """Read a collection's sorted clocks and require its referenced frames."""
    path = Path(path)
    frames = []
    checked = set()
    for item in ET.parse(path).iter("DataSet"):
        target = path.parent / item.attrib["file"]
        _require_frame(target, path, checked, set())
        frames.append(float(item.attrib["timestep"]))
    return _validated_times(sorted(frames))


def _require_frame(path: Path, collection: Path, checked: set, active: set) -> None:
    """Check small parallel/multiblock indexes, without reading volume payloads."""
    path = path.resolve()
    if path in active:
        raise ValueError(f"Cyclic saved-frame index in {collection}: {path}")
    if path in checked:
        return
    if not path.is_file():
        raise FileNotFoundError(f"Missing saved frame referenced by {collection}: {path}")
    if path.suffix.lower() in {".pvtu", ".pvtp", ".pvts", ".pvtr", ".pvti", ".vtm"}:
        active.add(path)
        for item in ET.parse(path).iter():
            for attribute in ("Source", "file"):
                if attribute in item.attrib:
                    _require_frame(path.parent / item.attrib[attribute], path, checked, active)
        active.remove(path)
    checked.add(path)
