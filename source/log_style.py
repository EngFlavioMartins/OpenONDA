"""One writing style for every OpenONDA log.

A log is a sequence of records. A record opens with a header line naming the
scope that produced it and the topic it reports, and continues with indented
detail rows that carry one quantity each: label on the left, value in a fixed
column, unit last. Startup material uses the same detail rows under a ruled
section title, and a coupling step opens with a banner.

Values are right-aligned so that short ones end on a common column and long
ones run rightwards from a common column; no call site pads by hand.
"""

from __future__ import annotations

import time
from typing import SupportsFloat, SupportsInt

WIDTH = 78

_SCOPE_WIDTH = 7
_VALUE_WIDTH = 12
_VALUE_END = 56

_RECORD_INDENT = 14
_RECORD_LABEL_WIDTH = _VALUE_END - _VALUE_WIDTH - _RECORD_INDENT

_SECTION_INDENT = 2
_SECTION_LABEL_WIDTH = _VALUE_END - _VALUE_WIDTH - _SECTION_INDENT

Row = tuple[str, object] | tuple[str, object, str]


def stamp() -> str:
    """Return the wall-clock prefix used by logs that write without a handler."""
    return time.strftime("%H:%M:%S")


def _detail(indent: int, label_width: int, entry: Row) -> str:
    label, value = entry[0], entry[1]
    unit = entry[2] if len(entry) > 2 else ""
    line = f"{' ' * indent}{label:<{label_width}}{str(value):>{_VALUE_WIDTH}}"
    if unit:
        line = f"{line}  {unit}"
    return line.rstrip()


def header(scope: str, topic: str, *, stamped: bool = False) -> str:
    """Return a record header naming the scope and what it reports."""
    line = f"{scope:<{_SCOPE_WIDTH}}  {topic}".rstrip()
    return f"{stamp()}  {line}" if stamped else line


def record(scope: str, topic: str, *rows: Row, stamped: bool = False) -> str:
    """Return one record: a scope and topic header over indented detail rows."""
    lines = [header(scope, topic, stamped=stamped)]
    lines.extend(_detail(_RECORD_INDENT, _RECORD_LABEL_WIDTH, row) for row in rows)
    return "\n".join(lines)


def section(title: str, rows: list[Row]) -> str:
    """Return a ruled startup section over the same detail rows."""
    rule = "-" * WIDTH
    lines = ["", rule, f" {title}", rule]
    lines.extend(_detail(_SECTION_INDENT, _SECTION_LABEL_WIDTH, row) for row in rows)
    return "\n".join(lines)


def banner(left: str, right: str = "") -> str:
    """Return a heavy-ruled banner opening a step or a major phase."""
    rule = "=" * WIDTH
    title = f" {left}"
    if right:
        title = f"{title}{right:>{max(1, WIDTH - len(title))}}"
    return "\n".join(("", rule, title, rule))


def count(value: SupportsInt) -> str:
    """Return an integer count with thousands separators."""
    return f"{int(value):,}"


def quantity(value: SupportsFloat, digits: int = 3) -> str:
    """Return a scientific-notation value with a fixed number of digits."""
    return f"{float(value):.{digits}e}"


def ratio(value: SupportsFloat, digits: int = 3) -> str:
    """Return a dimensionless ratio in fixed-point notation."""
    return f"{float(value):.{digits}f}"


__all__ = [
    "WIDTH",
    "banner",
    "count",
    "header",
    "quantity",
    "ratio",
    "record",
    "section",
    "stamp",
]
