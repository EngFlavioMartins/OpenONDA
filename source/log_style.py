"""Shared, bounded-width reports for OpenONDA console and file sinks.

Numerical owners pass already computed scalar values and short vectors. Rendering
never reads a solver, downloads a field, or performs a numerical reduction.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import logging
from numbers import Integral, Real
import textwrap
from typing import SupportsFloat, SupportsInt

WIDTH = 88
BLOCK_WIDTH = WIDTH
_VALUE_END = 63
_VALUE_START = 39
Row = tuple[str, object] | tuple[str, object, str]


_STEP_SECTION_ORDER = {
    name: index
    for index, name in enumerate(
        (
            "events",
            "time control",
            "particles",
            "fvm",
            "vpm",
            "convergence",
            "conservation",
            "energy",
            "flow integrals",
            "turbulence",
            "aerodynamic loads",
            "vlm wing/wake proximity",
            "stabilization",
            "interface transfer",
            "timing",
        )
    )
}


def step_section_order(title: str) -> int:
    """Keep common step sections in a fixed order, with timing last."""
    return _STEP_SECTION_ORDER.get(title.lower(), _STEP_SECTION_ORDER["timing"] - 1)


class FormattedText(str):
    """A complete shared-format report; sinks must not decorate it again."""


@dataclass(frozen=True, slots=True)
class Event:
    """An unformatted module report containing host-side measurements only.

    ``rows`` contains (label, value[, unit]) tuples. Values must be scalars,
    strings or short host vectors, never device fields or complete solutions.
    A callable may build rows from an immutable diagnostic result on demand.
    Conversion to text is delayed until an enabled sink actually emits it.
    """

    topic: str
    rows: tuple[Row, ...] | Callable[[], tuple[Row, ...]] = ()

    def __str__(self) -> str:
        return block_section(self.topic, self.rows() if callable(self.rows) else self.rows)


class Formatter(logging.Formatter):
    """Apply the shared layout to standard-library module logging as well."""

    def format(self, record: logging.LogRecord) -> str:
        title = "errors" if record.levelno >= logging.ERROR else "warnings"
        if isinstance(record.msg, FormattedText | Event):
            result = str(record.msg)
            if record.levelno >= logging.WARNING:
                result = block_section(title, []) + result
        else:
            result = block_section(
                title if record.levelno >= logging.WARNING else "events",
                [(record.getMessage(), "")],
            )
        if record.exc_info:
            result += block_section("exception", [(self.formatException(record.exc_info), "")])
        return result


def value_text(value: object) -> str:
    """Render one host scalar or short vector without allocating field copies."""
    if value is None:
        return "unavailable"
    if isinstance(value, bool):
        return "enabled" if value else "disabled"
    if isinstance(value, Integral):
        return f"{value:,}"
    if isinstance(value, Real):
        return f"{value:.4e}"
    if isinstance(value, tuple | list):
        return "[" + ", ".join(value_text(item) for item in value) + "]"
    return str(value)


def _detail(entry: Row) -> str:
    """Wrap long labels/paths instead of allowing them to break the columns."""
    label = str(entry[0]).strip()
    label = label[:1].upper() + label[1:]
    value = value_text(entry[1])
    unit = str(entry[2]) if len(entry) > 2 else ""
    if value == "":
        return textwrap.fill(label, WIDTH, initial_indent="  ", subsequent_indent="    ")
    suffix = f"  {unit}" if unit else ""
    if len(label) < _VALUE_START - 3 and len(value) + len(suffix) <= WIDTH - _VALUE_START:
        padding = max(_VALUE_START, _VALUE_END - len(value))
        return f"  {label:<{padding - 2}}{value}{suffix}"
    lines = textwrap.wrap(label, WIDTH - 2) or [""]
    result = ["  " + line for line in lines]
    result.append(
        textwrap.fill(value + suffix, WIDTH, initial_indent="    ", subsequent_indent="    ")
    )
    return "\n".join(result)


def block_section(title: str, rows: Iterable[Row], *, show_title: bool = True) -> FormattedText:
    """Render one populated section with a single gap before its heading."""
    lines = (
        ["", textwrap.fill(str(title).upper(), WIDTH, initial_indent=" ", subsequent_indent=" ")]
        if show_title
        else []
    )
    lines.extend(_detail(row) for row in rows)
    return FormattedText("\n".join(lines))


def section(title: str, rows: list[Row]) -> FormattedText:
    """Render an initialization section with the same layout as step sections."""
    return block_section(title, rows)


def record(scope: str, topic: str, *rows: Row) -> FormattedText:
    """Render a scoped module event through the common section formatter."""
    return block_section(f"{scope} {topic}", rows)


def elapsed_time(seconds: SupportsFloat) -> str:
    """Return elapsed seconds as HH:MM:SS.s, including runs over 24 hours."""
    total = max(0.0, float(seconds))
    hours = int(total // 3600.0)
    minutes = int((total - 3600.0 * hours) // 60.0)
    remaining = total - 3600.0 * hours - 60.0 * minutes
    return f"{hours:02d}:{minutes:02d}:{remaining:04.1f}"


def step_header(
    step: SupportsInt,
    flow_time: SupportsFloat,
    wall_time: SupportsFloat,
    *,
    scope: str = "VPM",
    total_steps: int | None = None,
) -> FormattedText:
    """Open a report for an accepted physical step and its elapsed wall time."""
    title = f" {scope.upper()} TIME STEP {int(step):,}"
    if total_steps is not None:
        title += f" / {total_steps:,}"
    clock = f" FLOW TIME {float(flow_time):.6e} s"
    elapsed = f"ELAPSED {elapsed_time(wall_time)}"
    clock += elapsed.rjust(max(1, WIDTH - len(clock)))
    return FormattedText("\n".join(("", "=" * WIDTH, title, clock, "-" * WIDTH)))


def block_report(title: str, sections: Iterable[tuple[str, Iterable[Row]]]) -> FormattedText:
    """Render one complete startup, completion or diagnostic report."""
    lines = [
        "",
        "=" * WIDTH,
        textwrap.fill(title.upper(), WIDTH, initial_indent=" ", subsequent_indent=" "),
        "-" * WIDTH,
    ]
    lines.extend(block_section(name, rows) for name, rows in sections if rows)
    return FormattedText("\n".join(lines))


def count(value: SupportsInt) -> str:
    """Format an integer count with thousands separators."""
    return f"{int(value):,}"


def quantity(value: SupportsFloat, digits: int = 3) -> str:
    """Format a scientific quantity with the requested decimal precision."""
    return f"{float(value):.{digits}e}"


def ratio(value: SupportsFloat, digits: int = 3) -> str:
    """Format a dimensionless ratio in fixed-point notation."""
    return f"{float(value):.{digits}f}"
