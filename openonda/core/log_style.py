"""Shared, bounded-width reports for OpenONDA console and file sinks.

Solvers pass already computed scalar values and short vectors. Rendering
never reads a solver, downloads a field, or performs a numerical reduction.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
import logging
from numbers import Integral, Real
from pathlib import Path
import platform
import sys
import textwrap
from typing import SupportsFloat, SupportsInt

from .version import __version__

__all__ = [
    "WIDTH",
    "Event",
    "FormattedText",
    "Formatter",
    "Row",
    "block_report",
    "block_section",
    "close_logging",
    "configure_logging",
    "count",
    "elapsed_time",
    "quantity",
    "ratio",
    "record",
    "run_header",
    "section",
    "step_header",
    "step_section_order",
    "value_text",
]

WIDTH = 88
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
            "vorticity transfer",
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


def block_section(title: str, rows: Iterable[Row]) -> FormattedText:
    """Render one populated section with a single gap before its heading."""
    lines = [
        "",
        textwrap.fill(str(title).upper(), WIDTH, initial_indent=" ", subsequent_indent=" "),
    ]
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


def run_header(solver_name: str, precision: str, *rows: Row) -> FormattedText:
    """Return the report that opens a solver run.

    Parameters
    ----------
    solver_name : str
        Solver shown in the title, for example ``"FVM"`` or ``"VPM"``.
    precision : str
        Floating-point precision of the solution fields, for example
        ``"f64"``.
    *rows : Row
        Further ``(label, value[, unit])`` rows, for example the compute
        backend, appended after the common rows.

    Returns
    -------
    FormattedText
        ``OpenONDA <solver_name>`` report with the OpenONDA version, the
        precision, the start date, the platform and the Python version.
    """
    return block_report(
        f"OpenONDA {solver_name}",
        [
            (
                "run",
                [
                    ("version", __version__),
                    ("precision", precision),
                    ("started", f"{datetime.now():%Y-%m-%d %H:%M:%S}"),
                    ("platform", f"{platform.system()} {platform.machine()}"),
                    ("Python", platform.python_version()),
                    *rows,
                ],
            )
        ],
    )


class _RunFileHandler(logging.FileHandler):
    """Case log file installed by :func:`configure_logging`."""


class _RunConsoleHandler(logging.StreamHandler):
    """Console output installed by :func:`configure_logging`."""


def close_logging(name: str) -> None:
    """Remove and close the handlers installed by :func:`configure_logging`.

    Handlers installed by the caller stay attached.

    Parameters
    ----------
    name : str
        Logger name, for example ``"openonda.fvm"``.
    """
    logger = logging.getLogger(name)
    for handler in list(logger.handlers):
        if isinstance(handler, _RunFileHandler | _RunConsoleHandler):
            logger.removeHandler(handler)
            handler.close()


def configure_logging(
    name: str,
    path: str | Path | None,
    *,
    console: bool = True,
    level: int = logging.INFO,
) -> logging.Logger:
    """Send a standard-library logger to a case log file and the console.

    Both outputs use the shared :class:`Formatter`. Calling the function
    again replaces the handlers of the previous call; handlers installed by
    the caller stay attached. The logger does not propagate to the root
    logger.

    Parameters
    ----------
    name : str
        Logger name, for example ``"openonda.fvm"``.
    path : str or pathlib.Path or None
        Log file, opened for appending; missing parent directories are
        created. ``None`` writes no file.
    console : bool, default=True
        Also write to standard output.
    level : int, default=logging.INFO
        Lowest level that is written.

    Returns
    -------
    logging.Logger
        The configured logger.
    """
    close_logging(name)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    formatter = Formatter()
    if path is not None:
        log_path = Path(path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = _RunFileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if console:
        console_handler = _RunConsoleHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger
