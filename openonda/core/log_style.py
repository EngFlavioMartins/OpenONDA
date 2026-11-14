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

from .schedules import EverySteps, Schedule
from .version import __version__

__all__ = [
    "Event",
    "FormattedText",
    "Row",
    "StepLog",
    "block_report",
    "block_section",
    "close_logging",
    "configure_logging",
    "count",
    "elapsed_time",
    "record",
    "run_header",
    "step_header",
    "step_section_order",
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
    """Case log file installed by :func:`configure_logging` or :class:`StepLog`."""


class _RunConsoleHandler(logging.StreamHandler):
    """Console output installed by :func:`configure_logging` or :class:`StepLog`."""


def _remove_run_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if isinstance(handler, _RunFileHandler | _RunConsoleHandler):
            logger.removeHandler(handler)
            handler.close()


def _install_run_handlers(logger: logging.Logger, path: str | Path | None, console: bool) -> None:
    """Replace the run handlers of ``logger`` by a log file and the console."""
    _remove_run_handlers(logger)
    logger.setLevel(logging.INFO)
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


def close_logging(name: str) -> None:
    """Remove and close the handlers installed by :func:`configure_logging`.

    Handlers installed by the caller stay attached.

    Parameters
    ----------
    name : str
        Logger name, for example ``"openonda.coupler"``.
    """
    _remove_run_handlers(logging.getLogger(name))


def configure_logging(
    name: str, path: str | Path | None, *, console: bool = True
) -> logging.Logger:
    """Send a standard-library logger to a case log file and the console.

    Both outputs use the shared :class:`Formatter`. Calling the function
    again replaces the handlers of the previous call; handlers installed by
    the caller stay attached. The logger does not propagate to the root
    logger.

    Parameters
    ----------
    name : str
        Logger name, for example ``"openonda.coupler"``.
    path : str or pathlib.Path or None
        Log file, opened for appending; missing parent directories are
        created. ``None`` writes no file.
    console : bool, default=True
        Also write to standard output.

    Returns
    -------
    logging.Logger
        The configured logger.
    """
    logger = logging.getLogger(name)
    _install_run_handlers(logger, path, console)
    return logger


class StepLog:
    """Step records and reports of one solver, written to its own log file.

    Each solver owns one instance. The instance writes through a logger of
    its own, so solvers in one process neither share nor overwrite each
    other's log file or step state. Rank 0 writes; the other ranks hold a
    disabled instance, so callers need no rank checks.

    A step is opened with :meth:`begin_step`. Until it is published, its
    sections and events are buffered: :meth:`section` keeps the latest rows
    per title, :meth:`event` keeps every event. :meth:`publish` writes the
    step header once per step, then the buffered sections in the common
    order of :func:`step_section_order`. Outside an open step, sections and
    events are written at once. Warnings are always written at once.

    Parameters
    ----------
    scope : str
        Solver name in the step header and the logger name, for example
        ``"FVM"``.
    path : str or pathlib.Path
        Log file, opened for appending.
    schedule : Schedule
        Accepted steps whose records are published.
    enabled : bool, default=True
        Write records; ``False`` on the ranks other than 0.
    console : bool, default=True
        Also write to standard output.
    debug : bool, default=False
        Accept the notes of :meth:`debug`.

    Attributes
    ----------
    muted : bool
        While true, :meth:`section`, :meth:`event` and :meth:`publish` write
        and buffer nothing. Warnings and reports are not affected.
    wall_time : float
        Sum in s of the ``elapsed`` arguments of :meth:`end_step`.
    n_steps : int
        Number of accepted steps passed to :meth:`end_step`.
    """

    def __init__(
        self,
        scope: str,
        path: str | Path,
        schedule: Schedule,
        *,
        enabled: bool = True,
        console: bool = True,
        debug: bool = False,
    ) -> None:
        self.scope = scope
        self.path = Path(path)
        self.schedule = schedule
        self.enabled = bool(enabled)
        self.debug_enabled = bool(debug)
        self.muted = False
        self.wall_time = 0.0
        self.n_steps = 0
        self._closed = False
        self._open: tuple[int, float, float] | None = None
        self._reported_step: int | None = None
        self._events: list[tuple[str, tuple[Row, ...]]] = []
        self._sections: dict[str, tuple[Row, ...]] = {}
        self._logger: logging.Logger | None = None
        if self.enabled:
            self._logger = logging.Logger(f"openonda.{scope.lower()}")
            _install_run_handlers(self._logger, self.path, console)

    @classmethod
    def disabled(cls) -> StepLog:
        """Return a log that writes nothing, for components used without a solver."""
        return cls("OpenONDA", Path(), EverySteps(1), enabled=False)

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has run."""
        return self._closed

    def _writes(self) -> bool:
        return self.enabled and not self._closed

    def _emit(self, text: str) -> None:
        if self._writes():
            self._logger.info(FormattedText(text))

    @property
    def in_step(self) -> bool:
        """Whether a step is open."""
        return self._open is not None

    def is_due(
        self, step: int, time: float, time_step_size: float, total_steps: int | None = None
    ) -> bool:
        """Return whether the schedule, or the last step of the run, selects the step."""
        return self.schedule.is_due(step, time, time_step_size) or (
            total_steps is not None and step == total_steps
        )

    def begin_step(self, step: int, time: float, time_step_size: float) -> None:
        """Open a step; the buffered records of an unpublished step are dropped.

        Parameters
        ----------
        step : int
            Index of the step.
        time : float
            Flow time of the step in s.
        time_step_size : float
            Step size in s.
        """
        self.abandon_step()
        self._open = (int(step), float(time), float(time_step_size))

    def abandon_step(self) -> None:
        """Close the open step and drop its buffered records."""
        self._open = None
        self._events.clear()
        self._sections.clear()

    def section(self, title: str, *rows: Row, immediate: bool = False) -> None:
        """Record one titled section: buffered in an open step, else written.

        Parameters
        ----------
        title : str
            Section title; a later section of the same title replaces the
            earlier one in the open step.
        *rows : Row
            ``(label, value[, unit])`` rows.
        immediate : bool, default=False
            Write the section at once, also in an open step.
        """
        if not self._writes() or self.muted:
            return
        if self._open is None or immediate:
            self._emit(block_section(title, rows))
        else:
            self._sections[title] = rows

    def event(self, topic: str, *rows: Row, immediate: bool = False) -> None:
        """Record an event: buffered in an open step, else written.

        Parameters
        ----------
        topic : str
            The event text, or the title of its detail rows.
        *rows : Row
            Optional detail rows.
        immediate : bool, default=False
            Write the event at once, also in an open step.
        """
        if not self._writes() or self.muted:
            return
        if self._open is None or immediate:
            self._emit(block_section(topic if rows else "events", rows or [(topic, "")]))
        else:
            self._events.append((topic, rows))

    def debug(self, text: str, *args: object) -> None:
        """Record an event in debug mode; ``text % args`` is formatted only then."""
        if self.debug_enabled:
            self.event(text % args if args else text)

    def warning(self, text: str, *rows: Row) -> None:
        """Write a warning at once, also when the step is not published.

        Without rows the warning is one line, led by the step while one is
        open. With rows it is a section titled ``text`` that repeats the text
        and shows the step as its first row.
        """
        if not self._writes():
            return
        step = None if self._open is None else self._open[0]
        if not rows:
            self._logger.warning(text if step is None else f"{self.scope} step {step}: {text}")
            return
        details: list[Row] = [(text, ""), *rows]
        if step is not None:
            details.insert(0, (f"{self.scope} step", step))
        self._logger.warning(block_section(text, details))

    def report(self, text: str) -> None:
        """Write one complete report, such as :func:`run_header` or :func:`block_report`."""
        self._emit(text)

    def publish(
        self,
        step: int,
        time: float,
        wall_time: float,
        *,
        total_steps: int | None = None,
        sections: Iterable[tuple[str, tuple[Row, ...]]] = (),
    ) -> None:
        """Write the step header, once per step, and the buffered sections.

        Parameters
        ----------
        step : int
            Index of the step.
        time : float
            Flow time in s.
        wall_time : float
            Elapsed wall time of the run in s.
        total_steps : int or None, default=None
            Last step of the run, shown in the header.
        sections : iterable of (str, tuple of Row), default=()
            Further sections, added to the buffered ones; a section replaces
            a buffered one of the same title.
        """
        if not self._writes() or self.muted:
            return
        parts: list[str] = []
        if self._reported_step != step:
            parts.append(
                step_header(step, time, wall_time, scope=self.scope, total_steps=total_steps)
            )
        sections = {**self._sections, **dict(sections)}
        if self._events:
            rows: list[Row] = []
            for topic, details in self._events:
                rows.append((topic, ""))
                rows.extend((f"  {detail[0]}", *detail[1:]) for detail in details)
            sections["events"] = tuple(rows)
        parts.extend(
            block_section(title, rows)
            for title, rows in sorted(
                sections.items(), key=lambda item: step_section_order(item[0])
            )
        )
        self._events.clear()
        self._sections.clear()
        if parts:
            self._emit("\n".join(parts))
        self._reported_step = step

    def end_step(self, elapsed: float, *, accepted: bool = True) -> bool:
        """Close the open step and publish it when it was accepted and is due.

        Parameters
        ----------
        elapsed : float
            Wall time of the step in s, also when it was not accepted.
        accepted : bool, default=True
            ``False`` drops the buffered records.

        Returns
        -------
        bool
            Whether the step was published.
        """
        self.wall_time += float(elapsed)
        opened, self._open = self._open, None
        if not accepted:
            self.abandon_step()
            return False
        self.n_steps += 1
        if opened is None or not self.is_due(*opened):
            self.abandon_step()
            return False
        self.publish(opened[0], opened[1], self.wall_time)
        return True

    def flush(self) -> None:
        """Flush the log handlers."""
        if self._writes():
            for handler in self._logger.handlers:
                handler.flush()

    def close(self) -> None:
        """Close the log file; later records are dropped. Idempotent."""
        if self._closed:
            return
        self.abandon_step()
        if self.enabled:
            _remove_run_handlers(self._logger)
        self._closed = True
