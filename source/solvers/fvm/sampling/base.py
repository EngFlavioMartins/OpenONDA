"""Common sampler abstraction shared by every FVM sampler.

A sampler is a small object that owns its output file name, its geometry and
its own deterministic cadence (:class:`RunSchedule`).  Sampling is driven
by the :class:`~source.solvers.fvm.sampling.executor.FVMSamplerExecutor`,
which runs after every accepted solver step and lets each sampler decide
whether it is due.  The same samplers drive live runs and offline
post-processing (:class:`~source.solvers.fvm.sampling.postprocess.PostProcess`),
so a schedule's decision must be reproducible from ``step`` / ``time``
alone — never from a mutable call counter.

Live sampler output lands in ``solver.samples_dir``, which defaults to
``<case_root>/samples/``. The directory belongs to the solver rather than to
individual samplers, so a named study case can route every sample coherently.

Examples
--------
>>> schedule = RunSchedule(every_n_steps=10)   # due at steps 0, 10, 20, ...
>>> schedule.is_due(0, 0.0, 0.01)
True
>>> schedule.is_due(5, 0.05, 0.01)
False
>>> RunSchedule(every_time=0.1).is_due(15, 0.15, 0.01)
True
"""

from __future__ import annotations

import csv
import os
from typing import Any, ClassVar
import xml.etree.ElementTree as ET

from ..config.scheduling import RunSchedule

# Canonical CSV column order for FVM field samplers. The leading coordinates
# and vector components match the VPM sampler so one reader serves both
# solvers. This list is the single source of truth for headers and rows.
SAMPLER_CSV_COLUMNS = [
    "position_x",
    "position_y",
    "position_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "vorticity_x",
    "vorticity_y",
    "vorticity_z",
    "kinematic_pressure",
]


def samples_dir(case_dir: str) -> str:
    """Return the default sampler output directory for offline post-processing."""
    return os.path.join(case_dir, "samples")


class Sampler:
    """Base class for FVM samplers.

    Subclasses implement :meth:`sample`, which returns a plain dict of
    canonical columns; the write methods are provided by the subclass.
    ``file_name`` defaults to the lower-cased class name without the
    ``Sampler`` suffix.

    Every concrete sampler participates in :data:`SAMPLER_REGISTRY` via
    :meth:`config_dict` / :meth:`from_config`, giving a stable, JSON-safe
    representation used for ``FVMSetup.save()/load()`` and configuration
    hashing.

    Examples
    --------
    >>> class MySampler(Sampler):
    ...     def sample(self, context):
    ...         return {"x": [0.0]}
    >>> MySampler(file_name="probe").name
    'probe'
    """

    sampler_kind: ClassVar[str] = "Sampler"

    def __init__(
        self,
        file_name: str | None = None,
        schedule: RunSchedule | None = None,
    ) -> None:
        self.file_name = file_name
        self.schedule = schedule if schedule is not None else RunSchedule(every_n_steps=1)

    @property
    def name(self) -> str:
        """Output stem: ``file_name`` or the class name without the suffix."""
        return self.file_name or self.__class__.__name__.lower().replace("sampler", "")

    def is_due(self, step: int, time: float, time_step_size: float | None = None) -> bool:
        """Whether this sampler runs at the given time/step."""
        return self.schedule.is_due(step, time, time_step_size)

    def __eq__(self, other) -> bool:
        """Samplers are equal when they reconstruct the same configuration.

        This is what makes ``FVMSetup.save()/load()`` round-trips and
        ``config_hash`` stable: two equivalent explicit samplers compare equal
        without relying on object identity.
        """
        return type(self) is type(other) and self.config_dict() == other.config_dict()

    def __hash__(self) -> int:
        items = tuple(sorted(self.config_dict().items()))
        return hash((type(self).__name__, items))

    def sample(self, context) -> dict[str, Any] | None:
        raise NotImplementedError

    def config_dict(self) -> dict:
        """Constructor keyword arguments for this sampler (JSON-safe)."""
        return {
            "file_name": self.file_name,
            "schedule": self.schedule.to_dict(),
        }

    @classmethod
    def from_config(cls, data: dict) -> Sampler:
        data = dict(data)
        schedule = data.pop("schedule", None)
        return cls(**data, schedule=RunSchedule.from_dict(schedule))


_SAMPLER_REGISTRY: dict[str, type[Sampler]] = {}


def _register_sampler(cls: type[Sampler]) -> type[Sampler]:
    _SAMPLER_REGISTRY[cls.__name__] = cls
    return cls


def sampler_to_dict(sampler: Sampler) -> dict:
    """Stable JSON-safe spec used by config hashing and ``FVMSetup.save``."""
    return {"type": type(sampler).__name__, **sampler.config_dict()}


def sampler_from_dict(spec: dict) -> Sampler:
    """Rebuild a sampler from the :func:`sampler_to_dict` specification."""
    spec = dict(spec)
    kind = spec.pop("type")
    try:
        cls = _SAMPLER_REGISTRY[kind]
    except KeyError:
        raise ValueError(f"Unknown sampler type {kind!r} in configuration") from None
    return cls.from_config(spec)


def append_csv_rows(
    filepath: str,
    header: list[str],
    rows: list[list[Any]],
) -> None:
    """Append ``rows`` under ``header`` to a CSV file, writing the header once."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    write_header = not os.path.exists(filepath) or os.path.getsize(filepath) == 0
    with open(filepath, "a", newline="") as stream:
        writer = csv.writer(stream)
        if write_header:
            writer.writerow(header)
        writer.writerows(rows)


def write_pvd(samples_dir: str, name: str, entries: list[tuple[float, str]]) -> None:
    """Atomically merge a ParaView time-series index across restarts."""
    destination = os.path.join(samples_dir, f"{name}.pvd")
    merged: dict[str, float] = {}
    if os.path.isfile(destination):
        tree = ET.parse(destination)
        for dataset in tree.findall(".//DataSet"):
            filename = dataset.attrib.get("file")
            timestep = dataset.attrib.get("timestep")
            if filename is not None and timestep is not None:
                merged[filename] = float(timestep)
    for time_val, filename in entries:
        merged[str(filename)] = float(time_val)
    ordered = sorted(
        ((time_val, filename) for filename, time_val in merged.items()),
        key=lambda item: (item[0], item[1]),
    )
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    lines += [
        f'    <DataSet timestep="{time_val}" file="{filename}"/>' for time_val, filename in ordered
    ]
    lines += ["  </Collection>", "</VTKFile>"]
    temporary = destination + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
