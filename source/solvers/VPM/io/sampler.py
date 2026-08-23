"""
Sampler execution module for the VPM solver.

Handles executing field samplers (SurfaceSampler, LineSampler, etc.) and
serialising their output to disk (VTS/CSV) with PVD time-series support.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: March 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from __future__ import annotations

import csv
import inspect
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .logging import Logging
from .sampling import SAMPLER_CSV_COLUMNS, resolve_samples_dir


class SamplerExecutor:
    """Orchestrates field-sampler execution for one solver log step.

    All methods are static — no per-instance state is held here.  The
    solver passes itself in and this class reads what it needs (particles,
    config, time counters, checkpoint directory) without tight coupling.
    """

    @staticmethod
    def execute(solver, sampler_entries=None, *, scheduled_only: bool | None = False) -> None:
        """Execute all configured samplers and persist their output.

        Reads ``solver.setup.samplers`` (list of sampler or
        ``(sampler, name_prefix)`` tuples) and, for each one, determines
        the target directory / file name, calls the appropriate ``save_*``
        method on the sampler, and keeps the PVD time-series index up to
        date.

        Sampling is skipped when the particle field is empty or numerically
        insignificant (< 2 particles or max |vortex_strength| < 1e-8) to avoid crashes
        inside pressure-gradient or SVD routines.
        """
        sampler_entries = solver.setup.samplers if sampler_entries is None else sampler_entries
        if sampler_entries is None:
            return

        selected_entries = []
        for sampler_entry in sampler_entries:
            sampler = sampler_entry[0] if isinstance(sampler_entry, tuple) else sampler_entry
            schedule = getattr(sampler, "schedule", None)
            if scheduled_only is True:
                if schedule is None or not schedule.is_due(
                    solver.step,
                    solver.time,
                    solver.time_step_size,
                ):
                    continue
            elif scheduled_only is False and schedule is not None:
                continue
            selected_entries.append(sampler_entry)
        if not selected_entries:
            return

        n_particles = solver.particles.n_particles
        if n_particles < 2:
            return

        vortex_strength = solver.particle_vortex_strength
        max_vortex_strength_magnitude = (
            np.max(np.linalg.norm(vortex_strength, axis=1)) if n_particles > 0 else 0.0
        )
        if max_vortex_strength_magnitude < 1e-8:
            return

        samples_dir = resolve_samples_dir(
            solver.case_dir,
            getattr(solver.setup, "sample_subdirectory", None),
        )
        samples_dir.mkdir(parents=True, exist_ok=True)

        for sampler_entry in selected_entries:
            sampler, name_prefix, solution_dir = SamplerExecutor._prepare_context(
                sampler_entry, samples_dir
            )
            if not hasattr(sampler, "_call_count"):
                sampler._call_count = 0
                sampler._pvd_entries = SamplerExecutor._read_pvd(solution_dir, name_prefix)
            sampler._call_count += 1

            seq_num = f"{solver.step:06d}"
            SamplerExecutor._save_output(
                sampler,
                solver,
                name_prefix,
                solution_dir,
                seq_num,
                solver.time,
                solver.step,
            )

    # ---- Helpers ----

    @staticmethod
    def _prepare_context(sampler_entry, samples_dir: Path):
        """Unpack a sampler entry and resolve *name_prefix*.

        The destination directory is always ``samples_dir`` — sampler output
        location is not configurable per-call.
        """
        if isinstance(sampler_entry, tuple):
            sampler, name_prefix = sampler_entry
        else:
            sampler = sampler_entry
            name_prefix = getattr(sampler, "file_name", None)
            if name_prefix is None:
                name_prefix = sampler.__class__.__name__.lower().replace("sampler", "")

        return sampler, name_prefix, samples_dir

    @staticmethod
    def _save_output(
        sampler,
        solver,
        name_prefix: str,
        solution_dir: Path,
        seq_num: str,
        time: float,
        step: int | None = None,
    ) -> None:
        """Persist one sampler event using the canonical format for its geometry."""
        try:
            if hasattr(sampler, "save_vtp"):
                filename = f"{name_prefix}_{seq_num}.vts"
                filepath = solution_dir / filename
                sampler.save_vtp(solver, filepath, time=time)
                sampler._pvd_entries = [
                    entry for entry in sampler._pvd_entries if entry[1] != filename
                ]
                sampler._pvd_entries.append((time, filename))
                sampler._pvd_entries.sort(key=lambda entry: (entry[0], entry[1]))
                SamplerExecutor._write_pvd(solution_dir, name_prefix, sampler._pvd_entries)
            elif getattr(sampler, "csv_time_series", False):
                SamplerExecutor._append_csv(
                    sampler,
                    solver,
                    solution_dir / f"{name_prefix}.csv",
                    time,
                    step,
                )
            elif hasattr(sampler, "save_csv"):
                save_csv = sampler.save_csv
                keywords = {"time": time}
                if "step" in inspect.signature(save_csv).parameters:
                    keywords["step"] = step
                save_csv(solver, solution_dir / f"{name_prefix}.csv", **keywords)
            else:
                SamplerExecutor._append_csv(
                    sampler,
                    solver,
                    solution_dir / f"{name_prefix}.csv",
                    time,
                    step,
                )
        except Exception as exc:
            Logging.warning(f"component=sampler name={name_prefix!r} status=failed error={exc!r}")

    @staticmethod
    def _append_csv(
        sampler,
        solver,
        filepath: Path,
        time: float,
        step: int | None,
    ) -> None:
        """Append one complete sampled field to a time-aware CSV table."""
        data = sampler.sample(solver)
        missing = [name for name in SAMPLER_CSV_COLUMNS if name not in data]
        if missing:
            raise ValueError(f"Sampler result is missing columns: {', '.join(missing)}")

        lengths = {len(np.asarray(data[name])) for name in SAMPLER_CSV_COLUMNS}
        if len(lengths) != 1:
            raise ValueError("Sampler result columns do not all have the same length")

        filepath.parent.mkdir(parents=True, exist_ok=True)
        write_header = not filepath.exists() or filepath.stat().st_size == 0
        with filepath.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            if write_header:
                writer.writerow(["time", "step", *SAMPLER_CSV_COLUMNS])
            step = "" if step is None else int(step)
            for values in zip(
                *(np.asarray(data[name]).reshape(-1) for name in SAMPLER_CSV_COLUMNS),
                strict=True,
            ):
                writer.writerow([float(time), step, *values])

    @staticmethod
    def _write_pvd(output_dir: Path, name_prefix: str, entries: list) -> None:
        """Write a PVD (ParaView Data Collection) XML index for time-series playback.

        Parameters
        ----------
        output_dir:
            Directory where ``<name_prefix>.pvd`` will be written.
        name_prefix:
            Base name used for the PVD file and referenced VTS files.
        entries:
            List of ``(time_value, filename)`` tuples, one per saved step.
        """
        pvd_path = output_dir / f"{name_prefix}.pvd"
        lines = [
            '<?xml version="1.0"?>',
            '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
            "  <Collection>",
        ]
        for time_val, filename in entries:
            lines.append(f'    <DataSet timestep="{time_val}" file="{filename}"/>')
        lines.append("  </Collection>")
        lines.append("</VTKFile>")
        with open(pvd_path, "w") as fh:
            fh.write("\n".join(lines))

    @staticmethod
    def _read_pvd(output_dir: Path, name_prefix: str) -> list[tuple[float, str]]:
        """Read an existing surface-sample index when a run is resumed."""
        pvd_path = output_dir / f"{name_prefix}.pvd"
        if not pvd_path.is_file():
            return []

        try:
            # The PVD index is written by this solver in this directory; the
            # same judgement the ruff suppression records applies to bandit.
            root = ET.parse(pvd_path).getroot()  # noqa: S314  # nosec B314
            entries = []
            for dataset in root.findall(".//DataSet"):
                filename = dataset.get("file")
                timestep = dataset.get("timestep")
                if filename is not None and timestep is not None:
                    entries.append((float(timestep), filename))
            return entries
        except (ET.ParseError, OSError, ValueError) as exc:
            Logging.warning(
                f"component=sampler_index path={str(pvd_path)!r} status=read_failed error={exc!r}"
            )
            return []
