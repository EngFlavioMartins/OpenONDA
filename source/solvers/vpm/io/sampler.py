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
from .sampling import resolve_samples_dir, sampler_csv_columns


class SamplerExecutor:
    """Orchestrates the sampler objects declared by ``VPMSetup.samplers``.

    All methods are static — no per-instance state is held here.  The
    solver passes itself in and this class reads what it needs (particles,
    config, time counters, backup directory) without tight coupling.
    """

    @staticmethod
    def selected(solver, mode: str) -> tuple:
        """Return the samples selected for ``scheduled``, ``manual``, or ``final`` execution."""
        selected = []
        for sampler in solver.setup.samplers.samples:
            schedule = getattr(sampler, "schedule", None)
            final_only = schedule is not None and schedule.at_end
            if mode == "scheduled":
                if (
                    schedule is None
                    or final_only
                    or not schedule.is_due(
                        solver.step,
                        solver.time,
                        solver.time_step_size,
                    )
                ):
                    continue
            elif mode == "manual":
                if final_only:
                    continue
            elif mode == "final":
                if not final_only:
                    continue
            else:
                raise ValueError(f"Unknown sampler execution mode {mode!r}")
            selected.append(sampler)
        return tuple(selected)

    @staticmethod
    def any_due(solver, step: int, time: float) -> bool:
        """Return whether any configured sample is due on an accepted state."""
        for sampler in solver.setup.samplers.samples:
            schedule = getattr(sampler, "schedule", None)
            if (
                schedule is not None
                and not schedule.at_end
                and schedule.is_due(step, time, solver.time_step_size)
            ):
                return True
        return False

    @staticmethod
    def flow_integrals_due(solver, step: int, time: float) -> bool:
        """Return whether a due sampler consumes the integral diagnostics."""
        for sampler in solver.setup.samplers.samples:
            schedule = getattr(sampler, "schedule", None)
            if (
                getattr(sampler, "requires_flow_integrals", False)
                and schedule is not None
                and not schedule.at_end
                and schedule.is_due(step, time, solver.time_step_size)
            ):
                return True
        return False

    @staticmethod
    def execute(solver, *, mode: str) -> None:
        """Execute one canonical selection of configured sampler objects."""
        selected_samples = SamplerExecutor.selected(solver, mode)
        if not selected_samples:
            return

        if any(getattr(sample, "requires_flow_integrals", False) for sample in selected_samples):
            if getattr(solver, "_flow_integrals_step", None) != solver.step:
                solver._update_all_flow_integrals()

        n_particles_total = solver.particles.n_particles_total
        if n_particles_total < 2:
            return

        vortex_strength = solver.particle_vortex_strength
        max_vortex_strength_magnitude = (
            np.max(np.linalg.norm(vortex_strength, axis=1)) if n_particles_total > 0 else 0.0
        )
        if max_vortex_strength_magnitude < 1e-8:
            return

        samples_dir = resolve_samples_dir(
            solver.case_dir,
            solver.setup.samplers.directory,
        )
        samples_dir.mkdir(parents=True, exist_ok=True)

        for sampler in selected_samples:
            name_prefix = getattr(sampler, "file_name", None)
            if not name_prefix:
                name_prefix = sampler.__class__.__name__.lower().removesuffix("sampler")
            solution_dir = samples_dir
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
        columns = sampler_csv_columns(sampler)
        missing = [name for name in columns if name not in data]
        if missing:
            raise ValueError(f"Sampler result is missing columns: {', '.join(missing)}")

        lengths = {len(np.asarray(data[name])) for name in columns}
        if len(lengths) != 1:
            raise ValueError("Sampler result columns do not all have the same length")

        filepath.parent.mkdir(parents=True, exist_ok=True)
        write_header = not filepath.exists() or filepath.stat().st_size == 0
        with filepath.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            if write_header:
                writer.writerow(["time", "step", *columns])
            step = "" if step is None else int(step)
            for values in zip(
                *(np.asarray(data[name]).reshape(-1) for name in columns),
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
