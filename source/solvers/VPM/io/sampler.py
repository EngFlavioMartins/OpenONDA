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
from pathlib import Path

import numpy as np

from ..utils.field_samplers import SAMPLER_CSV_COLUMNS


class SamplerExecutor:
    """Orchestrates field-sampler execution for one solver log step.

    All methods are static — no per-instance state is held here.  The
    solver passes itself in and this class reads what it needs (particles,
    config, time counters, backup directory) without tight coupling.
    """

    @staticmethod
    def execute(solver, sampler_entries=None) -> None:
        """Execute all configured samplers and persist their output.

        Reads ``solver.config.samplers`` (list of sampler or
        ``(sampler, name_prefix)`` tuples) and, for each one, determines
        the target directory / file name, calls the appropriate ``save_*``
        method on the sampler, and keeps the PVD time-series index up to
        date.

        Sampling is skipped when the particle field is empty or numerically
        insignificant (< 2 particles or max |Γ| < 1e-8) to avoid crashes
        inside pressure-gradient or SVD routines.
        """
        sampler_entries = solver.config.samplers if sampler_entries is None else sampler_entries
        if sampler_entries is None:
            return

        n_particles = solver.particles.number_of_particles
        if n_particles < 2:
            return

        circ = solver.particles_circulation
        max_circ_mag = np.max(np.linalg.norm(circ, axis=1)) if n_particles > 0 else 0.0
        if max_circ_mag < 1e-8:
            return

        samples_dir = Path(solver.backup_directory) / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        for sampler_entry in sampler_entries:
            sampler, name_prefix, solution_dir = SamplerExecutor._prepare_context(
                sampler_entry, samples_dir
            )
            if not hasattr(sampler, "_call_count"):
                sampler._call_count = 0
                sampler._pvd_entries = []
            sampler._call_count += 1

            seq_num = f"{solver.time_step:06d}"
            SamplerExecutor._save_output(
                sampler,
                solver,
                name_prefix,
                solution_dir,
                seq_num,
                solver.flow_time,
                solver.time_step,
            )

    # ---- Helpers ----

    @staticmethod
    def _prepare_context(sampler_entry, samples_dir: Path):
        """Unpack a sampler entry and resolve *name_prefix* and *solution_dir*."""
        if isinstance(sampler_entry, tuple):
            sampler, name_prefix = sampler_entry
        else:
            sampler = sampler_entry
            name_prefix = getattr(sampler, "file_name", None)
            if name_prefix is None:
                name_prefix = sampler.__class__.__name__.lower().replace("sampler", "")

        output_dir = getattr(sampler, "output_dir", None)
        if output_dir is not None:
            solution_dir = Path(output_dir)
            solution_dir.mkdir(parents=True, exist_ok=True)
        else:
            solution_dir = samples_dir

        return sampler, name_prefix, solution_dir

    @staticmethod
    def _save_output(
        sampler,
        solver,
        name_prefix: str,
        solution_dir: Path,
        seq_num: str,
        flow_time: float,
        time_step: int | None = None,
    ) -> None:
        """Persist one sampler event using the canonical format for its geometry."""
        try:
            if hasattr(sampler, "save_vtp"):
                filename = f"{name_prefix}_{seq_num}.vts"
                filepath = solution_dir / filename
                sampler.save_vtp(solver, filepath, time=flow_time)
                sampler._pvd_entries.append((flow_time, filename))
                SamplerExecutor._write_pvd(solution_dir, name_prefix, sampler._pvd_entries)
            else:
                SamplerExecutor._append_csv(
                    sampler,
                    solver,
                    solution_dir / f"{name_prefix}.csv",
                    flow_time,
                    time_step,
                )
        except Exception as exc:
            print(f"(Warning) Sampler '{name_prefix}' failed: {exc}")

    @staticmethod
    def _append_csv(
        sampler,
        solver,
        filepath: Path,
        flow_time: float,
        time_step: int | None,
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
            writer = csv.writer(stream)
            if write_header:
                writer.writerow(["flow_time", "time_step", *SAMPLER_CSV_COLUMNS])
            step = "" if time_step is None else int(time_step)
            for values in zip(
                *(np.asarray(data[name]).reshape(-1) for name in SAMPLER_CSV_COLUMNS),
                strict=True,
            ):
                writer.writerow([float(flow_time), step, *values])

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
