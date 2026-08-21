"""
Panel diagnostics module — recording and CSV export of panel force/circulation history.

Mirrors ``vlm/solver/diagnostics.py`` conventions.  Owns all logic for:
  - Appending per-step panel scalars to the solver diagnostics history dict.
  - Writing panel_forces.csv with per-group force decomposition.
  - Appending time / observed_dt history entries.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: June 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from __future__ import annotations

from pathlib import Path


class PanelDiagnostics:
    """Static helpers for recording panel diagnostics and writing CSV output."""

    @staticmethod
    def record_time(diagnostics_history: dict, time: float, observed_time_step_size: float) -> None:
        if "flow_time" not in diagnostics_history:
            return
        ft_hist = diagnostics_history["flow_time"]
        if len(ft_hist) > 0 and ft_hist[-1] == time:
            return
        ft_hist.append(time)
        if len(diagnostics_history["observed_dt"]) < len(ft_hist):
            diagnostics_history["observed_dt"].append(float(observed_time_step_size))

    @staticmethod
    def record_panel_diagnostics(
        panel_solver,
        diagnostics_history: dict,
        step: int,
        time: float,
        case_dir: str,
    ) -> None:
        if panel_solver is None:
            return
        try:
            n_panels = panel_solver.lattice.num_panels if panel_solver.lattice else 0
            forces = panel_solver.results.get("forces", [])
            last_forces = forces[-1] if forces else {}

            diagnostics_history["panel_n_panels"].append(float(n_panels))
            for gid, fvec in last_forces.items():
                diagnostics_history.setdefault(f"panel_F_group{gid}_x", []).append(float(fvec[0]))
                diagnostics_history.setdefault(f"panel_F_group{gid}_y", []).append(float(fvec[1]))
                diagnostics_history.setdefault(f"panel_F_group{gid}_z", []).append(float(fvec[2]))

            freq = max(1, int(getattr(panel_solver, "logging_interval_steps", 1)))
            if step % freq == 0:
                PanelDiagnostics.export_forces_csv(panel_solver, last_forces, time, step, case_dir)
        except Exception as exc:
            print(f"(Warning) Failed to record panel diagnostics: {exc}")

    @staticmethod
    def export_forces_csv(
        panel_solver,
        forces: dict,
        time: float,
        step: int,
        case_dir: str,
    ) -> None:
        samples_dir = Path(case_dir).resolve() / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        csv_path = samples_dir / "panel_forces.csv"

        n_panels = panel_solver.lattice.num_panels if panel_solver.lattice else 0

        row: list[float | int] = [time, step, n_panels]
        headers = ["time", "step", "n_panels"]
        for gid in sorted(forces.keys()):
            fvec = forces[gid]
            for comp, label in enumerate(["x", "y", "z"]):
                headers.append(f"F_group{gid}_{label}")
                row.append(float(fvec[comp]))

        write_header = not csv_path.exists()
        with open(csv_path, "a") as f:
            if write_header:
                f.write(",".join(headers) + "\n")
            f.write(",".join(str(v) for v in row) + "\n")
