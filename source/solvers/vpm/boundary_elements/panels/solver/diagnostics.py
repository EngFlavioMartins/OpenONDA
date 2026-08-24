"""
Panel diagnostics module — recording and CSV export of panel force/circulation history.

Mirrors ``vlm/solver/diagnostics.py`` conventions.  Owns all logic for:
  - Appending per-step panel scalars to the solver diagnostics history dict.
  - Writing panel_force.csv with per-group force decomposition.
  - Appending time / observed_time_step_size history entries.

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
        if "time" not in diagnostics_history:
            return
        ft_hist = diagnostics_history["time"]
        if len(ft_hist) > 0 and ft_hist[-1] == time:
            return
        ft_hist.append(time)
        if len(diagnostics_history["observed_time_step_size"]) < len(ft_hist):
            diagnostics_history["observed_time_step_size"].append(float(observed_time_step_size))

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
            n_panels = panel_solver.lattice.n_panels if panel_solver.lattice else 0
            force_history = panel_solver.results.get("force_history", [])
            last_forces = force_history[-1] if force_history else {}

            diagnostics_history.setdefault("n_panels", []).append(float(n_panels))
            force_by_group = diagnostics_history.setdefault("panel_force_by_group", [])
            force_by_group.append(
                {
                    int(group_id): tuple(float(value) for value in force)
                    for group_id, force in last_forces.items()
                }
            )

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
        csv_path = samples_dir / "panel_force.csv"

        n_panels = panel_solver.lattice.n_panels if panel_solver.lattice else 0

        headers = [
            "time",
            "step",
            "n_panels",
            "group_id",
            "panel_force_x",
            "panel_force_y",
            "panel_force_z",
        ]
        force_rows = forces.items() or [(-1, (0.0, 0.0, 0.0))]

        write_header = not csv_path.exists()
        with open(csv_path, "a") as f:
            if write_header:
                f.write(",".join(headers) + "\n")
            for group_id, panel_force in sorted(force_rows):
                row: list[float | int] = [
                    time,
                    step,
                    n_panels,
                    int(group_id),
                    float(panel_force[0]),
                    float(panel_force[1]),
                    float(panel_force[2]),
                ]
                f.write(",".join(str(value) for value in row) + "\n")
