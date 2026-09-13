"""
VLM diagnostics module — recording and CSV export of force/vector-strength history.

Owns all logic for:
  - Appending per-step VLM scalars to the solver diagnostics history dict.
  - Writing vlm_forces.csv.
  - Appending time / observed_time_step_size history entries.

Nothing in this module should import from the top-level VPM Solver class; all
required data is passed in explicitly so the solver itself stays a thin
orchestrator.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: March 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from __future__ import annotations

import numpy as np

from ....io.logging import Logging
from ....io.sampling import resolve_samples_dir


class VLMDiagnostics:
    """Static helpers for recording VLM diagnostics and writing CSV output."""

    @staticmethod
    def record_time(diagnostics_history: dict, time: float, observed_time_step_size: float) -> None:
        """Append *time* and *observed_time_step_size* to diagnostics history.

        Parameters
        ----------
        diagnostics_history:
            The solver's ``_diagnostics_history`` dict (mutated in-place).
        time:
            Current simulation time [s].
        observed_time_step_size:
            Wall-clock or physical output interval observed for this step [s].
        """
        if "time" not in diagnostics_history:
            return
        ft_hist = diagnostics_history["time"]
        if len(ft_hist) > 0 and ft_hist[-1] == time:
            return
        ft_hist.append(time)
        if len(diagnostics_history["observed_time_step_size"]) < len(ft_hist):
            diagnostics_history["observed_time_step_size"].append(float(observed_time_step_size))

    @staticmethod
    def record_vlm_diagnostics(
        vlm_solver,
        particles,
        particle_vortex_strength: np.ndarray,
        diagnostics_history: dict,
        step: int,
        time: float,
        case_dir: str,
        sample_directory: str | None = None,
    ) -> None:
        """Record VLM force and vector-strength scalars and flush one owner sample.

        Parameters
        ----------
        vlm_solver:
            Active ``VLMSolver`` instance.
        particles:
            Current ``Particles`` container (for particle count).
        particle_vortex_strength:
            Vortex-strength array (N, 3) for the current particles [m³/s].
        diagnostics_history:
            Solver's ``_diagnostics_history`` dict (mutated in-place).
        step:
            Current integer step counter.
        time:
            Current simulation time [s].
        case_dir:
            Solver backup/output root directory (CSV is written under
            ``<case_dir>/samples/vlm_forces.csv``).
        """
        if vlm_solver is None or not hasattr(vlm_solver, "_last_forces"):
            return
        forces = vlm_solver._last_forces
        n_panels = vlm_solver.lattice.n_panels
        # Use the same integrated three-leg field as the solver budget. A
        # quarter-chord-only reconstruction is wrong for tapered/twisted wings.
        bound_vortex_strength = vlm_solver.compute_total_bound_vortex_strength()
        bound_vortex_strength_y = float(bound_vortex_strength[1])

        n_p = particles.n_particles_total
        wake_vortex_strength_y = float(particle_vortex_strength[:, 1].sum(dtype=np.float64))

        lespnp = vlm_solver.lattice.leading_edge_suction_parameter.to_numpy()[:n_panels]
        max_leading_edge_suction_parameter = float(np.max(lespnp)) if n_panels > 0 else 0.0

        diagnostics_history["vlm_lift_coefficient"].append(float(forces["lift_coefficient"]))
        diagnostics_history["vlm_drag_coefficient"].append(float(forces["drag_coefficient"]))
        diagnostics_history["vlm_bound_vortex_strength_y"].append(bound_vortex_strength_y)
        diagnostics_history["vlm_wake_vortex_strength_y"].append(wake_vortex_strength_y)
        diagnostics_history["vlm_max_leading_edge_suction_parameter"].append(
            max_leading_edge_suction_parameter
        )
        diagnostics_history["vlm_n_particles_total"].append(float(n_p))

        # Coupled VLM is an owned VPM component.  The caller reaches this
        # method only after an accepted VPM step, so every row belongs to the
        # owner's accepted clock; no VLM-specific cadence may skip it.
        VLMDiagnostics.export_forces_csv(
            vlm_solver,
            forces,
            bound_vortex_strength_y,
            wake_vortex_strength_y,
            max_leading_edge_suction_parameter,
            n_p,
            time,
            step,
            case_dir,
            sample_directory,
        )

    # CSV export

    @staticmethod
    def export_forces_csv(
        vlm_solver,
        forces: dict,
        bound_vortex_strength: float,
        wake_vortex_strength: float,
        max_leading_edge_suction_parameter: float,
        n_p: int,
        time: float,
        step: int,
        case_dir: str,
        sample_directory: str | None = None,
    ) -> None:
        """Append one row to ``<case_dir>/samples/vlm_forces.csv``.

        Parameters
        ----------
        vlm_solver:
            Active ``VLMSolver`` instance.
        forces:
            Force dict returned by ``vlm_solver.compute_forces()``.
        bound_vortex_strength, wake_vortex_strength:
            Pre-computed bound and wake y-components of vector strength [m³/s].
        max_leading_edge_suction_parameter:
            Maximum Leading Edge Suction Parameter for this step.
        n_p:
            Number of VPM particles at this step.
        time:
            Current simulation time [s].
        step:
            Integer step counter.
        case_dir:
            Output root; CSV is written under ``<case_dir>/samples/``.
        """
        import pandas as pd

        samples_dir = resolve_samples_dir(case_dir, sample_directory)
        samples_dir.mkdir(parents=True, exist_ok=True)
        csv_path = samples_dir / "vlm_forces.csv"

        force_density = float(getattr(vlm_solver, "_force_density", vlm_solver.density))
        reference_velocity = getattr(vlm_solver, "_last_reference_velocity", None)
        if reference_velocity is None:
            reference_velocity = getattr(vlm_solver, "freestream_velocity", None)
        reference_speed = (
            float(np.linalg.norm(reference_velocity)) if reference_velocity is not None else 0.0
        )

        row = {
            "time": time,
            "step": step,
            "lift_coefficient": forces.get("lift_coefficient", 0.0),
            "drag_coefficient": forces.get("drag_coefficient", 0.0),
            "side_force_coefficient": forces.get("side_force_coefficient", 0.0),
            "force_x": forces.get("force_x", 0.0),
            "force_y": forces.get("force_y", 0.0),
            "force_z": forces.get("force_z", 0.0),
            "unsteady_force_x": forces.get("unsteady_force_x", 0.0),
            "unsteady_force_y": forces.get("unsteady_force_y", 0.0),
            "unsteady_force_z": forces.get("unsteady_force_z", 0.0),
            "moment_x": forces.get("moment_x", 0.0),
            "moment_y": forces.get("moment_y", 0.0),
            "moment_z": forces.get("moment_z", 0.0),
            "lift": forces.get("lift", 0.0),
            "drag": forces.get("drag", 0.0),
            "dynamic_pressure": forces.get("dynamic_pressure", 0.0),
            "reference_area": forces.get("reference_area", 0.0),
            "rolling_moment_coefficient": forces.get("rolling_moment_coefficient", 0.0),
            "pitching_moment_coefficient": forces.get("pitching_moment_coefficient", 0.0),
            "yawing_moment_coefficient": forces.get("yawing_moment_coefficient", 0.0),
            "rolling_moment_coefficient_quarter_chord": forces.get(
                "rolling_moment_coefficient_quarter_chord", 0.0
            ),
            "pitching_moment_coefficient_quarter_chord": forces.get(
                "pitching_moment_coefficient_quarter_chord", 0.0
            ),
            "yawing_moment_coefficient_quarter_chord": forces.get(
                "yawing_moment_coefficient_quarter_chord", 0.0
            ),
            "bound_vortex_strength_y": bound_vortex_strength,
            "wake_vortex_strength_y": wake_vortex_strength,
            "max_leading_edge_suction_parameter": max_leading_edge_suction_parameter,
            "n_particles_total": n_p,
            "force_density": force_density,
            "force_units": "N",
            "moment_units": "N*m",
            "reference_speed": reference_speed,
            "reference_speed_units": "m/s",
        }
        reference_velocity = getattr(vlm_solver, "_last_reference_velocity", None)
        if reference_velocity is None:
            reference_velocity = getattr(vlm_solver, "freestream_velocity", None)
        surface_forces = vlm_solver.compute_per_surface_forces(
            vlm_solver.density, reference_velocity
        )
        for key in ("power", "rotational_power", "translational_power"):
            row[key] = sum(surface[key] for surface in surface_forces.values())
        surfaces_path = csv_path.with_name("vlm_surface_forces.csv")
        pd.DataFrame(
            [
                {
                    "time": time,
                    "step": step,
                    "surface": name,
                    "force_density": force_density,
                    "force_units": "N",
                    "moment_units": "N*m",
                    **values,
                }
                for name, values in surface_forces.items()
            ]
        ).to_csv(surfaces_path, mode="a", header=not surfaces_path.exists(), index=False)
        df = pd.DataFrame([row])
        if not csv_path.exists():
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, mode="a", header=False, index=False)

    # ------------------------------------------------------------------
    # Surface-probe boundary leakage diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def record_vlm_leakage_diagnostics(
        vlm_solver,
        particles,
        physics,
        diagnostics_history: dict,
        step: int,
        time: float,
        case_dir: str,
        sample_directory: str | None = None,
    ) -> None:
        """Compute and record surface-probe boundary leakage (PR-1).

        Calls ``vlm_solver.compute_surface_leakage`` (observer-only, no
        solver mutation), appends scalar histories, writes
        ``vlm_leakage.csv`` and a VTK ``vlm_surface_leakage.vtp`` of
        probe residuals.

        Parameters
        ----------
        vlm_solver : VLMSolver
            Solved VLMSolver instance.
        particles : Particles
            Current VPM particle container.
        physics : Induction provider
            Must implement ``compute_target_velocity``.
        diagnostics_history : dict
            Solver's ``_diagnostics_history`` (mutated in-place).
        step, time : int, float
            Current step and simulation time.
        case_dir : str
            Root output directory.
        sample_directory : str or None
            Override for the samples subdirectory.
        """
        if vlm_solver is None or not hasattr(vlm_solver, "_last_forces"):
            return

        try:
            result = vlm_solver.compute_surface_leakage(particles, physics)
        except Exception as error:
            # Do not turn a missing diagnostic into apparent zero leakage.
            # Preserve the numerical run, but make the lost evidence visible.
            Logging.warning(f"VLM leakage diagnostic failed at step {step}: {error}")
            return

        if result.get("n_probes", 0) == 0:
            return

        # Append scalar history
        for key, hist_key in [
            ("R1", "vlm_leakage_R1"),
            ("Rinf", "vlm_leakage_Rinf"),
            ("edge_R1", "vlm_leakage_R1_edge"),
            ("edge_Rinf", "vlm_leakage_Rinf_edge"),
            ("interior_R1", "vlm_leakage_R1_interior"),
            ("interior_Rinf", "vlm_leakage_Rinf_interior"),
            ("reference_speed", "vlm_leakage_reference_speed"),
            ("transport_R1", "vlm_leakage_transport_R1"),
            ("transport_Rinf", "vlm_leakage_transport_Rinf"),
            ("collocation_R1", "vlm_leakage_collocation_R1"),
            ("collocation_Rinf", "vlm_leakage_collocation_Rinf"),
            ("off_grid_R1", "vlm_leakage_off_grid_R1"),
            ("off_grid_Rinf", "vlm_leakage_off_grid_Rinf"),
            ("independent_surface_R1", "vlm_leakage_independent_surface_R1"),
            ("independent_surface_Rinf", "vlm_leakage_independent_surface_Rinf"),
            ("two_sided_trace_R1", "vlm_leakage_two_sided_trace_R1"),
            ("two_sided_trace_Rinf", "vlm_leakage_two_sided_trace_Rinf"),
            ("transport_filter_radius", "vlm_leakage_transport_filter_radius"),
            ("boundary_filter_radius", "vlm_leakage_boundary_filter_radius"),
        ]:
            diagnostics_history.setdefault(hist_key, []).append(result.get(key, 0.0))

        events = getattr(vlm_solver, "_last_surface_events", ())
        event_codes = [event.get("event") for event in events]
        diagnostics_history.setdefault("vlm_surface_intersections", []).append(
            float(event_codes.count(1))
        )
        diagnostics_history.setdefault("vlm_surface_side_bypasses", []).append(
            float(event_codes.count(2))
        )
        diagnostics_history.setdefault("vlm_surface_core_overlaps", []).append(
            float(event_codes.count(3))
        )
        diagnostics_history.setdefault("vlm_stage_boundary_residual", []).append(
            float(getattr(vlm_solver, "_last_stage_boundary_residual", 0.0))
        )
        diagnostics_history.setdefault("vlm_stage_near_wake_elapsed", []).append(
            float(getattr(vlm_solver, "_last_stage_near_wake_elapsed", 0.0))
        )
        diagnostics_history.setdefault("vlm_stage_near_wake_matrix_norm", []).append(
            float(getattr(vlm_solver, "_last_stage_near_wake_matrix_norm", 0.0))
        )

        # Carry stage-response telemetry into the immutable observer result
        # before the static CSV/VTK writers receive it.
        result["stage_boundary_residual"] = float(
            getattr(vlm_solver, "_last_stage_boundary_residual", 0.0)
        )
        result["stage_near_wake_elapsed"] = float(
            getattr(vlm_solver, "_last_stage_near_wake_elapsed", 0.0)
        )
        result["stage_near_wake_matrix_norm"] = float(
            getattr(vlm_solver, "_last_stage_near_wake_matrix_norm", 0.0)
        )

        VLMDiagnostics._export_leakage_csv(result, step, time, case_dir, sample_directory)
        VLMDiagnostics._export_leakage_vtk(result, case_dir, sample_directory)

    @staticmethod
    def _export_leakage_csv(
        result: dict,
        step: int,
        time: float,
        case_dir: str,
        sample_directory: str | None = None,
    ) -> None:
        """Append one row to ``vlm_leakage.csv``."""
        import pandas as pd

        samples_dir = resolve_samples_dir(case_dir, sample_directory)
        samples_dir.mkdir(parents=True, exist_ok=True)
        csv_path = samples_dir / "vlm_leakage.csv"

        row = {
            "time": time,
            "step": step,
            "reference_speed": result["reference_speed"],
            "R1": result["R1"],
            "Rinf": result["Rinf"],
            "edge_R1": result["edge_R1"],
            "edge_Rinf": result["edge_Rinf"],
            "interior_R1": result["interior_R1"],
            "interior_Rinf": result["interior_Rinf"],
            "collocation_R1": result.get("collocation_R1", 0.0),
            "collocation_Rinf": result.get("collocation_Rinf", 0.0),
            "off_grid_R1": result.get("off_grid_R1", 0.0),
            "off_grid_Rinf": result.get("off_grid_Rinf", 0.0),
            "independent_surface_R1": result.get("independent_surface_R1", 0.0),
            "independent_surface_Rinf": result.get("independent_surface_Rinf", 0.0),
            "two_sided_trace_R1": result.get("two_sided_trace_R1", 0.0),
            "two_sided_trace_Rinf": result.get("two_sided_trace_Rinf", 0.0),
            "transport_R1": result.get("transport_R1", 0.0),
            "transport_Rinf": result.get("transport_Rinf", 0.0),
            "transport_filter_radius": result.get("transport_filter_radius", 0.0),
            "transport_operator": result.get("transport_operator", ""),
            "transport_filter_radii": ",".join(
                f"{float(value):.16g}" for value in result.get("transport_filter_radii", ())
            ),
            "boundary_filter_radius": result.get("boundary_filter_radius", 0.0),
            "stage_boundary_residual": result.get("stage_boundary_residual", 0.0),
            "stage_near_wake_elapsed": result.get("stage_near_wake_elapsed", 0.0),
            "stage_near_wake_matrix_norm": result.get("stage_near_wake_matrix_norm", 0.0),
            "n_probes": result["n_probes"],
            "n_panels": len(result["panel_area"]),
            "n_edge_panels": int(result["edge_mask"].sum()),
        }
        for sname, svals in result.get("per_surface", {}).items():
            safe = sname.replace(" ", "_")
            row[f"{safe}_R1"] = svals.get("R1", 0.0)
            row[f"{safe}_Rinf"] = svals.get("Rinf", 0.0)
            row[f"{safe}_n_probes"] = svals.get("n_probes", 0)

        df = pd.DataFrame([row])
        if not csv_path.exists():
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, mode="a", header=False, index=False)

    @staticmethod
    def _export_leakage_vtk(
        result: dict,
        case_dir: str,
        sample_directory: str | None = None,
    ) -> None:
        """Write ``vlm_surface_leakage.vtp`` with probe residuals."""
        import pyvista as pv

        samples_dir = resolve_samples_dir(case_dir, sample_directory)
        samples_dir.mkdir(parents=True, exist_ok=True)
        vtp_path = samples_dir / "vlm_surface_leakage.vtp"

        n = result["n_probes"]
        if n == 0:
            return

        points = result["probe_position"].astype(np.float64)
        cloud = pv.PolyData(points)

        cloud["r_k [m/s]"] = result["r_k"].astype(np.float64)
        if "transport_r_k" in result:
            cloud["transport_r_k [m/s]"] = result["transport_r_k"].astype(np.float64)
        cloud["probe_type"] = result["probe_type"].astype(np.int32)
        cloud["panel_index"] = result["probe_panel"].astype(np.int32)
        if "probe_weight" in result:
            cloud["probe_weight"] = result["probe_weight"].astype(np.float64)
        if "probe_scale" in result:
            cloud["probe_scale [m]"] = result["probe_scale"].astype(np.float64)
        cloud["is_edge_panel"] = result["edge_mask"][result["probe_panel"]].astype(np.int8)

        cloud.save(str(vtp_path))
