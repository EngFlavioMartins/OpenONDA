"""
VLM loading-distribution module — chord-wise and span-wise force extraction.

Extracts structured chord- and span-wise loading distributions from the per-panel
data already computed by the VLM solver.  No force recomputation is performed;
this is a pure reduction of existing lattice fields.

Output CSVs:
  <case_dir>/samples/vlm_spanwise_<surface>.csv   — one row per spanwise station
  <case_dir>/samples/vlm_chordwise_<surface>.csv  — one row per (station, chord cell)

Call pattern (mirrors VLMDiagnostics):
  VLMLoadingDistribution.record_loading_distributions(
      vlm_solver, diagnostics_history, time_step_size, time, case_dir)

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: June 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np

from ....io.sampling import resolve_samples_dir


class VLMLoadingDistribution:
    """Static helpers for extracting and exporting chord/span loading distributions."""

    # TOP-LEVEL RECORD HOOK (called from core/solver.py)

    @staticmethod
    def record_loading_distributions(
        vlm_solver,
        diagnostics_history: dict,
        step: int,
        time: float,
        case_dir: str,
        sample_subdirectory: str | None = None,
    ) -> None:
        """Iterate surfaces flagged sample_surface_forces and export distributions.

        Gated on vlm_solver.logging_interval_steps.  Wrapped in try/except so a
        failure never aborts the simulation.
        """
        if vlm_solver is None or not hasattr(vlm_solver, "_surface_sampling"):
            return
        if not vlm_solver._surface_sampling:
            return
        logging_interval_steps = max(1, int(getattr(vlm_solver, "logging_interval_steps", 1)))
        if step % logging_interval_steps != 0:
            return

        reference_velocity = getattr(vlm_solver, "_last_reference_velocity", None)
        density = getattr(vlm_solver, "density", 1.0)

        for surface_name, enabled in vlm_solver._surface_sampling.items():
            if not enabled:
                continue
            try:
                distributions = VLMLoadingDistribution.extract_distributions(
                    vlm_solver, surface_name, reference_velocity, density
                )
                VLMLoadingDistribution.export_distribution_csv(
                    vlm_solver,
                    surface_name,
                    distributions,
                    time,
                    step,
                    case_dir,
                    sample_subdirectory,
                )
            except Exception as exc:
                print(
                    f"(Warning) Failed to record loading distribution for '{surface_name}': {exc}"
                )

    # GRID INDEX

    @staticmethod
    def build_surface_grid_index(vlm_solver, surface_name: str) -> list[dict]:
        """Return one entry per (wing, segment) block belonging to surface_name.

        Each entry:
          wing_uid, segment_uid, segment_id, n_chordwise_panels, n_spanwise_panels, symmetry,
          original_panel_indices  : np.ndarray (n_spanwise_panels, n_chordwise_panels) of panel flat indices,
          mirrored_panel_indices: np.ndarray (n_spanwise_panels, n_chordwise_panels) | None
        """
        wing_ranges = vlm_solver._build_wing_panel_ranges()
        n_panels = vlm_solver.lattice.n_panels
        panel_segment_id = vlm_solver.lattice.segment_id.to_numpy()[:n_panels]

        # select wings for this surface (single- or multi-surface)
        # single-surface: aircraft.uid == surface_name, wing UIDs are bare
        # multi-surface:  aircraft.uid == "combined", wing UIDs are "{surface_name}_{wing_uid}"
        is_multi_surface = vlm_solver.aircraft.uid == "combined"
        selected_wings = {}
        for uid, wing in vlm_solver.aircraft.wings.items():
            if is_multi_surface:
                match = uid.startswith(surface_name + "_")
            else:
                match = vlm_solver.aircraft.uid == surface_name
            if match and uid in wing_ranges:
                selected_wings[uid] = wing

        surface_blocks: list[dict] = []
        global_segment_id = 0  # sequential segment ID across all wings of the aircraft
        for wing_uid, wing in vlm_solver.aircraft.wings.items():
            for segment_uid, segment in wing.segments.items():
                n_chordwise_panels = segment.n_chordwise_panels
                n_spanwise_panels = segment.n_spanwise_panels
                is_selected = wing_uid in selected_wings
                if is_selected:
                    wing_panel_start = wing_ranges[wing_uid][0]
                    # cursor inside this wing
                    cursor = wing_panel_start
                    for _prev_seg_uid, prev_seg in wing.segments.items():
                        if _prev_seg_uid == segment_uid:
                            break
                        n = prev_seg.n_chordwise_panels * prev_seg.n_spanwise_panels
                        cursor += n
                        if wing.symmetry > 0:
                            cursor += n

                    n = n_chordwise_panels * n_spanwise_panels
                    original_flat_indices = np.arange(cursor, cursor + n)
                    original_panel_indices = original_flat_indices.reshape(
                        n_spanwise_panels, n_chordwise_panels
                    )

                    mirrored_panel_indices = None
                    if wing.symmetry > 0:
                        mirrored_flat_indices = np.arange(cursor + n, cursor + 2 * n)
                        mirrored_panel_indices = mirrored_flat_indices.reshape(
                            n_spanwise_panels, n_chordwise_panels
                        )

                    # cross-check: segment IDs should match
                    expected_segment_id = global_segment_id
                    observed = panel_segment_id[original_flat_indices]
                    if not np.all(observed == expected_segment_id):
                        warnings.warn(
                            f"[VLMLoadingDistribution] segment_id mismatch for "
                            f"wing '{wing_uid}' segment '{segment_uid}' (expected {expected_segment_id}, "
                            f"got unique {np.unique(observed)}). Using index arithmetic.",
                            stacklevel=2,
                        )

                    surface_blocks.append(
                        {
                            "wing_uid": wing_uid,
                            "segment_uid": segment_uid,
                            "segment_id": global_segment_id,
                            "n_chordwise_panels": n_chordwise_panels,
                            "n_spanwise_panels": n_spanwise_panels,
                            "symmetry": wing.symmetry,
                            "original_panel_indices": original_panel_indices,
                            "mirrored_panel_indices": mirrored_panel_indices,
                        }
                    )
                global_segment_id += 1

        return surface_blocks

    # EXTRACT

    @staticmethod
    def extract_distributions(
        vlm_solver,
        surface_name: str,
        reference_velocity: np.ndarray | None,
        density: float,
    ) -> dict[str, Any]:
        """Extract spanwise and chordwise loading from existing per-panel arrays.

        Returns {'spanwise': pd.DataFrame, 'chordwise': pd.DataFrame}.
        """
        import pandas as pd

        if not vlm_solver._solved:
            return {"spanwise": pd.DataFrame(), "chordwise": pd.DataFrame()}

        # ---- per-panel data (no recompute) ----
        panel_force = vlm_solver.lattice.get_forces()  # (N,3)
        circulation = vlm_solver.lattice.get_circulation()  # (N,)
        panel_corner_position = vlm_solver.lattice.panel_corner_position.to_numpy()[
            : vlm_solver.lattice.n_panels
        ]  # (N,4,3)
        bound_vortex_midpoint = vlm_solver.lattice.bound_vortex_midpoint.to_numpy()[
            : vlm_solver.lattice.n_panels
        ]  # (N,3)
        bound_vortex_velocity = vlm_solver.lattice.bound_vortex_velocity.to_numpy()[
            : vlm_solver.lattice.n_panels
        ]  # (N,3)
        kinematic_velocity = vlm_solver.lattice.kinematic_velocity.to_numpy()[
            : vlm_solver.lattice.n_panels
        ]  # (N,3)
        vortex_point_position = vlm_solver.lattice.vortex_point_position.to_numpy()[
            : vlm_solver.lattice.n_panels
        ]  # (N,4,3)

        # V∞ — prefer explicit reference_velocity over the kinematic_velocity heuristic to
        # handle the static wind-frame case where kinematic_velocity is ~0.
        if reference_velocity is not None:
            freestream_speed = float(np.linalg.norm(reference_velocity))
        else:
            kinematic_speed = np.linalg.norm(kinematic_velocity, axis=1)
            freestream_speed = (
                float(np.median(kinematic_speed[kinematic_speed > 1e-8]))
                if kinematic_speed.max() > 1e-8
                else 1.0
            )
        freestream_speed = max(freestream_speed, 1e-10)
        dynamic_pressure = 0.5 * density * freestream_speed**2

        # Lift direction in wind axes (perpendicular to reference_velocity in the vertical plane)
        if reference_velocity is not None and np.linalg.norm(reference_velocity) > 1e-10:
            reference_direction = np.asarray(reference_velocity) / np.linalg.norm(
                reference_velocity
            )
        else:
            reference_direction = np.array([1.0, 0.0, 0.0])
        vertical_direction = np.array([0.0, 0.0, 1.0])
        lift_direction = (
            vertical_direction
            - np.dot(vertical_direction, reference_direction) * reference_direction
        )
        lift_direction_magnitude = np.linalg.norm(lift_direction)
        lift_direction = (
            lift_direction / lift_direction_magnitude
            if lift_direction_magnitude > 1e-10
            else vertical_direction
        )
        drag_direction = reference_direction.copy()

        # ΔCp denominator: panel chord from panel_corner_position (TE_mid − LE_mid)
        trailing_edge_midpoint = 0.5 * (
            panel_corner_position[:, 3] + panel_corner_position[:, 2]
        )  # S+R / 2
        leading_edge_midpoint = 0.5 * (
            panel_corner_position[:, 0] + panel_corner_position[:, 1]
        )  # P+Q / 2
        panel_chord = np.linalg.norm(trailing_edge_midpoint - leading_edge_midpoint, axis=1)  # (N,)

        surface_blocks = VLMLoadingDistribution.build_surface_grid_index(vlm_solver, surface_name)

        full_span: list[dict] = []
        full_chord: list[dict] = []

        for surface_block in surface_blocks:
            n_chordwise_panels = surface_block["n_chordwise_panels"]
            n_spanwise_panels = surface_block["n_spanwise_panels"]
            wing_uid = surface_block["wing_uid"]
            segment_uid = surface_block["segment_uid"]

            for half, panel_index_grid in [
                ("orig", surface_block["original_panel_indices"]),
                ("mirror", surface_block["mirrored_panel_indices"]),
            ]:
                if panel_index_grid is None:
                    continue

                # Span axis ŝ for this block — do NOT assume the global y-axis.
                # A vertical fin or a rotated rotor blade has bound legs with
                # ~zero y-component, which would drive spanwise_station_width → 0 and blow up
                # L_prime/section_lift_coefficient.  The span direction is what the bound legs
                # (V3−V2) actually point along; for a y-aligned wing this
                # reduces exactly to the previous |Δy| behaviour.
                block_bound_vortex_leg = (
                    vortex_point_position[panel_index_grid.ravel(), 2]
                    - vortex_point_position[panel_index_grid.ravel(), 1]
                )  # (n_spanwise_panels·n_chordwise_panels, 3)
                span_direction = block_bound_vortex_leg.mean(axis=0)
                span_direction_magnitude = np.linalg.norm(span_direction)
                span_direction = (
                    span_direction / span_direction_magnitude
                    if span_direction_magnitude > 1e-12
                    else np.array([0.0, 1.0, 0.0])
                )
                # Deterministic orientation: largest-magnitude component positive,
                # so 'orig' and 'mirror' halves share one axis and stations sort
                # consistently run-to-run.
                dominant_span_axis = int(np.argmax(np.abs(span_direction)))
                if span_direction[dominant_span_axis] < 0.0:
                    span_direction = -span_direction

                for j in range(n_spanwise_panels):
                    station_panel_indices = panel_index_grid[
                        j
                    ]  # chord-panel indices for station j, shape (n_chordwise_panels,)
                    station_panel_force = panel_force[
                        station_panel_indices
                    ]  # (n_chordwise_panels, 3)
                    station_circulation = circulation[
                        station_panel_indices
                    ]  # (n_chordwise_panels,)
                    station_panel_chord = panel_chord[
                        station_panel_indices
                    ]  # (n_chordwise_panels,)
                    station_bound_vortex_midpoint = bound_vortex_midpoint[
                        station_panel_indices
                    ]  # (n_chordwise_panels, 3)
                    station_relative_velocity = (
                        bound_vortex_velocity[station_panel_indices]
                        - kinematic_velocity[station_panel_indices]
                    )  # (n_chordwise_panels, 3)

                    # spanwise station coordinate: bound midpoints projected on ŝ
                    span_coordinate = float(np.mean(station_bound_vortex_midpoint @ span_direction))
                    absolute_span_coordinate = abs(span_coordinate)

                    # Physical strip edges projected on the same span axis.  The
                    # loading lives at panel centres, but the surface span is set
                    # by its corner coordinates.  Inferring the span from the
                    # first/last panel centres maps those centres to ±1 and makes
                    # a finite, cell-centred circulation look non-zero at the
                    # physical wing tip in exported plots.
                    station_corner_coordinates = (
                        panel_corner_position[station_panel_indices].reshape(-1, 3) @ span_direction
                    )
                    min_span_edge = float(station_corner_coordinates.min())
                    max_span_edge = float(station_corner_coordinates.max())

                    # station width: bound-leg length (V3−V2) projected on ŝ
                    bound_vortex_leg = (
                        vortex_point_position[station_panel_indices, 2]
                        - vortex_point_position[station_panel_indices, 1]
                    )  # (n_chordwise_panels, 3)
                    spanwise_station_width = float(
                        np.mean(np.abs(bound_vortex_leg @ span_direction))
                    )
                    spanwise_station_width = max(spanwise_station_width, 1e-10)

                    local_chord = float(station_panel_chord.sum())
                    local_relative_speed = float(
                        np.mean(np.linalg.norm(station_relative_velocity, axis=1))
                    )

                    # sectional forces
                    section_force = station_panel_force.sum(axis=0)  # (3,)
                    section_lift = float(np.dot(section_force, lift_direction))
                    section_drag = float(np.dot(section_force, drag_direction))
                    section_circulation = float(station_circulation.sum())
                    circulation_magnitude = abs(section_circulation)

                    lift_per_span = section_lift / spanwise_station_width
                    drag_per_span = section_drag / spanwise_station_width
                    section_lift_coefficient = (
                        lift_per_span / (dynamic_pressure * local_chord)
                        if dynamic_pressure * local_chord > 1e-15
                        else 0.0
                    )
                    section_lift_coefficient_from_circulation = (
                        2.0 * circulation_magnitude / (local_chord * local_relative_speed)
                        if local_chord * local_relative_speed > 1e-15
                        else 0.0
                    )

                    full_span.append(
                        {
                            "wing_uid": wing_uid,
                            "segment_uid": segment_uid,
                            "station_id": f"{wing_uid}:{segment_uid}:{half}:{j}",
                            "half": half,
                            "span_index": j,
                            "span_coordinate": span_coordinate,
                            "span_coordinate_absolute": absolute_span_coordinate,
                            "min_span_edge": min_span_edge,
                            "max_span_edge": max_span_edge,
                            "local_chord": local_chord,
                            "spanwise_station_width": spanwise_station_width,
                            "circulation": section_circulation,
                            "circulation_magnitude": circulation_magnitude,
                            "relative_speed": local_relative_speed,
                            "section_lift_coefficient_from_circulation": section_lift_coefficient_from_circulation,
                            "lift_per_span": lift_per_span,
                            "drag_per_span": drag_per_span,
                            "section_lift_coefficient": section_lift_coefficient,
                            "section_force_x": float(section_force[0]),
                            "section_force_y": float(section_force[1]),
                            "section_force_z": float(section_force[2]),
                        }
                    )

                    # chordwise
                    # x/c at panel midpoints: cumulative chord from LE
                    station_leading_edge_midpoint = leading_edge_midpoint[
                        station_panel_indices
                    ]  # (n_chordwise_panels, 3)
                    station_trailing_edge_midpoint = trailing_edge_midpoint[
                        station_panel_indices
                    ]  # (n_chordwise_panels, 3)
                    panel_chord_centre = 0.5 * (
                        station_leading_edge_midpoint + station_trailing_edge_midpoint
                    )
                    # chord_fraction: fractional distance along chord
                    if local_chord > 1e-10:
                        chord_fraction = (
                            np.linalg.norm(
                                panel_chord_centre - station_leading_edge_midpoint[0], axis=1
                            )
                            / local_chord
                        )
                    else:
                        chord_fraction = np.zeros(n_chordwise_panels)

                    pressure_coefficient_denominator = freestream_speed * station_panel_chord
                    pressure_jump_coefficient = np.where(
                        pressure_coefficient_denominator > 1e-15,
                        2.0 * station_circulation / pressure_coefficient_denominator,
                        0.0,
                    )

                    for i in range(n_chordwise_panels):
                        full_chord.append(
                            {
                                "wing_uid": wing_uid,
                                "segment_uid": segment_uid,
                                "station_id": f"{wing_uid}:{segment_uid}:{half}:{j}",
                                "half": half,
                                "span_index": j,
                                "span_coordinate": span_coordinate,
                                "span_coordinate_absolute": absolute_span_coordinate,
                                "min_span_edge": min_span_edge,
                                "max_span_edge": max_span_edge,
                                "chord_index": i,
                                "chord_fraction": float(chord_fraction[i]),
                                "panel_chord": float(station_panel_chord[i]),
                                "panel_circulation": float(station_circulation[i]),
                                "pressure_jump_coefficient": float(pressure_jump_coefficient[i]),
                                "panel_force_x": float(station_panel_force[i, 0]),
                                "panel_force_y": float(station_panel_force[i, 1]),
                                "panel_force_z": float(station_panel_force[i, 2]),
                            }
                        )

        # Assemble and sort by stable physical station.  The signed span
        # coordinate is retained for physical interpretation; the absolute
        # coordinate keeps the per-time CSV block root-to-tip ordered.
        df_span = (
            pd.DataFrame(full_span)
            .sort_values(["half", "span_coordinate_absolute"])
            .reset_index(drop=True)
        )
        df_chord = (
            pd.DataFrame(full_chord)
            .sort_values(["half", "span_coordinate_absolute", "chord_index"])
            .reset_index(drop=True)
        )

        # Full physical span for the normalized span coordinate.  Use strip
        # edges, not the outermost
        # cell centres: panel-centred output should remain strictly inside
        # [-1, 1], leaving the actual tip locations available for the Γ=0
        # closure used by visualisation and higher-order reconstruction.
        if not df_span.empty:
            span_min = float(df_span["min_span_edge"].min())
            span_max = float(df_span["max_span_edge"].max())
            b = span_max - span_min
            b = max(b, 1e-10)
            y_mid = 0.5 * (span_max + span_min)
            df_span["span_coordinate_normalized"] = 2.0 * (df_span["span_coordinate"] - y_mid) / b
            df_chord["span_coordinate_normalized"] = 2.0 * (df_chord["span_coordinate"] - y_mid) / b
        else:
            df_span["span_coordinate_normalized"] = np.nan
            df_chord["span_coordinate_normalized"] = np.nan

        return {"spanwise": df_span, "chordwise": df_chord}

    # EXPORT

    @staticmethod
    def export_distribution_csv(
        vlm_solver,
        surface_name: str,
        distributions: dict[str, Any],
        time: float,
        step: int,
        case_dir: str,
        sample_subdirectory: str | None = None,
    ) -> None:
        """Append one time-step's distributions to the per-surface CSVs."""
        import pandas as pd

        samples_dir = resolve_samples_dir(case_dir, sample_subdirectory)
        samples_dir.mkdir(parents=True, exist_ok=True)

        safe_name = surface_name.replace("/", "_").replace(" ", "_")

        for key, fname_tpl in [
            ("spanwise", f"vlm_spanwise_{safe_name}.csv"),
            ("chordwise", f"vlm_chordwise_{safe_name}.csv"),
        ]:
            df: pd.DataFrame = distributions.get(key, pd.DataFrame())
            if df.empty:
                continue

            # prepend metadata columns
            df = df.copy()
            df.insert(0, "time", time)
            df.insert(1, "step", step)
            df.insert(2, "surface", surface_name)

            csv_path = samples_dir / fname_tpl
            if not csv_path.exists():
                df.to_csv(csv_path, index=False)
            else:
                df.to_csv(csv_path, mode="a", header=False, index=False)
