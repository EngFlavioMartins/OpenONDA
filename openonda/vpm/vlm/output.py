"""
VLM output: the force and loading tables (``vlm_forces.csv``, ``vlm_surface_forces.csv``,
span- and chord-wise loading) and the lattice surface for VTK.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from openonda.core.vtk_output import write_vtk_dataset

if TYPE_CHECKING:
    from vtkmodules.vtkCommonDataModel import vtkFieldData

    from ..particles.container import Particles
    from .solver import VLMSolver


class VLMDiagnostics:
    """Static helpers for recording VLM diagnostics and writing CSV output."""

    @staticmethod
    def record_vlm_diagnostics(
        vlm_solver: VLMSolver,
        particles: Particles,
        particle_vortex_strength: np.ndarray,
        time: float,
        samples_dir: Path,
    ) -> None:
        """Record VLM force and vector-strength scalars and flush one solver sample.

        Parameters
        ----------
        vlm_solver:
            Active ``VLMSolver`` instance.
        particles:
            Current ``Particles`` container (for particle count).
        particle_vortex_strength:
            Vortex-strength array (N, 3) for the current particles [m³/s].
        time:
            Current simulation time [s].
        samples_dir:
            Sample directory of the case; the table is ``vlm_forces.csv``.
        """
        if vlm_solver is None or vlm_solver._last_forces is None:
            return
        forces = vlm_solver._last_forces
        # Use the same integrated three-leg field as the solver budget. A
        # quarter-chord-only reconstruction is wrong for tapered/twisted wings.
        bound_vortex_strength = vlm_solver.compute_total_bound_vortex_strength()
        bound_vortex_strength_y = float(bound_vortex_strength[1])

        n_p = particles.n_particles_total
        wake_vortex_strength_y = float(particle_vortex_strength[:, 1].sum(dtype=np.float64))

        # Coupled VLM is an owned VPM component.  The caller reaches this
        # method only after an accepted VPM step, so every row belongs to the
        # solver's accepted clock; no VLM-specific cadence may skip it.
        VLMDiagnostics.export_forces_csv(
            vlm_solver,
            forces,
            bound_vortex_strength_y,
            wake_vortex_strength_y,
            n_p,
            time,
            samples_dir,
        )

    # CSV export

    @staticmethod
    def export_forces_csv(
        vlm_solver: VLMSolver,
        forces: dict,
        bound_vortex_strength: float,
        wake_vortex_strength: float,
        n_p: int,
        time: float,
        samples_dir: Path,
    ) -> None:
        """Append one row to ``<samples_dir>/vlm_forces.csv``.

        Parameters
        ----------
        vlm_solver:
            Active ``VLMSolver`` instance.
        forces:
            Force dict returned by ``vlm_solver.compute_forces()``.
        bound_vortex_strength, wake_vortex_strength:
            Pre-computed bound and wake y-components of vector strength [m³/s].
        n_p:
            Number of VPM particles at this step.
        time:
            Current simulation time [s].
        samples_dir:
            Sample directory of the case.
        """
        import pandas as pd

        samples_dir.mkdir(parents=True, exist_ok=True)
        csv_path = samples_dir / "vlm_forces.csv"

        force_density = float(vlm_solver._force_density)
        reference_velocity = vlm_solver._last_reference_velocity
        if reference_velocity is None:
            reference_velocity = vlm_solver.freestream_velocity
        reference_speed = (
            float(np.linalg.norm(reference_velocity)) if reference_velocity is not None else 0.0
        )

        row = {
            "time": time,
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
            "n_particles_total": n_p,
            "force_density": force_density,
            "force_units": "N",
            "moment_units": "N*m",
            "reference_speed": reference_speed,
            "reference_speed_units": "m/s",
        }
        reference_velocity = vlm_solver._last_reference_velocity
        if reference_velocity is None:
            reference_velocity = vlm_solver.freestream_velocity
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


class VLMLoadingDistribution:
    """Static helpers for extracting and exporting chord/span loading distributions."""

    # Record hook called by the solver

    @staticmethod
    def record_loading_distributions(
        vlm_solver: VLMSolver,
        time: float,
        samples_dir: Path,
    ) -> None:
        """Export mandatory solver-clock loading distributions for each surface.

        Coupled VLM output is dispatched by the owning VPM accepted-step
        run. Output failures propagate so a run cannot claim success
        while required scientific samples are missing.
        """
        reference_velocity = vlm_solver._last_reference_velocity
        density = vlm_solver.density

        for surface_name in vlm_solver.surfaces:
            distributions = VLMLoadingDistribution.extract_distributions(
                vlm_solver, surface_name, reference_velocity, density
            )
            VLMLoadingDistribution.export_distribution_csv(
                vlm_solver,
                surface_name,
                distributions,
                time,
                samples_dir,
            )

    # Grid index

    @staticmethod
    def build_surface_grid_index(vlm_solver: VLMSolver, surface_name: str) -> list[dict]:
        """Return one entry per (wing, segment) block belonging to surface_name.

        Each entry:
          wing_uid, segment_uid, segment_id, n_chordwise_panels, n_spanwise_panels, symmetry,
          original_panel_indices  : np.ndarray (n_spanwise_panels, n_chordwise_panels) of panel flat indices,
          mirrored_panel_indices: np.ndarray (n_spanwise_panels, n_chordwise_panels) | None
        """
        n_panels = vlm_solver.lattice.n_panels
        panel_segment_id = vlm_solver.lattice.segment_id.to_numpy()[:n_panels]
        surface_panels = vlm_solver._surface_panels[surface_name]

        surface_blocks: list[dict] = []
        global_segment_id = 0  # sequential segment ID across all wings of the aircraft
        wing_panel_start = 0
        for wing_uid, wing in vlm_solver.aircraft.wings.items():
            is_selected = surface_panels.start <= wing_panel_start < surface_panels.stop
            for local_segment_id, (segment_uid, segment) in enumerate(wing.segments.items()):
                n_chordwise_panels = segment.n_chordwise_panels
                n_spanwise_panels = segment.n_spanwise_panels
                if is_selected:
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
                    expected_segment_id = local_segment_id
                    observed = panel_segment_id[original_flat_indices]
                    if not np.all(observed == expected_segment_id):
                        vlm_solver.logger.warning(
                            f"[VLMLoadingDistribution] segment_id mismatch for "
                            f"wing '{wing_uid}' segment '{segment_uid}' (expected {expected_segment_id}, "
                            f"got unique {np.unique(observed)}). Using index arithmetic."
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
            wing_panel_start += sum(
                segment.n_chordwise_panels
                * segment.n_spanwise_panels
                * (2 if wing.symmetry > 0 else 1)
                for segment in wing.segments.values()
            )

        return surface_blocks

    # Extraction

    @staticmethod
    def extract_distributions(
        vlm_solver: VLMSolver,
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

        # per-panel data (no recompute)
        panel_force = vlm_solver.lattice.get_forces()  # (N,3)
        unsteady_force = vlm_solver.lattice.unsteady_panel_force.to_numpy()[
            : vlm_solver.lattice.n_panels
        ]
        unsteady_cp = vlm_solver.lattice.unsteady_pressure_jump_coefficient.to_numpy()[
            : vlm_solver.lattice.n_panels
        ]
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
        kinematic_velocity = vlm_solver.lattice.bound_kinematic_velocity.to_numpy()[
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
                            "section_force_units": "N",
                            "force_density": float(density),
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
                    circulation_pressure_jump_proxy = np.where(
                        pressure_coefficient_denominator > 1e-15,
                        2.0 * station_circulation / pressure_coefficient_denominator,
                        0.0,
                    )
                    circulation_pressure_jump_proxy += unsteady_cp[station_panel_indices]

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
                                "circulation_pressure_jump_proxy": float(
                                    circulation_pressure_jump_proxy[i]
                                ),
                                "panel_force_x": float(station_panel_force[i, 0]),
                                "panel_force_y": float(station_panel_force[i, 1]),
                                "panel_force_z": float(station_panel_force[i, 2]),
                                "panel_force_units": "N",
                                "force_density": float(density),
                                "unsteady_force_x": float(
                                    unsteady_force[station_panel_indices[i], 0]
                                ),
                                "unsteady_force_y": float(
                                    unsteady_force[station_panel_indices[i], 1]
                                ),
                                "unsteady_force_z": float(
                                    unsteady_force[station_panel_indices[i], 2]
                                ),
                                "relative_velocity_x": float(station_relative_velocity[i, 0]),
                                "relative_velocity_y": float(station_relative_velocity[i, 1]),
                                "relative_velocity_z": float(station_relative_velocity[i, 2]),
                                "relative_velocity_units": "m/s",
                                "bound_x": float(station_bound_vortex_midpoint[i, 0]),
                                "bound_y": float(station_bound_vortex_midpoint[i, 1]),
                                "bound_z": float(station_bound_vortex_midpoint[i, 2]),
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
        vlm_solver: VLMSolver,
        surface_name: str,
        distributions: dict[str, Any],
        time: float,
        samples_dir: Path,
    ) -> None:
        """Append one time-step's distributions to the per-surface CSVs."""
        import pandas as pd

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
            df.insert(1, "surface", surface_name)

            csv_path = samples_dir / fname_tpl
            if not csv_path.exists():
                df.to_csv(csv_path, index=False)
            else:
                df.to_csv(csv_path, mode="a", header=False, index=False)


# Optional fields are exported only when present in the recorded state.
# In particular, absent unsteady loads must never be presented as measured zeros.
CELL_FIELDS = (
    "area",
    "normal",
    "is_trailing_edge",
    "is_leading_edge",
    "wing_id",
    "segment_id",
    "is_mirrored",
    "circulation",
    "cumulative_circulation",
    "bound_vortex_velocity",
    "bound_external_velocity",
    "bound_relative_velocity",
    "velocity",
    "kinematic_velocity",
    "relative_velocity",
    "panel_force",
    "unsteady_panel_force",
    "panel_moment_correction",
    "unsteady_pressure_jump_coefficient",
)


def write_lattice_vtk(
    fields: Mapping[str, np.ndarray],
    filename: Path | str,
    *,
    reference_speed: float,
    time: float,
    force_density: float | None = None,
) -> Path:
    """Write saved or live arrays without advancing motion or recomputing loads.

    Panel corners retain their stored precision. ``velocity`` and
    ``bound_vortex_velocity`` are inertial fluid velocities; the corresponding
    ``relative_velocity`` fields subtract the prescribed body motion. Circulation
    is in [m^2/s], and ``panel_force`` is the force in newtons at the solve
    density when ``force_density`` is supplied. The dimensionless
    ``circulation_pressure_jump_proxy`` combines the circulation-based
    lifting-surface estimate and any recorded unsteady contribution. It is
    distinct from ``panel_normal_load_coefficient``, which normalizes the
    actual normal panel force. All fields are selectable in ParaView.
    """
    from vtk import vtkCellArray, vtkPoints, vtkPolyData
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    corners = np.asarray(fields["panel_corner_position"])
    if corners.ndim != 3 or corners.shape[1:] != (4, 3) or not np.isfinite(corners).all():
        raise ValueError("VLM surface corners must be finite with shape (panels, 4, 3)")
    if not np.isfinite(time) or not np.isfinite(reference_speed):
        raise ValueError("VLM surface time and reference speed must be finite")
    count = len(corners)
    points = vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(corners.reshape(-1, 3)), deep=True))
    cells = vtkCellArray()
    cells.SetData(
        numpy_to_vtkIdTypeArray(np.arange(count + 1, dtype=np.int64) * 4, deep=True),
        numpy_to_vtkIdTypeArray(np.arange(count * 4, dtype=np.int64), deep=True),
    )
    surface = vtkPolyData()
    surface.SetPoints(points)
    surface.SetPolys(cells)

    def add(data: vtkFieldData, name: str, values: np.ndarray) -> None:
        """Attach a named deep-copy array to VTK cell or field data."""
        array = numpy_to_vtk(np.ascontiguousarray(values), deep=True)
        array.SetName(name)
        data.AddArray(array)

    computed_area = None
    if "area" not in fields:
        # Geometry determines area independently of the stored solution.
        # Reuse this value when normalizing the recorded normal force.
        computed_area = 0.5 * np.linalg.norm(
            np.cross(
                corners[:, 2] - corners[:, 0],
                corners[:, 1] - corners[:, 3],
            ),
            axis=1,
        )
        if not np.isfinite(computed_area).all():
            raise ValueError("Invalid geometry-derived VLM panel area")

    cell_data = surface.GetCellData()
    for name in CELL_FIELDS:
        if name in fields:
            values = np.asarray(fields[name])
            if values.shape not in ((count,), (count, 3)) or not np.isfinite(values).all():
                raise ValueError(f"Invalid saved VLM field {name}")
            add(cell_data, name, values)
    add(cell_data, "panel_id", np.arange(count, dtype=np.int32))
    add(cell_data, "panel_centre", corners.mean(axis=1))
    chord = np.linalg.norm(
        0.5 * (corners[:, 3] + corners[:, 2] - corners[:, 0] - corners[:, 1]),
        axis=1,
    )
    add(cell_data, "panel_chord", chord)
    if computed_area is not None:
        add(cell_data, "area", computed_area)
    if "vortex_point_position" in fields:
        vortex = np.asarray(fields["vortex_point_position"])
        add(cell_data, "bound_vortex_leg", vortex[:, 2] - vortex[:, 1])
    if "circulation" in fields:
        denominator = reference_speed * chord
        pressure = np.divide(
            2.0 * fields["circulation"],
            denominator,
            out=np.zeros(count, dtype=corners.dtype),
            where=denominator > 1e-15,
        )
        if "unsteady_pressure_jump_coefficient" in fields:
            pressure += fields["unsteady_pressure_jump_coefficient"]
        add(cell_data, "circulation_pressure_jump_proxy", pressure)
        cell_data.SetActiveScalars("circulation")
    if force_density is not None:
        force_density = float(force_density)
        if not np.isfinite(force_density) or force_density <= 0.0:
            raise ValueError("VLM force density must be finite and positive")
        area = np.asarray(fields["area"]) if "area" in fields else computed_area
        if {"panel_force", "normal"} <= set(fields) and area is not None:
            denominator = 0.5 * force_density * reference_speed**2 * area
            normal_load = np.divide(
                np.sum(np.asarray(fields["panel_force"]) * np.asarray(fields["normal"]), axis=1),
                denominator,
                out=np.zeros(count, dtype=corners.dtype),
                where=denominator > 1e-15,
            )
            add(cell_data, "panel_normal_load_coefficient", normal_load)
    # VTK requires the exact metadata key TimeValue to recover physical time
    # from standalone VTP file series.  It is not a model field; the native
    # API and scientific arrays keep their snake_case names.
    for name in ("time", "TimeValue"):
        add(surface.GetFieldData(), name, np.array([time], dtype=np.float64))
    from vtk import vtkStringArray

    metadata = {
        "velocity_frame": "inertial fluid velocity at collocation points [m/s]",
        "relative_velocity_definition": "relative_velocity = velocity - kinematic_velocity [m/s]",
        "bound_velocity_frame": "inertial fluid velocity at bound-vortex midpoints [m/s]",
        "bound_relative_velocity_definition": (
            "bound_relative_velocity = bound_vortex_velocity - bound_kinematic_velocity [m/s]"
        ),
        "cell_association": "all VLM fields are cell-centred; panel_id is lattice order",
        "circulation_pressure_jump_proxy_definition": (
            "2*circulation/(reference_speed*panel_chord) + unsteady contribution when present"
        ),
    }
    if force_density is not None:
        add(surface.GetFieldData(), "force_density", np.array([force_density], dtype=np.float64))
        metadata["panel_force_definition"] = "panel_force is dimensional force [N] at force_density"
        metadata["panel_normal_load_coefficient_definition"] = (
            "dot(panel_force, normal)/(0.5*force_density*reference_speed^2*area)"
        )
    else:
        metadata["panel_force_definition"] = (
            "panel_force units are preserved from input; force density unavailable"
        )
    for name, value in metadata.items():
        array = vtkStringArray()
        array.SetName(name)
        array.InsertNextValue(value)
        surface.GetFieldData().AddArray(array)
    return write_vtk_dataset(surface, Path(filename))
