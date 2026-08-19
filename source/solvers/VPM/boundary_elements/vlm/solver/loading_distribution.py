"""
VLM loading-distribution module — chord-wise and span-wise force extraction.

Extracts structured chord- and span-wise loading distributions from the per-panel
data already computed by the VLM solver.  No force recomputation is performed;
this is a pure reduction of existing lattice fields.

Output CSVs:
  <backup_dir>/samples/vlm_spanwise_<surface>.csv   — one row per spanwise station
  <backup_dir>/samples/vlm_chordwise_<surface>.csv  — one row per (station, chord cell)

Call pattern (mirrors VLMDiagnostics):
  VLMLoadingDistribution.record_loading_distributions(
      vlm_solver, diagnostics_history, time_step, flow_time, backup_directory)

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
        backup_directory: str,
        sample_subdirectory: str | None = None,
    ) -> None:
        """Iterate surfaces flagged sample_surface_forces and export distributions.

        Gated on vlm_solver.logging_frequency.  Wrapped in try/except so a
        failure never aborts the simulation.
        """
        if vlm_solver is None or not hasattr(vlm_solver, "_surface_sampling"):
            return
        if not vlm_solver._surface_sampling:
            return
        freq = max(1, int(getattr(vlm_solver, "logging_frequency", 1)))
        if step % freq != 0:
            return

        U_ref = getattr(vlm_solver, "_last_U_ref", None)
        density = getattr(vlm_solver, "density", 1.0)

        for surface_name, enabled in vlm_solver._surface_sampling.items():
            if not enabled:
                continue
            try:
                dists = VLMLoadingDistribution.extract_distributions(
                    vlm_solver, surface_name, U_ref, density
                )
                VLMLoadingDistribution.export_distribution_csv(
                    vlm_solver,
                    surface_name,
                    dists,
                    time,
                    step,
                    backup_directory,
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
          wing_uid, segment_uid, segment_id, nc, ns, symmetry,
          orig_idx  : np.ndarray (ns, nc) of panel flat indices,
          mirror_idx: np.ndarray (ns, nc) | None
        """
        wing_ranges = vlm_solver._build_wing_panel_ranges()
        n_panels = vlm_solver.lattice.num_panels
        panel_seg_id = vlm_solver.lattice.panel_segment_id.to_numpy()[:n_panels]

        # select wings for this surface (single- or multi-surface)
        # single-surface: aircraft.uid == surface_name, wing UIDs are bare
        # multi-surface:  aircraft.uid == "combined", wing UIDs are "{surface_name}_{wing_uid}"
        is_multi = vlm_solver.aircraft.uid == "combined"
        selected_wings = {}
        for uid, wing in vlm_solver.aircraft.wings.items():
            if is_multi:
                match = uid.startswith(surface_name + "_")
            else:
                match = vlm_solver.aircraft.uid == surface_name
            if match and uid in wing_ranges:
                selected_wings[uid] = wing

        blocks: list[dict] = []
        seg_global_id = 0  # sequential segment ID across all wings of the aircraft
        for wing_uid, wing in vlm_solver.aircraft.wings.items():
            for seg_uid, seg in wing.segments.items():
                nc = seg.panels_chord
                ns = seg.panels_span
                is_selected = wing_uid in selected_wings
                if is_selected:
                    w_start = wing_ranges[wing_uid][0]
                    # cursor inside this wing
                    cursor = w_start
                    for _prev_seg_uid, prev_seg in wing.segments.items():
                        if _prev_seg_uid == seg_uid:
                            break
                        n = prev_seg.panels_chord * prev_seg.panels_span
                        cursor += n
                        if wing.symmetry > 0:
                            cursor += n

                    n = nc * ns
                    orig_flat = np.arange(cursor, cursor + n)
                    orig_idx = orig_flat.reshape(ns, nc)

                    mirror_idx = None
                    if wing.symmetry > 0:
                        mirror_flat = np.arange(cursor + n, cursor + 2 * n)
                        mirror_idx = mirror_flat.reshape(ns, nc)

                    # cross-check: segment IDs should match
                    expected_seg_id = seg_global_id
                    observed = panel_seg_id[orig_flat]
                    if not np.all(observed == expected_seg_id):
                        warnings.warn(
                            f"[VLMLoadingDistribution] panel_segment_id mismatch for "
                            f"wing '{wing_uid}' seg '{seg_uid}' (expected {expected_seg_id}, "
                            f"got unique {np.unique(observed)}). Using index arithmetic.",
                            stacklevel=2,
                        )

                    blocks.append(
                        {
                            "wing_uid": wing_uid,
                            "segment_uid": seg_uid,
                            "segment_id": seg_global_id,
                            "nc": nc,
                            "ns": ns,
                            "symmetry": wing.symmetry,
                            "orig_idx": orig_idx,
                            "mirror_idx": mirror_idx,
                        }
                    )
                seg_global_id += 1

        return blocks

    # EXTRACT

    @staticmethod
    def extract_distributions(
        vlm_solver,
        surface_name: str,
        U_ref: np.ndarray | None,
        density: float,
    ) -> dict[str, Any]:
        """Extract spanwise and chordwise loading from existing per-panel arrays.

        Returns {'spanwise': pd.DataFrame, 'chordwise': pd.DataFrame}.
        """
        import pandas as pd

        if not vlm_solver._solved:
            return {"spanwise": pd.DataFrame(), "chordwise": pd.DataFrame()}

        # ---- per-panel data (no recompute) ----
        forces_np = vlm_solver.lattice.get_forces()  # (N,3)
        gamma_np = vlm_solver.lattice.get_circulation()  # (N,)
        corners_np = vlm_solver.lattice.corners.to_numpy()[
            : vlm_solver.lattice.num_panels
        ]  # (N,4,3)
        bm_np = vlm_solver.lattice.bound_midpoints.to_numpy()[
            : vlm_solver.lattice.num_panels
        ]  # (N,3)
        bv_np = vlm_solver.lattice.bound_velocity.to_numpy()[
            : vlm_solver.lattice.num_panels
        ]  # (N,3)
        kin_np = vlm_solver.lattice.kinematic_velocity.to_numpy()[
            : vlm_solver.lattice.num_panels
        ]  # (N,3)
        vp_np = vlm_solver.lattice.vortex_points.to_numpy()[
            : vlm_solver.lattice.num_panels
        ]  # (N,4,3)

        # V∞ — prefer explicit U_ref over the kinematic_velocity heuristic to
        # handle the static wind-frame case where kinematic_velocity is ~0.
        if U_ref is not None:
            V_inf_mag = float(np.linalg.norm(U_ref))
        else:
            kin_mag = np.linalg.norm(kin_np, axis=1)
            V_inf_mag = float(np.median(kin_mag[kin_mag > 1e-8])) if kin_mag.max() > 1e-8 else 1.0
        V_inf_mag = max(V_inf_mag, 1e-10)
        q_inf = 0.5 * density * V_inf_mag**2

        # Lift direction in wind axes (perpendicular to U_ref in the vertical plane)
        if U_ref is not None and np.linalg.norm(U_ref) > 1e-10:
            V_hat = np.asarray(U_ref) / np.linalg.norm(U_ref)
        else:
            V_hat = np.array([1.0, 0.0, 0.0])
        z_hat = np.array([0.0, 0.0, 1.0])
        L_hat = z_hat - np.dot(z_hat, V_hat) * V_hat
        L_hat_n = np.linalg.norm(L_hat)
        L_hat = L_hat / L_hat_n if L_hat_n > 1e-10 else z_hat
        D_hat = V_hat.copy()

        # ΔCp denominator: panel chord from corners (TE_mid − LE_mid)
        te_mid = 0.5 * (corners_np[:, 3] + corners_np[:, 2])  # S+R / 2
        le_mid = 0.5 * (corners_np[:, 0] + corners_np[:, 1])  # P+Q / 2
        panel_chord = np.linalg.norm(te_mid - le_mid, axis=1)  # (N,)

        blocks = VLMLoadingDistribution.build_surface_grid_index(vlm_solver, surface_name)

        full_span: list[dict] = []
        full_chord: list[dict] = []

        for blk in blocks:
            nc = blk["nc"]
            ns = blk["ns"]
            wing_uid = blk["wing_uid"]
            seg_uid = blk["segment_uid"]

            for half, idx2d in [("orig", blk["orig_idx"]), ("mirror", blk["mirror_idx"])]:
                if idx2d is None:
                    continue

                # Span axis ŝ for this block — do NOT assume the global y-axis.
                # A vertical fin or a rotated rotor blade has bound legs with
                # ~zero y-component, which would drive dy → 0 and blow up
                # L_prime/cl.  The span direction is what the bound legs
                # (V3−V2) actually point along; for a y-aligned wing this
                # reduces exactly to the previous |Δy| behaviour.
                block_legs = vp_np[idx2d.ravel(), 2] - vp_np[idx2d.ravel(), 1]  # (ns·nc, 3)
                s_hat = block_legs.mean(axis=0)
                s_norm = np.linalg.norm(s_hat)
                s_hat = s_hat / s_norm if s_norm > 1e-12 else np.array([0.0, 1.0, 0.0])
                # Deterministic orientation: largest-magnitude component positive,
                # so 'orig' and 'mirror' halves share one axis and stations sort
                # consistently run-to-run.
                k_dom = int(np.argmax(np.abs(s_hat)))
                if s_hat[k_dom] < 0.0:
                    s_hat = -s_hat

                for j in range(ns):
                    cidx = idx2d[j]  # chord-panel indices for station j, shape (nc,)
                    F_j = forces_np[cidx]  # (nc, 3)
                    G_j = gamma_np[cidx]  # (nc,)
                    pc_j = panel_chord[cidx]  # (nc,)
                    bm_j = bm_np[cidx]  # (nc, 3)
                    Vrel_j = bv_np[cidx] - kin_np[cidx]  # (nc, 3)

                    # spanwise station coordinate: bound midpoints projected on ŝ
                    span_coordinate = float(np.mean(bm_j @ s_hat))
                    span_coordinate_abs = abs(span_coordinate)

                    # Physical strip edges projected on the same span axis.  The
                    # loading lives at panel centres, but the surface span is set
                    # by its corner coordinates.  Inferring the span from the
                    # first/last panel centres maps those centres to ±1 and makes
                    # a finite, cell-centred circulation look non-zero at the
                    # physical wing tip in exported plots.
                    station_corner_coordinates = corners_np[cidx].reshape(-1, 3) @ s_hat
                    span_edge_min = float(station_corner_coordinates.min())
                    span_edge_max = float(station_corner_coordinates.max())

                    # station width: bound-leg length (V3−V2) projected on ŝ
                    bound_legs = vp_np[cidx, 2] - vp_np[cidx, 1]  # (nc, 3)
                    dy = float(np.mean(np.abs(bound_legs @ s_hat)))
                    dy = max(dy, 1e-10)

                    chord_local = float(pc_j.sum())
                    Vrel_local = float(np.mean(np.linalg.norm(Vrel_j, axis=1)))

                    # sectional forces
                    F_sec = F_j.sum(axis=0)  # (3,)
                    L_sec = float(np.dot(F_sec, L_hat))
                    D_sec = float(np.dot(F_sec, D_hat))
                    Gamma_sec = float(G_j.sum())
                    Gamma_abs = abs(Gamma_sec)

                    Lprime = L_sec / dy
                    Dprime = D_sec / dy
                    cl = Lprime / (q_inf * chord_local) if q_inf * chord_local > 1e-15 else 0.0
                    cl_from_gamma = (
                        2.0 * Gamma_abs / (chord_local * Vrel_local)
                        if chord_local * Vrel_local > 1e-15
                        else 0.0
                    )

                    full_span.append(
                        {
                            "wing_uid": wing_uid,
                            "segment_uid": seg_uid,
                            "station_id": f"{wing_uid}:{seg_uid}:{half}:{j}",
                            "half": half,
                            "span_index": j,
                            "span_coordinate": span_coordinate,
                            "span_coordinate_abs": span_coordinate_abs,
                            "span_edge_min": span_edge_min,
                            "span_edge_max": span_edge_max,
                            "r": span_coordinate_abs,
                            "y": span_coordinate,
                            "chord_local": chord_local,
                            "dy": dy,
                            "Gamma": Gamma_sec,
                            "Gamma_abs": Gamma_abs,
                            "V_rel": Vrel_local,
                            "cl_from_gamma": cl_from_gamma,
                            "L_prime": Lprime,
                            "D_prime": Dprime,
                            "cl": cl,
                            "Fx_sec": float(F_sec[0]),
                            "Fy_sec": float(F_sec[1]),
                            "Fz_sec": float(F_sec[2]),
                        }
                    )

                    # chordwise
                    # x/c at panel midpoints: cumulative chord from LE
                    le_mids_j = le_mid[cidx]  # (nc, 3)
                    te_mids_j = te_mid[cidx]  # (nc, 3)
                    panel_centers_chord = 0.5 * (le_mids_j + te_mids_j)
                    # x_over_c: fractional distance along chord
                    if chord_local > 1e-10:
                        x_over_c = (
                            np.linalg.norm(panel_centers_chord - le_mids_j[0], axis=1) / chord_local
                        )
                    else:
                        x_over_c = np.zeros(nc)

                    denom_cp = V_inf_mag * pc_j
                    delta_cp = np.where(denom_cp > 1e-15, 2.0 * G_j / denom_cp, 0.0)

                    for i in range(nc):
                        full_chord.append(
                            {
                                "wing_uid": wing_uid,
                                "segment_uid": seg_uid,
                                "station_id": f"{wing_uid}:{seg_uid}:{half}:{j}",
                                "half": half,
                                "span_index": j,
                                "span_coordinate": span_coordinate,
                                "span_coordinate_abs": span_coordinate_abs,
                                "span_edge_min": span_edge_min,
                                "span_edge_max": span_edge_max,
                                "r": span_coordinate_abs,
                                "y": span_coordinate,
                                "chord_index": i,
                                "x_over_c": float(x_over_c[i]),
                                "panel_chord": float(pc_j[i]),
                                "Gamma_panel": float(G_j[i]),
                                "DeltaCp": float(delta_cp[i]),
                                "Fx": float(F_j[i, 0]),
                                "Fy": float(F_j[i, 1]),
                                "Fz": float(F_j[i, 2]),
                            }
                        )

        # assemble and sort by stable physical station.  ``y`` is retained as a
        # signed span-coordinate compatibility alias, but rotating blades can
        # change its sign with azimuth; the absolute coordinate remains root-tip
        # ordered in the per-time CSV block.
        df_span = (
            pd.DataFrame(full_span)
            .sort_values(["half", "span_coordinate_abs"])
            .reset_index(drop=True)
        )
        df_chord = (
            pd.DataFrame(full_chord)
            .sort_values(["half", "span_coordinate_abs", "chord_index"])
            .reset_index(drop=True)
        )

        # Full physical span for y_over_b.  Use strip edges, not the outermost
        # cell centres: panel-centred output should remain strictly inside
        # [-1, 1], leaving the actual tip locations available for the Γ=0
        # closure used by visualisation and higher-order reconstruction.
        if not df_span.empty:
            span_min = float(df_span["span_edge_min"].min())
            span_max = float(df_span["span_edge_max"].max())
            b = span_max - span_min
            b = max(b, 1e-10)
            y_mid = 0.5 * (span_max + span_min)
            df_span["y_over_b"] = 2.0 * (df_span["y"] - y_mid) / b
            df_chord["y_over_b"] = 2.0 * (df_chord["y"] - y_mid) / b
        else:
            df_span["y_over_b"] = np.nan
            df_chord["y_over_b"] = np.nan

        return {"spanwise": df_span, "chordwise": df_chord}

    # EXPORT

    @staticmethod
    def export_distribution_csv(
        vlm_solver,
        surface_name: str,
        dists: dict[str, Any],
        time: float,
        step: int,
        backup_directory: str,
        sample_subdirectory: str | None = None,
    ) -> None:
        """Append one time-step's distributions to the per-surface CSVs."""
        import pandas as pd

        samples_dir = resolve_samples_dir(backup_directory, sample_subdirectory)
        samples_dir.mkdir(parents=True, exist_ok=True)

        safe_name = surface_name.replace("/", "_").replace(" ", "_")

        for key, fname_tpl in [
            ("spanwise", f"vlm_spanwise_{safe_name}.csv"),
            ("chordwise", f"vlm_chordwise_{safe_name}.csv"),
        ]:
            df: pd.DataFrame = dists.get(key, pd.DataFrame())
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
