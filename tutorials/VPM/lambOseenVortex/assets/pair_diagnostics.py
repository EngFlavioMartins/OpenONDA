"""Compact particle-moment diagnostics for Lamb--Oseen vortex pairs."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


CSV_COLUMNS = (
    "flow_time",
    "time_step",
    "x_core",
    "y_core",
    "core_radius",
    "separation",
    "core_area",
    "angle_rad",
    "surface_circulation",
    "merged",
    "core_0_x",
    "core_0_y",
    "core_1_x",
    "core_1_y",
)


def _group_moment(
    positions: np.ndarray,
    weights: np.ndarray,
    radii: np.ndarray,
    group_ids: np.ndarray,
    group: int,
) -> tuple[np.ndarray, float, float]:
    selected = group_ids == group
    group_weights = weights[selected]
    total = float(group_weights.sum())
    if total <= np.finfo(float).tiny:
        return np.full(2, np.nan), np.nan, 0.0

    group_positions = positions[selected]
    center = np.einsum("i,ij->j", group_weights, group_positions) / total
    radial_distance_squared = np.sum((group_positions - center) ** 2, axis=1)
    field_radius_squared = (
        np.dot(
            group_weights,
            radial_distance_squared + radii[selected] ** 2,
        )
        / total
    )
    return center, float(field_radius_squared), total


def _particle_data(solver) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(solver.particles_positions, dtype=np.float64)
    strengths = np.asarray(solver.particles_strengths, dtype=np.float64)
    radii = np.asarray(solver.particles_radii, dtype=np.float64)
    group_ids = np.asarray(solver.particles_group_ids, dtype=np.int32)
    weights = np.linalg.norm(strengths, axis=1)

    finite = (
        np.isfinite(positions).all(axis=1)
        & np.isfinite(weights)
        & np.isfinite(radii)
        & (weights > 0.0)
        & (radii > 0.0)
    )
    return positions[finite], weights[finite], radii[finite], group_ids[finite]


class PairDiagnosticsSampler:
    """Track mid-plane vortex-pair motion without saving field grids."""

    file_name = "pair_diagnostics"

    def __init__(
        self,
        case_name: str,
        separation: float,
        column_length: float,
        slab_half_width: float | None = None,
    ) -> None:
        if case_name not in {"dipole", "merging"}:
            raise ValueError(f"Pair diagnostics do not apply to {case_name!r}")
        self.case_name = case_name
        self.separation = float(separation)
        self.column_length = float(column_length)
        self.slab_half_width = slab_half_width
        self.last_step: int | None = None

    def _restore(self, path: Path) -> None:
        if self.last_step is not None or not path.is_file() or path.stat().st_size == 0:
            return
        try:
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            valid_steps = [int(float(row["time_step"])) for row in rows if row.get("time_step")]
            self.last_step = max(valid_steps, default=None)
        except (OSError, ValueError, KeyError):
            self.last_step = None

    def save_csv(
        self,
        solver,
        path: Path,
        *,
        time: float,
        step: int | None,
    ) -> None:
        self._restore(path)
        if step is not None and self.last_step is not None and step <= self.last_step:
            return

        all_positions, all_weights, all_radii, all_group_ids = _particle_data(solver)
        circulation_0_full = float(all_weights[all_group_ids == 0].sum()) / self.column_length
        circulation_full = float(all_weights.sum()) / self.column_length

        if self.slab_half_width is None:
            central = np.ones(len(all_positions), dtype=bool)
        else:
            central = np.abs(all_positions[:, 2]) <= self.slab_half_width
            if np.count_nonzero(central) < 4:
                nearest = np.argsort(np.abs(all_positions[:, 2]))[:4]
                central = np.zeros(len(all_positions), dtype=bool)
                central[nearest] = True

        positions = all_positions[central, :2]
        weights = all_weights[central]
        radii = all_radii[central]
        group_ids = all_group_ids[central]
        center_0, radius_squared_0, circulation_0 = _group_moment(
            positions,
            weights,
            radii,
            group_ids,
            0,
        )

        if self.case_name == "dipole":
            row = (
                *center_0,
                np.sqrt(radius_squared_0),
                np.nan,
                np.nan,
                np.nan,
                circulation_0_full,
                False,
                *center_0,
                np.nan,
                np.nan,
            )
        else:
            center_1, _, circulation_1 = _group_moment(
                positions,
                weights,
                radii,
                group_ids,
                1,
            )
            centers = np.array([center_0, center_1])
            separation = float(np.linalg.norm(center_0 - center_1))
            midpoint = centers.mean(axis=0)
            selected = np.isin(group_ids, (0, 1))
            pair_weights = weights[selected]
            pair_positions = positions[selected]
            pair_radii = radii[selected]
            total = float(pair_weights.sum())
            field_radius_squared = (
                np.dot(
                    pair_weights,
                    np.sum((pair_positions - midpoint) ** 2, axis=1) + pair_radii**2,
                )
                / total
            )
            core_area = 0.5 * max(field_radius_squared - (separation / 2.0) ** 2, 0.0)
            angle = float(np.arctan2(center_0[1] - midpoint[1], center_0[0] - midpoint[0]))
            merged = not np.isfinite(separation) or not (
                0.05 * self.separation <= separation <= 1.15 * self.separation
            )
            row = (
                *center_0,
                np.nan,
                separation,
                core_area,
                angle,
                circulation_full,
                merged,
                *centers.ravel(),
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            if write_header:
                writer.writerow(CSV_COLUMNS)
            writer.writerow([time, "" if step is None else step, *row])
        self.last_step = step
