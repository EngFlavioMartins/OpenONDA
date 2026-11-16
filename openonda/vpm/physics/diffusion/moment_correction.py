"""Weak-node removal with the nine-moment correction of the GBD regeneration.

Grid diffusion keeps the nodes whose strength reaches a threshold and drops the rest.
The dropped strength moves to its nearest retained node, and a weighted minimum-norm
correction then restores the total vortex strength, linear impulse and angular impulse of
the complete diffused field. The retained support is augmented first when its geometry
cannot carry the nine constraints, and a component whose correction would be too large
keeps its diffused field instead. All of it is host NumPy on bounded chunks.
"""

from collections.abc import Callable

import numpy as np

# Removed nodes per batch of wall-visibility queries.
_WALL_QUERY_CHUNK = 65536

# The nine moment constraints need 27 coefficients per survivor when written
# as one dense tensor.  Production clouds can contain O(10^6) survivors, so
# form that tensor only in bounded host chunks while accumulating the fixed
# 9x9 normal matrix and applying its correction.
_GBD_MOMENT_CHUNK_SIZE = 65536
_GBD_MOMENT_RCOND = 1.0e-12
_GBD_MOMENT_CONDITION_LIMIT = 1.0e12
_GBD_MOMENT_RESIDUAL_LIMIT = 1.0e-5
# If the complete post-diffusion field has one vortex-vector direction, the
# moment correction must stay in that physical subspace.  A fully vector-valued
# nine-moment solve can otherwise satisfy the same global moments by inventing
# transverse strength components.
_GBD_DIRECTION_SUBSPACE_TOLERANCE = 1.0e-12
# A deficient support normally needs at most three replacements.  Limit the
# cap-full search to a small weakest-node frontier so each candidate is judged
# with fixed 9x9 algebra instead of an O(N_retained) trial cloud.
_GBD_MOMENT_REPLACEMENT_POOL_SIZE = 32
# A moment correction whose L1 change is comparable with the complete diffused
# field is no longer small.  Stop before mutating the production grid instead of
# accepting a cloud that meets the moments numerically but not physically.
_GBD_MOMENT_CORRECTION_FRACTION_LIMIT = 0.5


class GBDMomentCorrectionError(RuntimeError):
    """A retained support cannot safely represent its discarded moments."""

    def __init__(self, message: str, *, correction_label: int | None):
        super().__init__(message)
        self.correction_label = correction_label


def nearest_fluid_connected_nodes(
    survivors: np.ndarray, removed: np.ndarray, segments_cross_wall: Callable | None = None
) -> np.ndarray:
    """Return nearest retained nodes, querying occluded alternatives in batches."""
    from scipy.spatial import cKDTree  # pyrefly: ignore[missing-module-attribute]

    tree = cKDTree(survivors, compact_nodes=False)
    nearest = tree.query(removed, k=1, workers=-1)[1]
    if segments_cross_wall is None:
        return nearest

    for start in range(0, len(removed), _WALL_QUERY_CHUNK):
        stop = min(start + _WALL_QUERY_CHUNK, len(removed))
        hidden = segments_cross_wall(removed[start:stop], survivors[nearest[start:stop]])
        nearest[start:stop][hidden] = -1
    pending = np.flatnonzero(nearest < 0)
    # Asking for a list of ranks stores only that list's results, rather than
    # a growing removed-by-retained matrix. Most removed nodes use the initial
    # nearest result; only those behind the wall enter these geometry queries.
    for first_rank in range(2, len(survivors) + 1, 32):
        if not len(pending):
            break
        ranks = np.arange(first_rank, min(first_rank + 32, len(survivors) + 1))
        batch_size = max(1, _WALL_QUERY_CHUNK // len(ranks))
        for start in range(0, len(pending), batch_size):
            rows = pending[start : start + batch_size]
            candidates = tree.query(removed[rows], k=ranks, workers=-1)[1]
            sources = np.broadcast_to(removed[rows, None, :], (*candidates.shape, 3))
            connected = ~segments_cross_wall(
                sources.reshape(-1, 3), survivors[candidates].reshape(-1, 3)
            ).reshape(candidates.shape)
            found = connected.any(axis=1)
            nearest[rows[found]] = candidates[found, connected[found].argmax(axis=1)]
        pending = pending[nearest[pending] < 0]
    if len(pending):
        raise RuntimeError(
            "GBD weak-strength removal has no fluid-connected retained node for a removed "
            "node; reduce the removal threshold or increase the regeneration cap."
        )
    return nearest


def moment_constraint_rows(scaled_position: np.ndarray) -> np.ndarray:
    """Return one bounded chunk of the nine GBD moment constraints."""
    radius_squared = np.einsum(
        "ij,ij->i",
        scaled_position,
        scaled_position,
    )
    rows = np.zeros((len(scaled_position), 9, 3), dtype=np.float64)
    rows[:, :3, :] = np.eye(3)
    rows[:, 3, 1] = -scaled_position[:, 2]
    rows[:, 3, 2] = scaled_position[:, 1]
    rows[:, 4, 0] = scaled_position[:, 2]
    rows[:, 4, 2] = -scaled_position[:, 0]
    rows[:, 5, 0] = -scaled_position[:, 1]
    rows[:, 5, 1] = scaled_position[:, 0]
    rows[:, 6:, :] = (
        np.einsum("ij,ik->ijk", scaled_position, scaled_position)
        - radius_squared[:, None, None] * np.eye(3)
    ) / 3.0
    return rows


def moment_gram(
    position: np.ndarray,
    vortex_strength: np.ndarray,
    particle_spacing: float,
    direction: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Build the bounded-memory weighted moment Gram matrix.

    With a ``direction`` the corrections are restricted to the one vector direction
    of a rank-one field; the constraint rows are projected onto it.
    """
    support = "moment" if direction is None else "directional moment"
    weight_magnitude = np.linalg.norm(vortex_strength, axis=1)
    weight_sum = float(weight_magnitude.sum(dtype=np.float64))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise RuntimeError(f"GBD {support} support has no finite retained strength")
    weights = weight_magnitude / weight_sum
    length_scale = max(
        float(
            np.sqrt(
                np.sum(
                    weights
                    * np.einsum(
                        "ij,ij->i",
                        position,
                        position,
                    )
                )
            )
        ),
        float(particle_spacing),
    )
    gram = np.zeros((9, 9), dtype=np.float64)
    for start in range(0, len(position), _GBD_MOMENT_CHUNK_SIZE):
        stop = min(start + _GBD_MOMENT_CHUNK_SIZE, len(position))
        rows = moment_constraint_rows(position[start:stop] / length_scale)
        if direction is None:
            gram += np.einsum(
                "i,ijk,ilk->jl",
                weights[start:stop],
                rows,
                rows,
            )
        else:
            projected_rows = np.einsum("ijk,k->ij", rows, direction)
            gram += np.einsum(
                "i,ij,ik->jk",
                weights[start:stop],
                projected_rows,
                projected_rows,
            )
    return gram, length_scale, weights


def rank_one_direction(*vortex_strengths: np.ndarray) -> np.ndarray | None:
    """Return the shared vortex-vector direction of a rank-one field, if any."""
    covariance = np.zeros((3, 3), dtype=np.float64)
    for vortex_strength in vortex_strengths:
        strength = np.asarray(vortex_strength, dtype=np.float64)
        covariance += strength.T @ strength
    if not np.all(np.isfinite(covariance)):
        return None
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    total_energy = float(eigenvalues.sum(dtype=np.float64))
    if total_energy <= 0.0:
        return None
    transverse_fraction = max(0.0, total_energy - float(eigenvalues[-1])) / total_energy
    if transverse_fraction > _GBD_DIRECTION_SUBSPACE_TOLERANCE:
        return None
    return eigenvectors[:, -1]


def moment_gram_quality(gram: np.ndarray) -> tuple[int, float]:
    """Return numerical rank and condition under the moment-correction limits."""
    if not np.all(np.isfinite(gram)):
        return 0, float("inf")
    singular_values = np.linalg.svd(gram, compute_uv=False)
    if len(singular_values) == 0 or singular_values[0] <= 0.0:
        return 0, float("inf")
    cutoff = _GBD_MOMENT_RCOND * singular_values[0]
    rank = int(np.count_nonzero(singular_values > cutoff))
    condition = (
        float("inf")
        if rank == 0 or singular_values[rank - 1] <= 0.0
        else float(singular_values[0] / singular_values[rank - 1])
    )
    return rank, condition


def augment_moment_correction_support(
    grid_np: np.ndarray,
    vortex_strength_magnitude: np.ndarray,
    ix: np.ndarray,
    iy: np.ndarray,
    iz: np.ndarray,
    grid_min_np: np.ndarray,
    particle_spacing: float,
    cap: int,
    labels: np.ndarray | None = None,
    strict_labels: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, bool]:
    """Introduce the strongest removed nodes needed for a well-conditioned moment basis.

    The threshold/cap selection remains the physical decision on the particle
    count.  This method only augments that set when its geometry cannot
    carry the nine moment constraints.  It first fills unused
    capacity, then (only when the cap is full) exchanges the weakest
    retained node for the strongest discarded candidate that improves the
    moment basis.  It never exceeds the cap.

    The final boolean selects a moment correction per group.  If any group
    was sparse in the original threshold result, the correction is global.
    """
    shape = vortex_strength_magnitude.shape
    flat_grid = grid_np.reshape(-1, 3)
    flat_magnitude = vortex_strength_magnitude.reshape(-1)
    retained_flat = np.ravel_multi_index((ix, iy, iz), shape).astype(np.int64)
    nonzero_flat = np.flatnonzero(flat_magnitude > 0.0)
    if len(retained_flat) >= len(nonzero_flat):
        return ix, iy, iz, 0, False

    retained_marker = np.zeros(flat_magnitude.shape, dtype=bool)
    retained_marker[retained_flat] = True
    candidate_flat = nonzero_flat[~retained_marker[nonzero_flat]]

    flat_labels = None if labels is None else np.asarray(labels).reshape(-1)
    preserve_groups = False
    unique_labels: np.ndarray | None = None
    if flat_labels is not None:
        unique_labels = np.unique(flat_labels[nonzero_flat])
        retained_labels = flat_labels[retained_flat]

        def minimum_group_support(label: int) -> int:
            group_nonzero = nonzero_flat[flat_labels[nonzero_flat] == label]
            return 6 if rank_one_direction(flat_grid[group_nonzero]) is not None else 4

        preserve_groups = strict_labels or (
            len(unique_labels) > 1
            and all(
                np.count_nonzero(retained_labels == label) >= minimum_group_support(int(label))
                for label in unique_labels
            )
        )

    selected = retained_flat.tolist()
    selected_set = set(selected)
    introduced: set[int] = set()
    retired: set[int] = set()

    def support_geometry(selected_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sx, sy, sz = np.unravel_index(selected_flat, shape)
        position = np.stack(
            [
                grid_min_np[0] + sx * particle_spacing,
                grid_min_np[1] + sy * particle_spacing,
                grid_min_np[2] + sz * particle_spacing,
            ],
            axis=1,
        ).astype(np.float64)
        strength = flat_grid[selected_flat].astype(np.float64)
        return position, strength

    def quality_improves(
        trial_rank: int,
        trial_condition: float,
        rank: int,
        condition: float,
    ) -> bool:
        if trial_rank != rank:
            return trial_rank > rank
        if not np.isfinite(trial_condition):
            return False
        return not np.isfinite(condition) or trial_condition < condition * (1.0 - 1.0e-10)

    def augment_scope(label: int | None) -> None:
        def scope_selection(*, with_indices: bool = False) -> np.ndarray:
            selected_array = np.asarray(selected, dtype=np.int64)
            if label is None or flat_labels is None:
                selection = np.arange(len(selected_array), dtype=np.int64)
            else:
                selection = np.flatnonzero(flat_labels[selected_array] == label)
            return selection if with_indices else selected_array[selection]

        scoped_candidates = candidate_flat
        if label is not None and flat_labels is not None:
            scoped_candidates = scoped_candidates[flat_labels[scoped_candidates] == label]
        # A per-group correction needs no basis for a group from which no
        # node was removed.
        if len(scoped_candidates) == 0:
            return

        scoped_nonzero = nonzero_flat
        if label is not None and flat_labels is not None:
            scoped_nonzero = scoped_nonzero[flat_labels[scoped_nonzero] == label]
        direction = rank_one_direction(flat_grid[scoped_nonzero])
        # A scalar strength correction along one fixed 3-D direction has
        # at most six independent moment modes: one total vortex strength, two
        # linear-impulse, and three angular-impulse modes.  Degenerate
        # physical geometries can carry fewer, so measure the rank of the
        # complete support before removal rather than requiring nonexistent
        # modes from the retained subset.
        target_rank = 9
        if direction is not None:
            complete_position, complete_strength = support_geometry(scoped_nonzero)
            complete_gram, _complete_scale, _complete_weights = moment_gram(
                complete_position,
                complete_strength,
                particle_spacing,
                direction,
            )
            target_rank, _complete_condition = moment_gram_quality(complete_gram)

        def support_gram(
            selected_flat: np.ndarray,
        ) -> tuple[np.ndarray, float, np.ndarray]:
            position, strength = support_geometry(selected_flat)
            return moment_gram(position, strength, particle_spacing, direction)

        def support_quality(selected_flat: np.ndarray) -> tuple[int, float]:
            if len(selected_flat) < 4:
                return 0, float("inf")
            gram, _length_scale, _weights = support_gram(selected_flat)
            return moment_gram_quality(gram)

        rank, condition = support_quality(scope_selection())
        if rank == target_rank and condition <= _GBD_MOMENT_CONDITION_LIMIT:
            return

        candidate_order = np.argsort(
            -flat_magnitude[scoped_candidates],
            kind="stable",
        )
        scoped_candidates = scoped_candidates[candidate_order]

        for candidate in scoped_candidates:
            if len(selected) >= cap:
                break
            candidate_int = int(candidate)
            if candidate_int in selected_set:
                continue
            selected.append(candidate_int)
            selected_set.add(candidate_int)
            introduced.add(candidate_int)
            rank, condition = support_quality(scope_selection())
            if rank == target_rank and condition <= _GBD_MOMENT_CONDITION_LIMIT:
                return
        else:
            # Every nonzero node in this scope is now retained, so there is
            # no discarded strength for its moment basis to redistribute.
            return

        # The cap is full but its support can still be geometrically
        # deficient (for example, four nearly collinear strongest nodes).
        # Keep the strongest selected nodes whenever possible: examine a
        # bounded weakest-node frontier and accept the first, strongest
        # discarded candidate that gives a strict rank/condition gain.
        while len(selected) >= cap:
            scoped_flat = scope_selection()
            if len(scoped_flat) < 4:
                break
            position, strength = support_geometry(scoped_flat)
            gram, length_scale, _weights = support_gram(scoped_flat)
            rank, condition = moment_gram_quality(gram)
            if rank == target_rank and condition <= _GBD_MOMENT_CONDITION_LIMIT:
                return
            raw_gram = gram * float(np.linalg.norm(strength, axis=1).sum(dtype=np.float64))

            scope_indices = scope_selection(with_indices=True)
            weakest_order = np.argsort(
                flat_magnitude[scoped_flat],
                kind="stable",
            )[:_GBD_MOMENT_REPLACEMENT_POOL_SIZE]
            replacement_indices = scope_indices[weakest_order]
            replacement_flat = scoped_flat[weakest_order]
            replacement_rows = moment_constraint_rows(position[weakest_order] / length_scale)
            if direction is not None:
                replacement_rows = np.einsum(
                    "ijk,k->ij",
                    replacement_rows,
                    direction,
                )
                replacement_contributions = np.einsum(
                    "i,ij,ik->ijk",
                    np.linalg.norm(strength[weakest_order], axis=1),
                    replacement_rows,
                    replacement_rows,
                )
            else:
                replacement_contributions = np.einsum(
                    "i,ijk,ilk->ijl",
                    np.linalg.norm(strength[weakest_order], axis=1),
                    replacement_rows,
                    replacement_rows,
                )

            accepted = False
            for candidate in scoped_candidates:
                candidate_int = int(candidate)
                if candidate_int in selected_set or candidate_int in retired:
                    continue
                cx, cy, cz = np.unravel_index(candidate_int, shape)
                candidate_position = np.asarray(
                    [
                        grid_min_np[0] + cx * particle_spacing,
                        grid_min_np[1] + cy * particle_spacing,
                        grid_min_np[2] + cz * particle_spacing,
                    ],
                    dtype=np.float64,
                )
                candidate_rows = moment_constraint_rows(candidate_position[None, :] / length_scale)[
                    0
                ]
                if direction is not None:
                    candidate_rows = candidate_rows @ direction
                    candidate_contribution = flat_magnitude[candidate_int] * np.outer(
                        candidate_rows,
                        candidate_rows,
                    )
                else:
                    candidate_contribution = flat_magnitude[candidate_int] * (
                        candidate_rows @ candidate_rows.T
                    )

                improving_replacements: list[int] = []
                for local_index, removed_contribution in enumerate(replacement_contributions):
                    trial_gram = raw_gram - removed_contribution + candidate_contribution
                    trial_rank, trial_condition = moment_gram_quality(trial_gram)
                    if not quality_improves(
                        trial_rank,
                        trial_condition,
                        rank,
                        condition,
                    ):
                        continue
                    improving_replacements.append(local_index)
                if not improving_replacements:
                    continue

                # ``weakest_order`` is ascending, so retain every stronger
                # threshold/cap winner unless replacing a weaker node does
                # not produce a genuine full-support improvement after the
                # exact length-scale normalization is recomputed.
                for local_index in improving_replacements:
                    replacement_index = int(replacement_indices[local_index])
                    removed_flat = int(replacement_flat[local_index])
                    selected[replacement_index] = candidate_int
                    trial_rank, trial_condition = support_quality(scope_selection())
                    if not quality_improves(
                        trial_rank,
                        trial_condition,
                        rank,
                        condition,
                    ):
                        selected[replacement_index] = removed_flat
                        continue

                    selected_set.remove(removed_flat)
                    selected_set.add(candidate_int)
                    retired.add(removed_flat)
                    introduced.discard(removed_flat)
                    introduced.add(candidate_int)
                    rank, condition = trial_rank, trial_condition
                    accepted = True
                    break
                if accepted:
                    break
            if not accepted:
                break
            if rank == target_rank and condition <= _GBD_MOMENT_CONDITION_LIMIT:
                return
        scope_name = "the complete cloud" if label is None else f"group {label!r}"
        raise RuntimeError(
            "GBD moment correction cannot form full-rank support within the "
            f"regeneration cap for {scope_name}: rank={rank}/{target_rank}, "
            f"condition={condition:.6e}, retained={len(selected):,}, cap={cap:,}"
        )

    if preserve_groups:
        assert unique_labels is not None
        for label in unique_labels:
            augment_scope(int(label))
    else:
        augment_scope(None)

    augmented_flat = np.asarray(selected, dtype=np.int64)
    augmented_ix, augmented_iy, augmented_iz = np.unravel_index(
        augmented_flat,
        shape,
    )
    return (
        np.asarray(augmented_ix),
        np.asarray(augmented_iy),
        np.asarray(augmented_iz),
        len(introduced),
        preserve_groups,
    )


def retain_diffused_component(
    magnitude: np.ndarray,
    ix: np.ndarray,
    iy: np.ndarray,
    iz: np.ndarray,
    cap: int,
    labels: np.ndarray | None,
    correction_label: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Keep the diffused field of a component whose moment correction is too large.

    A rank-complete retained basis can still need an excessive strength correction
    when a small disconnected component loses much of its local strength.
    Retaining that component's diffused nodes preserves its field
    and all moments without moving any strength across component boundaries.
    Capacity remains a hard limit; no retained winner is replaced.
    """
    diffused_mask = magnitude > 0.0
    if correction_label is not None:
        if labels is None:
            raise RuntimeError(
                "GBD retention of a diffused component requires its correction labels"
            )
        diffused_mask &= labels == correction_label
    diffused_mask[ix, iy, iz] = False
    added_ix, added_iy, added_iz = np.where(diffused_mask)
    added_count = len(added_ix)
    if added_count == 0:
        raise RuntimeError("GBD moment correction has no additional diffused nodes to retain")
    required = len(ix) + added_count
    if required > cap:
        scope = "the complete cloud" if correction_label is None else f"group {correction_label}"
        raise RuntimeError(
            "GBD retention of the diffused field of "
            f"{scope} needs {required:,} particles, exceeding max_n_particles={cap:,}; "
            "increase the declared particle capacity"
        )
    return (
        np.concatenate((ix, added_ix)),
        np.concatenate((iy, added_iy)),
        np.concatenate((iz, added_iz)),
        added_count,
    )


def redistribute_removed_moments(
    grid_np: np.ndarray,
    vortex_strength_magnitude: np.ndarray,
    ix: np.ndarray,
    iy: np.ndarray,
    iz: np.ndarray,
    grid_min_np: np.ndarray,
    particle_spacing: float,
    labels: np.ndarray | None = None,
    diagnostics: dict[str, bool | int | float] | None = None,
    strict_labels: bool = False,
    segments_cross_wall: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Coarsen removed nodes locally while preserving diffusive moments.

    The retained cloud preserves total vortex strength, linear impulse, and
    angular impulse of the complete post-diffusion grid.  A nearest-node
    coarsening first keeps discarded tail strength close to its physical
    location; a weighted minimum-norm correction then satisfies the nine
    moment constraints.  Discarded nodes alone are queried against the
    retained-node tree, and the constraint algebra is evaluated in bounded
    chunks so memory use does not grow by 27 float64 values per survivor.

    The returned values have already been rounded to the storage dtype and
    validated.  Singular geometry, excessive correction, or moments that fail
    after the cast to storage precision raise before the production grid is mutated.
    """
    retained_vortex_strength = grid_np[ix, iy, iz].astype(np.float64)
    retained_position = np.stack(
        [
            grid_min_np[0] + ix * particle_spacing,
            grid_min_np[1] + iy * particle_spacing,
            grid_min_np[2] + iz * particle_spacing,
        ],
        axis=1,
    ).astype(np.float64)

    discarded_mask = vortex_strength_magnitude > 0.0
    discarded_mask[ix, iy, iz] = False
    discarded_i, discarded_j, discarded_k = np.where(discarded_mask)
    if len(discarded_i) == 0:
        if diagnostics is not None:
            diagnostics.update(
                correction_fraction=0.0,
                normalized_vortex_strength_residual=0.0,
                normalized_linear_impulse_residual=0.0,
                normalized_angular_impulse_residual=0.0,
            )
        return retained_vortex_strength.astype(grid_np.dtype, copy=False)
    discarded_vortex_strength = grid_np[
        discarded_i,
        discarded_j,
        discarded_k,
    ].astype(np.float64)
    discarded_position = np.stack(
        [
            grid_min_np[0] + discarded_i * particle_spacing,
            grid_min_np[1] + discarded_j * particle_spacing,
            grid_min_np[2] + discarded_k * particle_spacing,
        ],
        axis=1,
    ).astype(np.float64)

    def moment_state(
        position: np.ndarray,
        vortex_strength: np.ndarray,
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[float, float, float]]:
        linear_terms = np.cross(position, vortex_strength)
        angular_terms = np.cross(position, linear_terms) / 3.0
        moments = (
            vortex_strength.sum(axis=0, dtype=np.float64),
            linear_terms.sum(axis=0, dtype=np.float64),
            angular_terms.sum(axis=0, dtype=np.float64),
        )
        scales = (
            float(np.linalg.norm(vortex_strength, axis=1).sum(dtype=np.float64)),
            float(np.linalg.norm(linear_terms, axis=1).sum(dtype=np.float64)),
            float(np.linalg.norm(angular_terms, axis=1).sum(dtype=np.float64)),
        )
        return moments, scales

    def validate(
        survivor_position: np.ndarray,
        raw_survivor_strength: np.ndarray,
        removed_position: np.ndarray,
        removed_strength: np.ndarray,
        corrected: np.ndarray,
        *,
        scope: str,
        record_diagnostics: bool = False,
        correction_label: int | None = None,
    ) -> np.ndarray:
        stored = corrected.astype(grid_np.dtype, copy=False)
        stored_float64 = stored.astype(np.float64, copy=False)
        if not np.all(np.isfinite(stored_float64)):
            raise RuntimeError(f"GBD moment correction produced non-finite strengths for {scope}")

        survivor_moments, survivor_scales = moment_state(
            survivor_position,
            raw_survivor_strength,
        )
        removed_moments, removed_scales = moment_state(
            removed_position,
            removed_strength,
        )
        target_moments = tuple(
            survivor + removed
            for survivor, removed in zip(
                survivor_moments,
                removed_moments,
                strict=True,
            )
        )
        target_scales = tuple(
            survivor + removed
            for survivor, removed in zip(
                survivor_scales,
                removed_scales,
                strict=True,
            )
        )
        actual_moments, _actual_scales = moment_state(
            survivor_position,
            stored_float64,
        )

        complete_strength_scale = max(target_scales[0], np.finfo(np.float64).tiny)
        position_scale = max(
            float(
                np.sqrt(
                    np.einsum(
                        "ij,ij->i",
                        survivor_position,
                        survivor_position,
                    ).mean()
                )
            ),
            float(particle_spacing),
        )
        normalization = (
            complete_strength_scale,
            max(target_scales[1], complete_strength_scale * position_scale),
            max(target_scales[2], complete_strength_scale * position_scale**2),
        )
        relative_residual = np.asarray(
            [
                np.linalg.norm(target - actual) / scale
                for target, actual, scale in zip(
                    target_moments,
                    actual_moments,
                    normalization,
                    strict=True,
                )
            ],
            dtype=np.float64,
        )
        correction_fraction = float(
            np.linalg.norm(stored_float64 - raw_survivor_strength, axis=1).sum(dtype=np.float64)
            / complete_strength_scale
        )
        if not np.all(np.isfinite(relative_residual)) or np.any(
            relative_residual > _GBD_MOMENT_RESIDUAL_LIMIT
        ):
            message = (
                "GBD moment correction failed after the cast to storage precision for "
                f"{scope}: normalized residuals "
                f"gamma={relative_residual[0]:.6e}, "
                f"linear={relative_residual[1]:.6e}, "
                f"angular={relative_residual[2]:.6e}, "
                f"limit={_GBD_MOMENT_RESIDUAL_LIMIT:.6e}; "
                f"correction fraction={correction_fraction:.6e}"
            )
            # Large cancelling corrections from nearly dependent constraints
            # (e.g. one slab layer) lose the moments when cast to storage
            # precision. A fluid component keeps its diffused field instead.
            if correction_label is not None:
                raise GBDMomentCorrectionError(message, correction_label=correction_label)
            raise RuntimeError(message)

        if (
            not np.isfinite(correction_fraction)
            or correction_fraction > _GBD_MOMENT_CORRECTION_FRACTION_LIMIT
        ):
            raise GBDMomentCorrectionError(
                "GBD moment correction requires an excessive strength correction for "
                f"{scope}: fraction={correction_fraction:.6e}, "
                f"limit={_GBD_MOMENT_CORRECTION_FRACTION_LIMIT:.6e}",
                correction_label=correction_label,
            )
        if record_diagnostics and diagnostics is not None:
            diagnostics.update(
                correction_fraction=correction_fraction,
                normalized_vortex_strength_residual=float(relative_residual[0]),
                normalized_linear_impulse_residual=float(relative_residual[1]),
                normalized_angular_impulse_residual=float(relative_residual[2]),
            )
        return stored

    def redistribute(
        survivor_position: np.ndarray,
        survivor_vortex_strength: np.ndarray,
        removed_position: np.ndarray,
        removed_vortex_strength: np.ndarray,
        *,
        scope: str,
        record_diagnostics: bool = False,
        correction_label: int | None = None,
    ) -> np.ndarray:
        weight_magnitude = np.linalg.norm(survivor_vortex_strength, axis=1)
        weight_sum = float(weight_magnitude.sum())
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            raise RuntimeError(f"GBD moment correction has no finite retained strength for {scope}")
        if len(removed_position) == 0:
            return survivor_vortex_strength.astype(grid_np.dtype, copy=False)

        nearest = nearest_fluid_connected_nodes(
            survivor_position, removed_position, segments_cross_wall
        )
        corrected = survivor_vortex_strength.copy()
        np.add.at(corrected, nearest, removed_vortex_strength)

        survivor_moments, _survivor_scales = moment_state(
            survivor_position,
            survivor_vortex_strength,
        )
        removed_moments, _removed_scales = moment_state(
            removed_position,
            removed_vortex_strength,
        )
        corrected_moments, _corrected_scales = moment_state(
            survivor_position,
            corrected,
        )
        residuals = tuple(
            survivor + removed - current
            for survivor, removed, current in zip(
                survivor_moments,
                removed_moments,
                corrected_moments,
                strict=True,
            )
        )
        if not any(np.any(np.abs(residual) > 0.0) for residual in residuals):
            return validate(
                survivor_position,
                survivor_vortex_strength,
                removed_position,
                removed_vortex_strength,
                corrected,
                scope=scope,
                record_diagnostics=record_diagnostics,
                correction_label=correction_label,
            )
        if len(survivor_position) < 4:
            raise RuntimeError(
                "GBD moment correction requires at least four retained nodes "
                f"for {scope}; got {len(survivor_position)}"
            )

        direction = rank_one_direction(
            survivor_vortex_strength,
            removed_vortex_strength,
        )
        gram, length_scale, weights = moment_gram(
            survivor_position,
            survivor_vortex_strength,
            particle_spacing,
            direction,
        )
        right_hand_side = np.concatenate(
            (
                residuals[0],
                residuals[1] / length_scale,
                residuals[2] / length_scale**2,
            )
        )
        if not np.all(np.isfinite(gram)) or not np.all(np.isfinite(right_hand_side)):
            raise RuntimeError(f"GBD moment correction formed non-finite constraints for {scope}")

        if direction is not None:
            multipliers, _residual, rank, singular_values = np.linalg.lstsq(
                gram,
                right_hand_side,
                rcond=_GBD_MOMENT_RCOND,
            )
            condition = (
                float("inf")
                if rank == 0 or singular_values[rank - 1] <= 0.0
                else float(singular_values[0] / singular_values[rank - 1])
            )
            if not np.isfinite(condition) or condition > _GBD_MOMENT_CONDITION_LIMIT:
                raise RuntimeError(
                    "GBD directional moment correction constraints are ill-conditioned "
                    f"for {scope}: rank={rank}, condition={condition:.6e}, "
                    f"limit={_GBD_MOMENT_CONDITION_LIMIT:.6e}"
                )
            if not np.all(np.isfinite(multipliers)):
                raise RuntimeError(
                    f"GBD directional moment correction produced non-finite multipliers for {scope}"
                )

            moment_corrected = corrected.copy()
            for start in range(0, len(survivor_position), _GBD_MOMENT_CHUNK_SIZE):
                stop = min(start + _GBD_MOMENT_CHUNK_SIZE, len(survivor_position))
                rows = moment_constraint_rows(survivor_position[start:stop] / length_scale)
                projected_rows = np.einsum("ijk,k->ij", rows, direction)
                scalar_correction = weights[start:stop] * (projected_rows @ multipliers)
                moment_corrected[start:stop] += scalar_correction[:, None] * direction
            return validate(
                survivor_position,
                survivor_vortex_strength,
                removed_position,
                removed_vortex_strength,
                moment_corrected,
                scope=scope,
                record_diagnostics=record_diagnostics,
                correction_label=correction_label,
            )

        multipliers, _residual, rank, singular_values = np.linalg.lstsq(
            gram,
            right_hand_side,
            rcond=_GBD_MOMENT_RCOND,
        )
        condition = (
            float("inf")
            if len(singular_values) == 0 or singular_values[-1] <= 0.0
            else float(singular_values[0] / singular_values[-1])
        )
        if rank < 9 or not np.isfinite(condition) or condition > _GBD_MOMENT_CONDITION_LIMIT:
            raise RuntimeError(
                "GBD moment correction constraints are rank-deficient or ill-conditioned "
                f"for {scope}: rank={rank}/9, condition={condition:.6e}, "
                f"limit={_GBD_MOMENT_CONDITION_LIMIT:.6e}"
            )
        if not np.all(np.isfinite(multipliers)):
            raise RuntimeError(f"GBD moment correction produced non-finite multipliers for {scope}")

        moment_corrected = corrected.copy()
        for start in range(0, len(survivor_position), _GBD_MOMENT_CHUNK_SIZE):
            stop = min(start + _GBD_MOMENT_CHUNK_SIZE, len(survivor_position))
            rows = moment_constraint_rows(survivor_position[start:stop] / length_scale)
            moment_corrected[start:stop] += np.einsum(
                "i,ijk,j->ik",
                weights[start:stop],
                rows,
                multipliers,
            )
        return validate(
            survivor_position,
            survivor_vortex_strength,
            removed_position,
            removed_vortex_strength,
            moment_corrected,
            scope=scope,
            record_diagnostics=record_diagnostics,
            correction_label=correction_label,
        )

    if labels is None:
        return redistribute(
            retained_position,
            retained_vortex_strength,
            discarded_position,
            discarded_vortex_strength,
            scope="the complete cloud",
            record_diagnostics=True,
        )

    survivor_labels = labels[ix, iy, iz]
    discarded_labels = labels[discarded_i, discarded_j, discarded_k]
    unique_labels = np.union1d(survivor_labels, discarded_labels)
    if len(unique_labels) == 1:
        return redistribute(
            retained_position,
            retained_vortex_strength,
            discarded_position,
            discarded_vortex_strength,
            scope=f"group {unique_labels[0]!r}",
            record_diagnostics=True,
            correction_label=int(unique_labels[0]),
        )
    corrected = np.zeros_like(retained_vortex_strength)
    for label in unique_labels:
        survivor_selection = survivor_labels == label
        discarded_selection = discarded_labels == label
        if not np.any(discarded_selection):
            corrected[survivor_selection] = retained_vortex_strength[survivor_selection]
            continue
        if np.count_nonzero(survivor_selection) < 4:
            if strict_labels:
                raise RuntimeError(
                    "GBD weak-strength removal has insufficient support in a fluid component; "
                    "increase the regeneration cap or reduce the removal threshold."
                )
            return redistribute(
                retained_position,
                retained_vortex_strength,
                discarded_position,
                discarded_vortex_strength,
                scope=f"the complete cloud (sparse group {label!r})",
                record_diagnostics=True,
            )
        corrected[survivor_selection] = redistribute(
            retained_position[survivor_selection],
            retained_vortex_strength[survivor_selection],
            discarded_position[discarded_selection],
            discarded_vortex_strength[discarded_selection],
            scope=f"group {label!r}",
            correction_label=int(label),
        )
    return validate(
        retained_position,
        retained_vortex_strength,
        discarded_position,
        discarded_vortex_strength,
        corrected,
        scope="the complete cloud after the per-group correction",
        record_diagnostics=True,
    )
