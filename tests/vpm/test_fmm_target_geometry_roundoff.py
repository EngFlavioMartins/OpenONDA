"""Separate exact geometry transport from nondeterministic CUDA list ordering.

No backend methods are wrapped. Atomic interaction-list reservations already
make independent original rebuilds non-bitwise on CUDA. We measure that control
spread, compare both paths to the same independent finite-image pair oracle,
and separately require every saved/restored geometry component to be bitwise.
The 16-epsilon absolute-conditioning allowance is the existing target-operator
qualification budget, not a relative tolerance inflated at cancelled values.
WINCKELMANS exercises reflected FMM targets; Gaussian slab fields have separate
finite mesh qualifications.
"""

import json

import numpy as np
import pytest
import taichi as ti

from tests.vpm._fmm_geometry_harness import Harness
from tests.vpm.test_fmm_targets import _slab_oracle


@pytest.fixture(scope="module", autouse=True)
def runtime():
    owns_runtime = ti.lang.impl.get_runtime().prog is None
    if owns_runtime:
        ti.init(arch=ti.cpu, cpu_max_num_threads=2, offline_cache=False)
    yield
    if owns_runtime:
        ti.reset()


def _geometry_snapshot(target, count):
    """Exactly the downstream geometry, including reconstructed ancestry."""
    leaves = int(target.leaf_count[None])
    fields = {
        "position": (target.tree.position, count),
        "sorted_indices": (target.tree.sorted_indices, count),
        "leaf_depth": (target.tree.node_depth, count),
        "leaf_nodes": (target.leaf_nodes, leaves),
        "centre": (target.tree.node_centre, 2 * count - 1),
        "half_size": (target.tree.node_half_size, 2 * count - 1),
        "particle_count": (target.tree.node_particle_count, 2 * count - 1),
        "left": (target.tree.node_left, 2 * count - 1),
        "right": (target.tree.node_right, 2 * count - 1),
        "parent": (target.tree.node_parent, 2 * count - 1),
        "bound_radius": (target.node_bound_radius, 2 * count - 1),
        "path_length": (target.target_path_length, count),
    }
    result = {name: field.to_numpy()[:length].copy() for name, (field, length) in fields.items()}
    paths = target.target_path.to_numpy()
    for slot, length in enumerate(result["path_length"]):
        result[f"path_{slot}"] = paths[:length, slot].copy()
    result["counts"] = np.array(
        [
            target._prepared_count,
            int(target.tree.n_particles_total[None]),
            leaves,
            int(target.target_path_error[None]),
        ]
    )
    return result


def test_all_geometry_and_paths_restore_bitwise_across_interleaved_remainder_tiles():
    h = Harness("WINCKELMANS")
    try:
        h.run()
        base = h.slab.base
        cache, target = base._image_geometry_cache, base._target_workspace
        with cache.scope(h.x, h.n, 4):
            originals = {}
            for start, count in ((0, 4), (8, 3), (4, 4)):
                cache.prepare(target, h.x, count, target_start=start)
                originals[start] = _geometry_snapshot(target, count)
            before = base.diagnostics.image_target_geometry_restores
            for start, count in ((8, 3), (0, 4), (4, 4), (8, 3)):
                cache.prepare(target, h.x, count, target_start=start)
                actual = _geometry_snapshot(target, count)
                for name, expected in originals[start].items():
                    np.testing.assert_array_equal(
                        actual[name], expected, err_msg=f"tile={start}, {name}"
                    )
            assert base.diagnostics.image_target_geometry_restores - before == 4
    finally:
        h.slab.base.close()


def _norm(value):
    return float(np.linalg.norm(np.asarray(value, dtype=np.float64)))


def field_repeat_evidence(h, kernel, controls, candidates):
    """Validate measured direct envelopes and return compact diagnostic data."""
    reference_tail = controls[0][1]
    for _, tail in (*controls, *candidates):
        for key in ("shell", "block_start", "target_batches"):
            assert tail[key] == reference_tail[key], (key, tail, reference_tail)
        assert tail["relative"] <= h.slab.tail_tolerance
    gamma = h.gamma.to_numpy().astype(np.float64)
    exact, conditioning = _slab_oracle(
        kernel,
        h.positions.astype(np.float64),
        gamma,
        h.radius.to_numpy().astype(np.float64),
        h.positions.astype(np.float64),
        reference_tail["shell"],
        stage=True,
    )
    assert h.slab.stretching_scheme == "TRANSPOSED"
    exact.append(np.einsum("nji,nj->ni", exact[1], gamma))
    conditioning.append(conditioning[1] * np.linalg.norm(gamma, axis=1))
    evidence = {"kernel": kernel, "arch": str(ti.lang.impl.current_cfg().arch), "fields": {}}
    for index, name in enumerate(("velocity", "gradient", "strength_rate")):
        control_errors = [_norm(fields[index] - exact[index]) for fields, _ in controls]
        candidate_errors = [_norm(fields[index] - exact[index]) for fields, _ in candidates]
        control_spread = max(_norm(a[0][index] - b[0][index]) for a in controls for b in controls)
        candidate_delta = max(
            _norm(a[0][index] - b[0][index]) for a in controls for b in candidates
        )
        allowance = float(16 * np.finfo(np.float32).eps * _norm(conditioning[index]))
        assert max(candidate_errors) <= min(control_errors) + allowance
        assert candidate_delta <= control_spread + allowance
        evidence["fields"][name] = {
            "control_direct_errors": control_errors,
            "candidate_direct_errors": candidate_errors,
            "control_repeat_spread": control_spread,
            "candidate_control_delta": candidate_delta,
            "absolute_conditioning_allowance": allowance,
        }
    # Norm maxima are Lipschitz in their input fields; the same absolute
    # conditioning budget bounds scalar tail differences without shifting or
    # relaxing the actual convergence gate. Total conditioning overbounds the
    # last block conservatively and is also recorded above for inspection.
    tail_allowance = [16 * np.finfo(np.float32).eps * np.max(c) for c in conditioning[:2]]
    for index, name in enumerate(("velocity", "gradient")):
        values = [tail[name] for _, tail in (*controls, *candidates)]
        assert max(values) - min(values) <= tail_allowance[index]
    evidence["tails"] = [tail for _, tail in (*controls, *candidates)]
    return evidence


@pytest.mark.parametrize("kernel", ["WINCKELMANS"])
def test_repeated_rebuild_and_bank_fields_remain_inside_same_direct_error_envelope(kernel):
    h = Harness(kernel)
    try:
        h.slab.base.max_image_geometry_bytes = 0
        controls = [h.run() for _ in range(3)]
        h.slab.base.max_image_geometry_bytes = 64 * 1024 * 1024
        candidates = [h.run() for _ in range(3)]
        evidence = field_repeat_evidence(h, kernel, controls, candidates)
        print("GEOMETRY_ROUNDOFF " + json.dumps(evidence, sort_keys=True))
    finally:
        h.slab.base.close()
