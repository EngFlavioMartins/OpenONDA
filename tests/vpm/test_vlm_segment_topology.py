import numpy as np

from source.solvers.vpm.boundary_elements.vlm.solver.mesh import _stitch_segment_neighbors


def test_stitch_segment_neighbors_connects_openvsp_style_span_segments():
    """Consecutive one-span OpenVSP segments must not become artificial tips."""
    neigh = np.full((12, 4), -1, dtype=np.int32)
    blocks = [
        {"wing_id": 0, "segment_order": 0, "is_mirrored": False, "start": 0, "nc": 3, "ns": 1},
        {"wing_id": 0, "segment_order": 1, "is_mirrored": False, "start": 3, "nc": 3, "ns": 1},
        {"wing_id": 0, "segment_order": 2, "is_mirrored": False, "start": 6, "nc": 3, "ns": 1},
        {"wing_id": 1, "segment_order": 0, "is_mirrored": False, "start": 9, "nc": 3, "ns": 1},
    ]

    _stitch_segment_neighbors(blocks, neigh)

    for chord_idx in range(3):
        assert neigh[chord_idx, 1] == 3 + chord_idx
        assert neigh[3 + chord_idx, 0] == chord_idx
        assert neigh[3 + chord_idx, 1] == 6 + chord_idx
        assert neigh[6 + chord_idx, 0] == 3 + chord_idx

    assert np.all(neigh[9:12, 0:2] == -1)
