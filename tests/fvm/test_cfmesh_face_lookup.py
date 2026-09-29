"""Compact cfMesh face lookup retains the original exact-key semantics."""

import numpy as np
import pytest

from source.solvers.fvm.mesh.cartesian.cfmesh_template import (
    _matching_face_id_chunks,
    _quad_signature,
)


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize("layout", ["contiguous", "fortran", "strided"])
def test_compact_lookup_preserves_face_ids_and_last_duplicate_across_chunks(dtype, layout):
    rng = np.random.default_rng(71)
    faces = rng.integers(0, 10_000, size=(1537, 4), dtype=dtype)
    faces[600] = faces[9, ::-1]
    faces[1536] = faces[9]
    if layout == "fortran":
        faces = np.asfortranarray(faces)
    elif layout == "strided":
        storage = np.zeros((len(faces), 8), dtype=dtype)
        storage[:, ::2] = faces
        faces = storage[:, ::2]
    lookup = {_quad_signature(face): index for index, face in enumerate(faces)}
    order = rng.permutation(len(faces))
    records = [(_quad_signature(faces[index]), False, ()) for index in order]
    matches = np.concatenate(list(_matching_face_id_chunks(faces, records, chunk_size=23)))
    assert matches.tolist() == [lookup[record[0]] for record in records]
    assert lookup[_quad_signature(faces[9])] == 1536


def test_compact_lookup_rejects_missing_octree_face():
    faces = np.array([[1, 2, 3, 4]], dtype=np.int64)
    records = [((1, 2, 3, 5), False, ())]
    with pytest.raises(RuntimeError, match="absent octree face"):
        list(_matching_face_id_chunks(faces, records))
