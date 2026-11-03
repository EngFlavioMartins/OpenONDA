"""A wall vertex is tied only to feature facets within reach of its own faces."""

import numpy as np

from openonda.fvm.mesh.cartesian.mesher import _planar_feature_constraints


def _pitched_thin_slab():
    """NACA 0018 slab with a flat base at 0.95 chord, pitched 17 and tilted 7 degrees.

    Chord 0.35 m, span 0.2 m: the case 06 blade tip, whose 7.4 mm base is
    thinner than the 0.025 m wall faces of its mesh.
    """
    chord, span, thickness, cut, n_points = 0.35, 0.2, 0.18, 0.95, 61
    x = cut * 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_points)))
    half = (
        5.0
        * thickness
        * chord
        * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    )
    lower = np.column_stack((x * chord - 0.5 * chord, -half))
    upper = np.column_stack((x * chord - 0.5 * chord, half))
    ring = np.vstack((lower, upper[:0:-1]))
    n_outline = len(ring)
    vertices = np.vstack(
        [np.column_stack((ring, np.full(n_outline, z))) for z in (-0.5 * span, 0.5 * span)]
    )
    triangles = []
    for i in range(n_outline):
        j = (i + 1) % n_outline
        triangles += [(i, j, n_outline + j), (i, n_outline + j, n_outline + i)]
    cap = [(i, i + 1, n_outline - i - 1) for i in range(n_points - 1)]
    cap += [(i, n_outline - i - 1, n_outline - i) for i in range(1, n_points - 1)]
    triangles += [(n_outline + a, n_outline + b, n_outline + c) for a, b, c in cap]
    triangles += [(a, c, b) for a, b, c in cap]
    pitch, tilt = np.radians(17.0), np.radians(7.0)
    about_z = np.array(
        [[np.cos(pitch), -np.sin(pitch), 0.0], [np.sin(pitch), np.cos(pitch), 0.0], [0, 0, 1]]
    )
    about_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(tilt), -np.sin(tilt)], [0.0, np.sin(tilt), np.cos(tilt)]]
    )
    faces = (vertices @ (about_x @ about_z).T)[np.asarray(triangles)]
    volume = np.einsum("ij,ij->i", faces[:, 0], np.cross(faces[:, 1], faces[:, 2])).sum()
    return faces if volume > 0.0 else faces[:, ::-1]


# Wall vertices and faces of the cfMesh wrapper at the slab's trailing edge beside its lower
# cap, as optimised before the wall projection: projecting them onto that cap inverted a cell
WALL_POINTS = np.array(
    [
        (0.11233975446338004, 0.027754195167761442, -0.08348254904777042),
        (0.12619679084938717, 0.03778373312229385, -0.07953712462792262),
        (0.11602163791758857, 0.04533430710518369, -0.0958374093728332),
        (0.1405836661469792, 0.044203184933138503, -0.09276992445332677),
        (0.1541557308174798, 0.053470551643647564, -0.08113956355113307),
        (0.11387376229914313, 0.05905684311987009, -0.08434638331881457),
        (0.1323773906943309, 0.0630090704179033, -0.08313365614978536),
        (0.14538145309046965, 0.06351729295265827, -0.07654862451783039),
        (0.1293618627272693, 0.038866401227847805, -0.06334633970521607),
        (0.13881636870943573, 0.04538124689715158, -0.07268265096231157),
        (0.14833008717665574, 0.046374229823158065, -0.06299455414665002),
        (0.1131897877241925, 0.05370390039881026, -0.06627996274031304),
        (0.12911427032423856, 0.05504482306930573, -0.0652722141274995),
        (0.14233876028507042, 0.059364382987174696, -0.062385087608768076),
    ]
)
WALL_FACES = [
    [0, 1, 3, 2],
    [2, 3, 6, 5],
    [3, 4, 7, 6],
    [1, 8, 9, 3],
    [3, 9, 10, 4],
    [5, 6, 12, 11],
    [4, 10, 13, 7],
    [6, 7, 13, 12],
]


def test_cap_narrower_than_a_wall_face_does_not_capture_its_vertices():
    triangles = _pitched_thin_slab()
    constraints = _planar_feature_constraints(
        WALL_POINTS, [np.asarray(face) for face in WALL_FACES], triangles
    )
    for point_id, planes in constraints.items():
        point = WALL_POINTS[point_id]
        smallest_face = min(
            np.linalg.norm(
                WALL_POINTS[face][:, None, :] - WALL_POINTS[face][None, :, :], axis=2
            ).max()
            for face in WALL_FACES
            if point_id in face
        )
        for _normal, _offset, index in planes:
            assert index.nearest_point(point)[1] <= 0.5 * smallest_face
