"""Typed layouts of the native FVM mesh, boundary patch and geometry dictionaries.

The mesher and importers build these dictionaries; the solver, assembly and
output layers consume them.  Index arrays keep the integer width chosen by the
producer (``int32`` for topology, ``int64`` for global identifiers), so they are
typed as plain ``np.ndarray``.  Geometric quantities are always ``float64``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NotRequired, TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from scipy import sparse

    from ..core.parallel import ParallelContext
    from ..fields.boundary_reconstruction import BoundaryStencilCache
    from .partition import CellPartition

FloatArray: TypeAlias = npt.NDArray[np.float64]

# Face-to-vertex connectivity: a rectangular array for fixed-width faces or one
# index array per face for polyhedral meshes.
FaceVertices: TypeAlias = np.ndarray | list[np.ndarray]


class BoundaryPatchData(TypedDict, total=False):
    """One boundary patch of a native mesh, with its boundary conditions.

    The mesh layer sets the patch range and type; the solver adds the velocity
    and pressure boundary conditions and the coupled-patch pairing.
    """

    name: str
    start_face: int
    n_faces: int
    type: str
    physical_tag: int | None
    source_dimension: int | None
    neighbour_patch: str
    transform: str

    # Velocity boundary condition.
    velocity_type: str
    velocity_value: FloatArray
    velocity_value_field: FloatArray
    normal_velocity_field: FloatArray
    tangential_gradient_field: FloatArray
    max_removed_tangential_gradient_normal_component: float
    boundary_condition_type: str
    value: Any  # scalar component or vector value of a per-component copy

    # Pressure boundary condition.
    pressure_type: str
    kinematic_pressure_value: float
    kinematic_pressure_value_field: FloatArray
    fixed_flux_pressure_delta: FloatArray
    fixed_flux_pressure_external: bool
    fixed_gradient_delta: FloatArray
    external_face_flux: FloatArray

    # Freestream outflow masks and coupled-patch pairing filled by the solver.
    _freestream_outflow: npt.NDArray[np.bool_]
    _fixed_freestream_outflow: npt.NDArray[np.bool_]
    _directional_fixed_flux_pressure: bool
    _paired_cells: np.ndarray
    _paired_faces: np.ndarray


class MeshData(TypedDict):
    """Native face-based FVM mesh.

    Interior faces occupy ``[0, n_interior_faces)``; boundary faces follow,
    grouped by patch.  Vertex coordinates are in mesh length units.
    """

    vertex_position: FloatArray
    faces: FaceVertices
    owners: np.ndarray
    neighbours: np.ndarray
    boundary: list[BoundaryPatchData]
    n_cells: int
    n_faces: int
    n_interior_faces: int
    n_points: NotRequired[int]

    # Cell-to-face CSR connectivity and cell descriptions.
    cell_face_indices: NotRequired[np.ndarray]
    cell_face_offset: NotRequired[np.ndarray]
    cell_face_reversed: NotRequired[npt.NDArray[np.bool_]]
    cell_vertex_indices: NotRequired[np.ndarray]
    cell_type_code: NotRequired[np.ndarray]
    cell_order: NotRequired[np.ndarray]
    cell_families: NotRequired[np.ndarray]
    cell_levels: NotRequired[np.ndarray]
    cell_sizes: NotRequired[np.ndarray]
    boundary_layer_index: NotRequired[np.ndarray]

    # Global and source numbering.
    global_cell_id: NotRequired[np.ndarray]
    global_face_id: NotRequired[np.ndarray]
    global_point_ids: NotRequired[np.ndarray]
    source_cell_id: NotRequired[np.ndarray]
    source_point_ids: NotRequired[np.ndarray]

    # Coupled (cyclic) boundary pairing.
    boundary_neighbour_cell: NotRequired[np.ndarray]
    boundary_pair_face: NotRequired[np.ndarray]

    # Mesh generation record and cache keys (JSON-like metadata).
    mesh_generation: NotRequired[dict[str, Any]]
    meshing_input_hash: NotRequired[str]
    global_mesh_hash: NotRequired[str]

    # Partitioned (local rank) mesh.
    global_boundary_names: NotRequired[tuple[str, ...]]
    global_boundary_ranges: NotRequired[dict[str, tuple[int, int]]]
    partition: NotRequired[CellPartition]
    _n_owned: NotRequired[int]
    _visualization_mesh: NotRequired[MeshData]
    _parallel_context: NotRequired[ParallelContext]

    # cfMesh workflow state carried between meshing stages.
    _cfmesh_cell_face_order: NotRequired[list[list[int]]]
    _cfmesh_octree_leaves: NotRequired[np.ndarray]
    _cfmesh_boundary_column_inner: NotRequired[dict[str, Any]]

    # Sparse-matrix patterns cached on the static mesh, keyed by whether
    # boundary couplings are included.
    _fvm_csr_patterns: NotRequired[dict[bool, Any]]


class GeometryData(TypedDict):
    """Cell and face geometry derived from a :class:`MeshData`.

    Lengths are in mesh units, areas in units², volumes in units³.
    """

    face_centre: FloatArray
    face_area_vector: FloatArray
    face_area: FloatArray
    cell_centre: FloatArray
    cell_volume: FloatArray
    face_interpolation_weight: FloatArray
    cell_connection_vector: FloatArray
    wall_distance: FloatArray

    gradient_scheme: NotRequired[str]
    _operator_backend: NotRequired[str]
    boundary_translation: NotRequired[FloatArray]

    # Least-squares gradient stencils.
    least_squares_neighbour_field_index: NotRequired[np.ndarray]
    least_squares_owner_cell: NotRequired[np.ndarray]
    least_squares_neighbour_weighted_displacement: NotRequired[FloatArray]
    least_squares_normal_matrix_inverse: NotRequired[FloatArray]
    least_squares_gradient_operator: NotRequired[sparse.csr_matrix]
    lsq_condition: NotRequired[FloatArray]
    lsq_rank: NotRequired[np.ndarray]
    lsq_solver_method: NotRequired[np.ndarray]

    # Boundary-owner least-squares stencils cached by the boundary reconstruction.
    _boundary_owner_lsq_with_ghosts: NotRequired[BoundaryStencilCache]
    _boundary_owner_lsq_real: NotRequired[BoundaryStencilCache]


class LeastSquaresGeometry(TypedDict):
    """Least-squares gradient stencils added to a :class:`GeometryData`."""

    least_squares_neighbour_field_index: np.ndarray
    least_squares_owner_cell: np.ndarray
    least_squares_neighbour_weighted_displacement: FloatArray
    least_squares_normal_matrix_inverse: FloatArray
    least_squares_gradient_operator: sparse.csr_matrix
    lsq_condition: FloatArray
    lsq_rank: np.ndarray
    lsq_solver_method: np.ndarray
    gradient_scheme: str
