import numpy as np

try:
    import gmsh
except ImportError:  # Optional FVM mesh dependency; checked when the importer is used.
    gmsh = None


SUPPORTED_GMSH_CELL_TYPES = {
    4: "tetrahedron",
    5: "hexahedron",
    6: "prism",
    7: "pyramid",
}


def _find_mesh_dimension(model) -> int:
    """Return the highest mesh dimension containing elements.

    Args:
        model: A Gmsh model instance with a populated mesh.

    Returns:
        The highest dimension (0–3) that has at least one element.
        Defaults to 3 if no elements are found.
    """
    dims = [d for d in range(4) if len(model.mesh.getElements(dim=d)[0]) > 0]
    return max(dims) if dims else 3


def _hexahedral_cell_vertices(cell_types, cell_node_tags_list, node_tag_to_idx: dict):
    """Return per-cell vertex indices for a uniformly hexahedral mesh.

    Recombined Gmsh meshes are frequently pure hexahedra.  Recording the
    explicit cell corners lets the VTK exporter emit native
    ``VTK_HEXAHEDRON`` cells instead of falling back to the general
    ``VTK_POLYHEDRON`` representation, which ParaView interpolates and
    renders far better.  Gmsh's 8-node hexahedron uses the same corner
    ordering as VTK, so the connectivity is copied without permutation.

    Args:
        cell_types:          Gmsh element type codes returned by ``getElements``.
        cell_node_tags_list: Flat node-tag arrays, one per element type.
        node_tag_to_idx:     Mapping from Gmsh node tag to point index.

    Returns:
        An ``(n_cells, 8)`` array of point indices, or ``None`` when the
        mesh is not uniformly hexahedral.
    """
    if len(cell_types) != 1 or int(cell_types[0]) != 5:
        return None
    node_tags = np.asarray(cell_node_tags_list[0])
    if node_tags.size % 8:
        return None
    lookup = np.vectorize(node_tag_to_idx.__getitem__, otypes=[np.int64])
    return np.ascontiguousarray(lookup(node_tags).reshape(-1, 8), dtype=np.int32)


def _register_face_nodes(
    nodes_slice, face_map: dict, face_nodes_map: dict, node_tag_to_idx: dict, cell_idx: int
) -> None:
    """Register a single face into the face map and face_nodes_map.

    Uses sorted node indices as a canonical key so that shared faces
    are recognised across adjacent cells.

    Args:
        nodes_slice: Sequence of Gmsh node tags defining the face.
        face_map: Mapping from sorted-node-key to list of cell indices.
        face_nodes_map: Mapping from sorted-node-key to original node order.
        node_tag_to_idx: Mapping from Gmsh node tag to local index.
        cell_idx: Index of the cell that owns this face instance.
    """
    f_node_indices = tuple(node_tag_to_idx[tag] for tag in nodes_slice)
    key = tuple(sorted(f_node_indices))
    if key not in face_map:
        face_map[key] = []
        face_nodes_map[key] = list(f_node_indices)
    face_map[key].append(cell_idx)


def _process_cell_faces_by_type(
    model,
    elem_type: int,
    cell_tags,
    face_map: dict,
    face_nodes_map: dict,
    node_tag_to_idx: dict,
    cell_tag_to_idx: dict,
    face_type: int,
) -> None:
    """Register all faces of a given face_type for a cell group.

    Handles both triangular (face_type=3) and quadrilateral (face_type=4)
    faces by delegating each individual face to _register_face_nodes.

    Args:
        model: A Gmsh model instance.
        elem_type: Gmsh element type code for the cell group.
        cell_tags: Iterable of Gmsh element tags for this group.
        face_map: Mapping from sorted-node-key to list of cell indices.
        face_nodes_map: Mapping from sorted-node-key to original node order.
        node_tag_to_idx: Mapping from Gmsh node tag to local index.
        cell_tag_to_idx: Mapping from Gmsh element tag to local cell index.
        face_type: Number of nodes per face (3 for tri, 4 for quad).
    """
    face_nodes = model.mesh.getElementFaceNodes(elem_type, face_type)
    if len(face_nodes) == 0:
        return
    num_nodes_per_face = face_type
    num_faces_per_elem = len(face_nodes) // (len(cell_tags) * num_nodes_per_face)
    for j, cell_tag in enumerate(cell_tags):
        cell_idx = cell_tag_to_idx[cell_tag]
        start = j * num_faces_per_elem * num_nodes_per_face
        for f in range(num_faces_per_elem):
            f_start = start + f * num_nodes_per_face
            _register_face_nodes(
                face_nodes[f_start : f_start + num_nodes_per_face],
                face_map,
                face_nodes_map,
                node_tag_to_idx,
                cell_idx,
            )


def _build_face_map_from_cells(
    model, cell_types, cell_tags_list, node_tag_to_idx: dict, cell_tag_to_idx: dict
) -> tuple[dict, dict]:
    """Extract all faces from cells and build face→owner-cells map.

    Iterates over every cell type present in the mesh and, for each,
    extracts triangular and quadrilateral faces.  The result is a pair of
    dictionaries that map a sorted node-key to (i) the list of cell indices
    sharing that face, and (ii) the face nodes in their original order.

    Args:
        model: A Gmsh model instance with a populated mesh.
        cell_types: Array of Gmsh element-type codes.
        cell_tags_list: List of tag arrays, one per cell type.
        node_tag_to_idx: Mapping from Gmsh node tag to local index.
        cell_tag_to_idx: Mapping from Gmsh element tag to local cell index.

    Returns:
        Tuple of (face_map, face_nodes_map).
        - face_map: dict[sorted_node_key, list[cell_idx]]
        - face_nodes_map: dict[sorted_node_key, list[node_idx]]
    """
    face_map: dict[tuple[int, ...], list[int]] = {}
    face_nodes_map: dict[tuple[int, ...], list[int]] = {}
    for i, elem_type in enumerate(cell_types):
        for face_type in [3, 4]:
            _process_cell_faces_by_type(
                model,
                elem_type,
                cell_tags_list[i],
                face_map,
                face_nodes_map,
                node_tag_to_idx,
                cell_tag_to_idx,
                face_type,
            )
    return face_map, face_nodes_map


def _collect_boundary_patches(
    model, physical_groups, face_map: dict, node_tag_to_idx: dict, max_dim: int
) -> tuple[list, set]:
    """Collect boundary patch definitions from physical groups.

    For each physical group of dimension max_dim-1, finds all faces that
    belong to the group and records them as a named patch.  Faces that
    appear in a physical group are also added to the boundary-faces set
    so they can be distinguished from internal faces later.

    Args:
        model: A Gmsh model instance.
        physical_groups: Sequence of (dim, tag) tuples from
            model.getPhysicalGroups().
        face_map: Mapping from sorted-node-key to list of cell indices.
        node_tag_to_idx: Mapping from Gmsh node tag to local index.
        max_dim: The highest mesh dimension (used to infer boundary dim).

    Returns:
        Tuple of (patch_info, boundary_faces_all).
        - patch_info: list of dicts with keys "name" and "keys".
        - boundary_faces_all: set of face keys that belong to any patch.
    """
    patch_info: list[dict] = []
    boundary_faces_all: set = set()
    face_patch: dict[tuple[int, ...], str] = {}
    for dim, tag in physical_groups:
        name = model.getPhysicalName(dim, tag)
        entities = model.getEntitiesForPhysicalGroup(dim, tag)
        patch_face_keys: list = []
        for entity in entities:
            elem_types, elem_tags_list, elem_node_tags_list = model.mesh.getElements(
                dim, tag=entity
            )
            for i_type in range(len(elem_types)):
                tags = elem_node_tags_list[i_type]
                elem_tags = elem_tags_list[i_type]
                num_nodes = len(tags) // len(elem_tags)
                for k in range(len(elem_tags)):
                    f_nodes = tags[k * num_nodes : (k + 1) * num_nodes]
                    key = tuple(sorted(node_tag_to_idx[t] for t in f_nodes))
                    if key in face_map:
                        previous = face_patch.get(key)
                        if previous is not None and previous != name:
                            raise ValueError(
                                "Gmsh boundary face belongs to multiple physical groups: "
                                f"{previous!r} and {name!r}"
                            )
                        face_patch[key] = name
                        patch_face_keys.append(key)
                        boundary_faces_all.add(key)
        patch_info.append(
            {"name": name, "keys": patch_face_keys, "physical_tag": int(tag), "dimension": dim}
        )
    return patch_info, boundary_faces_all


class GmshImporter:
    """
    Imports a Gmsh mesh and converts it to OpenONDA FVM mesh_data format.
    Strictly follows OpenFOAM topological conventions:
    - Internal faces come first.
    - Boundary faces follow, grouped by patch.
    - Owner < Neighbour for internal faces.
    """

    def __init__(self):
        """Initialise the importer and start the Gmsh API.

        Gmsh is lazily initialised if it has not been started yet.  Call
        finalize() to shut down the Gmsh kernel when the importer is no
        longer needed.
        """
        if gmsh is None:
            raise ImportError(
                "Gmsh support requires the optional FVM dependencies: pip install 'OpenONDA[fvm]'"
            )
        if not gmsh.isInitialized():
            gmsh.initialize()
        self.source_path = None

    def finalize(self):
        """Finalize the Gmsh API.

        Safe to call multiple times; only the first call shuts down the
        Gmsh kernel.
        """
        if gmsh is not None and gmsh.isInitialized():
            gmsh.finalize()

    def load_mesh(self, filename: str):
        """Open and load a Gmsh mesh file.

        Args:
            filename: Path to a ``.msh`` file (or other format supported
                by Gmsh).

        Raises:
            FileNotFoundError: If the file does not exist.
            gmsh.GmshException: If Gmsh cannot parse the file.
        """
        gmsh.open(filename)
        self.source_path = str(filename)

    def get_mesh_data(self) -> dict:
        """Extract mesh data from the current Gmsh model.

        Nodes, cells, and boundary patches are collected from the Gmsh
        model that was previously loaded via load_mesh().  The returned
        dictionary follows the OpenONDA FVM mesh_data convention with
        internal faces first, owners/neighbours arrays, and a list of
        boundary patch descriptors.

        Returns:
            Dictionary with the following keys:
            - points:         ndarray of shape (n_points, 3).
            - faces:          list of ndarray face-node indices.
            - owners:         ndarray of owner cell indices.
            - neighbours:     ndarray of neighbour cell indices.
            - boundary:       list of patch dicts (name, startFace,
                              nFaces, type).
            - n_points:       int.
            - n_faces:        int.
            - n_interior_faces: int.
            - n_elements:     int.
        """
        model = gmsh.model

        # 1. Get Nodes
        node_tags, coords, _ = model.mesh.getNodes()
        node_tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}
        points = coords.reshape(-1, 3)

        # 2. Get Elements
        max_dim = _find_mesh_dimension(model)
        if max_dim != 3:
            raise ValueError(
                f"Unsupported Gmsh mesh dimension {max_dim}; the FVM importer requires "
                "first-order three-dimensional volume cells"
            )
        cell_types, cell_tags_list, cell_node_tags_list = model.mesh.getElements(dim=max_dim)

        cell_type_codes = []
        cell_families = []
        cell_orders = []
        for element_type, tags in zip(cell_types, cell_tags_list, strict=True):
            name, _dim, order, _n_nodes, _local, _n_primary = model.mesh.getElementProperties(
                element_type
            )
            if int(order) != 1:
                raise ValueError(
                    f"Unsupported high-order Gmsh element {name!r} (type {element_type}, "
                    f"order {order}); the FVM importer accepts first-order cells only"
                )
            if int(element_type) not in SUPPORTED_GMSH_CELL_TYPES:
                raise ValueError(
                    f"Unsupported Gmsh volume element {name!r} (type {element_type}); "
                    "supported first-order cell types: tetrahedron (4), hexahedron (5), "
                    "prism (6), pyramid (7)"
                )
            cell_type_codes.extend([int(element_type)] * len(tags))
            cell_families.extend([str(name)] * len(tags))
            cell_orders.extend([int(order)] * len(tags))

        all_cell_tags: list = []
        for tags in cell_tags_list:
            all_cell_tags.extend(tags)
        n_cells = len(all_cell_tags)
        cell_tag_to_idx = {tag: i for i, tag in enumerate(all_cell_tags)}

        # 3. Extract Faces
        face_map, face_nodes_map = _build_face_map_from_cells(
            model, cell_types, cell_tags_list, node_tag_to_idx, cell_tag_to_idx
        )
        non_manifold = [(key, cells) for key, cells in face_map.items() if len(cells) > 2]
        if non_manifold:
            key, cells = non_manifold[0]
            raise ValueError(
                f"Non-manifold Gmsh face with source nodes {key} is shared by "
                f"{len(cells)} cells: {cells}"
            )

        # 4. Identify Boundary Patches
        physical_groups = model.getPhysicalGroups(dim=max_dim - 1)
        patch_info, boundary_faces_all = _collect_boundary_patches(
            model, physical_groups, face_map, node_tag_to_idx, max_dim
        )

        orphan_boundary_keys = [
            k for k, v in face_map.items() if len(v) == 1 and k not in boundary_faces_all
        ]
        if orphan_boundary_keys:
            patch_info.append({"name": "defaultFaces", "keys": orphan_boundary_keys})

        # 5. Assemble final face lists
        internal_face_keys = [k for k, v in face_map.items() if len(v) == 2]
        final_faces: list = []
        owners: list = []
        neighbours: list = []

        for key in internal_face_keys:
            cells = face_map[key]
            owner, neighbour = min(cells[0], cells[1]), max(cells[0], cells[1])
            owners.append(owner)
            neighbours.append(neighbour)
            final_faces.append(np.array(face_nodes_map[key], dtype=np.int32))

        n_internal = len(final_faces)
        boundary_patches: list = []
        current_face_idx = n_internal

        for patch in patch_info:
            start_face = current_face_idx
            n_faces_in_patch = 0
            for key in patch["keys"]:
                owners.append(face_map[key][0])
                final_faces.append(np.array(face_nodes_map[key], dtype=np.int32))
                n_faces_in_patch += 1
                current_face_idx += 1
            boundary_patches.append(
                {
                    "name": patch["name"],
                    "startFace": start_face,
                    "nFaces": n_faces_in_patch,
                    "type": "patch",
                    "physical_tag": patch.get("physical_tag"),
                    "source_dimension": patch.get("dimension"),
                }
            )

        mesh_data = {
            "points": points,
            "faces": final_faces,
            "owners": np.array(owners, dtype=np.int32),
            "neighbours": np.array(neighbours, dtype=np.int32),
            "boundary": boundary_patches,
            "n_points": len(node_tags),
            "n_faces": len(final_faces),
            "n_interior_faces": n_internal,
            "n_elements": n_cells,
            "source_point_ids": np.asarray(node_tags, dtype=np.int64),
            "source_cell_ids": np.asarray(all_cell_tags, dtype=np.int64),
            "cell_type_codes": np.asarray(cell_type_codes, dtype=np.int32),
            "cell_families": np.asarray(cell_families),
            "cell_orders": np.asarray(cell_orders, dtype=np.int8),
            "global_cell_ids": np.asarray(all_cell_tags, dtype=np.int64),
            "global_face_ids": np.arange(len(final_faces), dtype=np.int64),
            "provenance": {
                "format": "gmsh",
                "api_version": getattr(gmsh, "__version__", "unknown"),
                "mesh_file_version": float(gmsh.option.getNumber("Mesh.MshFileVersion")),
                "contract": "gmsh-api-first-order-3d-v1",
                "source": self.source_path,
            },
        }

        cell_vertices = _hexahedral_cell_vertices(cell_types, cell_node_tags_list, node_tag_to_idx)
        if cell_vertices is not None:
            mesh_data["cell_vertices"] = cell_vertices

        return mesh_data


if __name__ == "__main__":
    # Simple test if run directly
    importer = GmshImporter()
    # Need a .msh file to test
    importer.finalize()
