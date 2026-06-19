import gmsh
import numpy as np


def _find_mesh_dimension(model) -> int:
    """Return the highest mesh dimension containing elements."""
    dims = [d for d in range(4) if len(model.mesh.getElements(dim=d)[0]) > 0]
    return max(dims) if dims else 3


def _register_face_nodes(
    nodes_slice, face_map: dict, face_nodes_map: dict, node_tag_to_idx: dict, cell_idx: int
) -> None:
    """Register a single face into the face map."""
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
    """Register all faces of a given face_type (3=tri, 4=quad) for a cell group."""
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
    """Extract all faces from cells and build face→owner-cells map."""
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
    """Collect boundary patch definitions from physical groups."""
    patch_info: list[dict] = []
    boundary_faces_all: set = set()
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
                        patch_face_keys.append(key)
                        boundary_faces_all.add(key)
        patch_info.append({"name": name, "keys": patch_face_keys})
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
        if not gmsh.isInitialized():
            gmsh.initialize()

    def finalize(self):
        if gmsh.isInitialized():
            gmsh.finalize()

    def load_mesh(self, filename: str):
        gmsh.open(filename)

    def get_mesh_data(self) -> dict:
        """Extracts mesh data from the current Gmsh model."""
        model = gmsh.model

        # 1. Get Nodes
        node_tags, coords, _ = model.mesh.getNodes()
        node_tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}
        points = coords.reshape(-1, 3)

        # 2. Get Elements
        max_dim = _find_mesh_dimension(model)
        cell_types, cell_tags_list, _ = model.mesh.getElements(dim=max_dim)

        all_cell_tags: list = []
        for tags in cell_tags_list:
            all_cell_tags.extend(tags)
        n_cells = len(all_cell_tags)
        cell_tag_to_idx = {tag: i for i, tag in enumerate(all_cell_tags)}

        # 3. Extract Faces
        face_map, face_nodes_map = _build_face_map_from_cells(
            model, cell_types, cell_tags_list, node_tag_to_idx, cell_tag_to_idx
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
                }
            )

        return {
            "points": points,
            "faces": final_faces,
            "owners": np.array(owners, dtype=np.int32),
            "neighbours": np.array(neighbours, dtype=np.int32),
            "boundary": boundary_patches,
            "n_points": len(node_tags),
            "n_faces": len(final_faces),
            "n_interior_faces": n_internal,
            "n_elements": n_cells,
        }


if __name__ == "__main__":
    # Simple test if run directly
    importer = GmshImporter()
    # Need a .msh file to test
    importer.finalize()
