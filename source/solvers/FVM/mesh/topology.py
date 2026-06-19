import numpy as np


def compute_element_topology(owners, neighbours, n_elements, n_faces, n_interior_faces, face_nodes):
    """
    Compute element topology (connectivity) from face-based connectivity.

    Args:
        owners (np.ndarray): Owner cell index for each face (0-based).
        neighbours (np.ndarray): Neighbour cell index for interior faces (0-based).
        n_elements (int): Total number of elements (cells).
        n_faces (int): Total number of faces.
        n_interior_faces (int): Number of interior faces.
        face_nodes (list of np.ndarray): List of node indices for each face.

    Returns:
        dict: Dictionary containing topological data:
            - element_faces: List of lists of face indices for each element.
            - element_neighbours: List of lists of neighbour element indices.
            - element_nodes: List of lists of node indices for each element.
            - upper_anb_coeff_index: Array for upper diagonal coefficient indices.
            - lower_anb_coeff_index: Array for lower diagonal coefficient indices.
    """

    # Initialize lists
    element_faces = [[] for _ in range(n_elements)]
    element_neighbours = [[] for _ in range(n_elements)]

    # Process Interior Faces
    for face_idx in range(n_interior_faces):
        own = owners[face_idx]
        nei = neighbours[face_idx]

        # Add neighbour connectivity
        element_neighbours[own].append(nei)
        element_neighbours[nei].append(own)

        # Add face connectivity
        element_faces[own].append(face_idx)
        element_faces[nei].append(face_idx)

    # Process Boundary Faces
    for face_idx in range(n_interior_faces, n_faces):
        own = owners[face_idx]
        element_faces[own].append(face_idx)

    # Compute Element Nodes
    element_nodes = [[] for _ in range(n_elements)]
    for elem_idx in range(n_elements):
        nodes = set()
        for face_idx in element_faces[elem_idx]:
            # Add all nodes of this face
            # face_nodes[face_idx] is an array of node indices
            nodes.update(face_nodes[face_idx])
        element_nodes[elem_idx] = sorted(nodes)

    # Compute Anb Coefficient Indices
    # These are used for sparse matrix assembly (upper/lower triangles)
    # uFVM logic:
    # For each element, iterate over its faces.
    # If face is interior:
    #   If element is owner, this face connects to a neighbour (upper or lower depending on index?)
    #   Actually uFVM assigns indices based on the order of faces in elementFaces.
    #   Let's replicate uFVM logic exactly.

    upper_anb_coeff_index = np.zeros(n_interior_faces, dtype=np.int32)
    lower_anb_coeff_index = np.zeros(n_interior_faces, dtype=np.int32)

    for elem_idx in range(n_elements):
        # Wait, uFVM uses 1-based indexing for everything.
        # "iNb = 1"
        # "upperAnbCoeffIndex(faceIndex) = iNb"
        # This iNb seems to be the local index of the neighbour in the element's neighbour list?
        # Or is it the column index in the sparse row?
        # In uFVM, sparse matrices are often constructed using these indices.
        # Let's stick to 0-based indexing for Python, but we need to understand what this index represents.
        # It seems to be the index into the 'coefficients' array for this element's row.
        # Since we will likely use scipy.sparse.csr_matrix, we might not need these manual indices in the same way.
        # But for exact reproduction of uFVM logic (if we port assembly 1-to-1), we might need them.
        # However, uFVM's assembly often uses:
        #   theCoefficients.upperAnbCoeff(upperAnbCoeffIndex(iFace)) = ...
        # This implies a global array of coefficients indexed by these indices.
        # Let's compute them 0-based.

        i_nb = 0
        for face_idx in element_faces[elem_idx]:
            if face_idx >= n_interior_faces:
                continue

            own = owners[face_idx]
            nei = neighbours[face_idx]

            if elem_idx == own:
                upper_anb_coeff_index[face_idx] = i_nb
            elif elem_idx == nei:
                lower_anb_coeff_index[face_idx] = i_nb

            i_nb += 1

    return {
        "element_faces": element_faces,
        "element_neighbours": element_neighbours,
        "element_nodes": element_nodes,
        "upper_anb_coeff_index": upper_anb_coeff_index,
        "lower_anb_coeff_index": lower_anb_coeff_index,
    }


def get_element_faces(owners, neighbours, n_elements, n_faces):
    """
    Helper to compute just the element_faces mapping.
    """
    n_interior_faces = len(neighbours)
    element_faces = [[] for _ in range(n_elements)]

    # Process Interior Faces
    for face_idx in range(n_interior_faces):
        own = owners[face_idx]
        nei = neighbours[face_idx]
        element_faces[own].append(face_idx)
        element_faces[nei].append(face_idx)

    # Process Boundary Faces
    for face_idx in range(n_interior_faces, n_faces):
        own = owners[face_idx]
        element_faces[own].append(face_idx)

    return element_faces
