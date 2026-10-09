"""Row-wise algebra on ``(n, 3)`` arrays of face and cell vectors."""

import numpy as np


def vector_dot(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Row-wise dot product of ``(n, 3)`` vector arrays.

    The components are summed in the order of ``np.sum(first * second,
    axis=1)``, so results are bit-identical, without the short-axis
    reduction overhead.
    """
    return first[:, 0] * second[:, 0] + first[:, 1] * second[:, 1] + first[:, 2] * second[:, 2]


def gathered_vector_dot(vectors: np.ndarray, rows: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return ``vector_dot(vectors[rows], second)`` gathering one column at a time.

    Column gathers avoid copying whole rows of strided gradient slices.
    """
    return (
        vectors[rows, 0] * second[:, 0]
        + vectors[rows, 1] * second[:, 1]
        + vectors[rows, 2] * second[:, 2]
    )


def vector_magnitude(vector: np.ndarray) -> np.ndarray:
    """Row-wise Euclidean magnitude of an ``(n, 3)`` array, as ``norm(axis=1)``."""
    return np.sqrt(vector_dot(vector, vector))
