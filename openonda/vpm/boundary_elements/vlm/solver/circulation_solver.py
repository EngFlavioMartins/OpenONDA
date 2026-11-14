"""Dense solve of the VLM circulation system with cached LU factors."""

import warnings

import numpy as np
import scipy.linalg
import taichi as ti


class CirculationSolver:
    """Solve ``influence @ circulation = right_hand_side`` with SciPy LU factors.

    Factorization costs O(N³); right-hand sides of an identical matrix reuse
    the factors. A singular or non-finite system raises before either the
    cache or the output circulation is changed.
    """

    def __init__(self) -> None:
        """Start with an empty matrix and factor cache."""
        self._matrix: np.ndarray | None = None
        self._factorization: tuple[np.ndarray, np.ndarray] | None = None

    def solve(
        self, influence: ti.Field, right_hand_side: ti.Field, circulation: ti.Field, n_panels: int
    ) -> None:
        """Solve the active ``n_panels`` system in the precision of the fields.

        Parameters
        ----------
        influence : Taichi field, shape (capacity, capacity)
            Dense influence matrix; entries map panel circulation to the
            collocation-point normal velocity.
        right_hand_side : Taichi field, shape (capacity,)
            Boundary-condition vector.
        circulation : Taichi field, shape (capacity,)
            Solution storage; the active prefix is overwritten and the rest
            set to zero.
        n_panels : int
            Active system size.

        Raises
        ------
        numpy.linalg.LinAlgError
            If the influence matrix is singular or the solution is not finite.
        """
        matrix = influence.to_numpy()[:n_panels, :n_panels]
        rhs = right_hand_side.to_numpy()[:n_panels]
        if self._matrix is None or not np.array_equal(matrix, self._matrix):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", scipy.linalg.LinAlgWarning)
                    factorization = scipy.linalg.lu_factor(matrix)
            except scipy.linalg.LinAlgWarning as error:
                raise np.linalg.LinAlgError("VLM influence matrix is singular") from error
            self._matrix, self._factorization = matrix, factorization
        solution = scipy.linalg.lu_solve(self._factorization, rhs)
        if not np.all(np.isfinite(solution)):
            raise np.linalg.LinAlgError("VLM circulation solve produced non-finite values")
        full = np.zeros(circulation.shape[0], dtype=matrix.dtype)
        full[:n_panels] = solution
        circulation.from_numpy(full)
