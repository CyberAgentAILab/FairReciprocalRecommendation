import numpy as np
import cvxpy as cp
from utils import to_deterministic_policy, to_stochastic_policy

def iter_lp(pref_left_to_right: np.ndarray, pref_right_to_left: np.ndarray, solver: str = "CLARABEL") -> tuple[np.ndarray]:
    """Returns stochastic policy for left and right computed via iterative LP.

    Parameters
    ----------
    pref_left_to_right : np.ndarray
        The preference of left to right. Shape is (num_left, num_right).
    pref_right_to_left : np.ndarray
        The preference of right to left. Shape is (num_right, num_left).
    solver : str, optional
        Solver for CVXPY, like "CLARABEL", "ECOS", etc. by default "CLARABEL".
    
    Returns
    -------
    tuple[np.ndarray]
        The stochastic policy for left and right. Shapes are (num_left, num_right, num_right) and (num_right, num_left, num_left).
    """
    assert pref_left_to_right.shape == pref_right_to_left.T.shape
    coeff_matrix = pref_left_to_right * pref_right_to_left.T
    return __iter_lp(coeff_matrix + 1, solver=solver), __iter_lp(coeff_matrix.T + 1, solver=solver)


def __iter_lp(coeff_matrix: np.ndarray, solver: str) -> np.ndarray:
    """Returns stochastic policy computed via iterative LP for one side.

    Parameters
    ----------
    coeff_matrix : np.ndarray
        Coefficient matrix for LP.
    
    Returns
    -------
    np.ndarray
        Stochastic policy for one side.
    """
    num_left, num_right = coeff_matrix.shape
    x = cp.Variable(shape=(num_left, num_right))
    constraints = [0 <= x, x @ np.ones(num_right) <= np.ones(num_left), x.T @ np.ones(num_left) <= np.ones(num_right)]
    A = coeff_matrix.copy()
    res = np.zeros(shape=(num_left, num_right))

    for t in range(num_right):
        obj = cp.Maximize(cp.sum(cp.multiply(A, x)))
        cp.Problem(objective=obj, constraints=constraints).solve(solver=solver, verbose=False)
        argmax = np.argmax(x.value, axis=1)
        for m in range(num_left):
            res[m, argmax[m]] = num_right - t
            A[m, argmax[m]] = -100000
    
    return to_stochastic_policy(to_deterministic_policy(res))
