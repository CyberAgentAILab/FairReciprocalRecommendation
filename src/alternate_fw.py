import numpy as np
import cvxpy as cp
import time


def nsw_maximize(
        pref_left_to_right: np.ndarray,
        pref_right_to_left: np.ndarray,
        v_left: np.ndarray,
        v_right: np.ndarray,
        maxit: int = 100,
        maxtol: float = 0.01,
        lr: float = 0.1,
        solver: str = "CLARABEL",
        output: bool = True,
        eps: float = 1e-6
) -> tuple[np.ndarray]:
    """Returns stochastic policy for left and right computed via NSW maximization.

    Parameters
    ----------
    pref_left_to_right : np.ndarray
        The preference of left to right. Shape is (num_left, num_right).
    pref_right_to_left : np.ndarray
        The preference of right to left. Shape is (num_right, num_left).
    v_left : np.ndarray
        The examination vector for left side agents.
    v_right : np.ndarray
        The examination vector for right side agents.
    maxit : int, optional
        Maximum number of iterations, by default 100.
    maxtol : float, optional
        Maximum tolerance for convergence, by default 0.01.
    lr : float, optional
        Learning rate, by default 0.1.
    solver : str, optional
        Solver for CVXPY, like "CLARABEL", "ECOS", etc. by default "CLARABEL".
    output : bool, optional
        Whether to output the progress, by default True.
    eps : float, optional
        A small value to prevent division by zero.

    Returns
    -------
    tuple[np.ndarray]
        The stochastic policy for left and right. Shapes are (num_left, num_right, num_right) and (num_right, num_left, num_left).
    """
    return alternate_fw(pref_left_to_right, pref_right_to_left, v_left, v_right, maxit, maxtol, lr, solver, output, alpha=0, eps=eps)


def sw_maximize(
        pref_left_to_right: np.ndarray,
        pref_right_to_left: np.ndarray,
        v_left: np.ndarray,
        v_right: np.ndarray,
        maxit: int = 100,
        maxtol: float = 0.01,
        lr: float = 0.1,
        solver: str = "CLARABEL",
        output: bool = True,
        eps: float = 1e-6
) -> tuple[np.ndarray]:
    """Returns stochastic policy for left and right computed via SW maximization.

    Parameters
    ----------
    pref_left_to_right : np.ndarray
        The preference of left to right. Shape is (num_left, num_right).
    pref_right_to_left : np.ndarray
        The preference of right to left. Shape is (num_right, num_left).
    v_left : np.ndarray
        The examination vector for left side agents.
    v_right : np.ndarray
        The examination vector for right side agents.
    maxit : int, optional
        Maximum number of iterations, by default 100.
    maxtol : float, optional
        Maximum tolerance for convergence, by default 0.01.
    lr : float, optional
        Learning rate, by default 0.1.
    solver : str, optional
        Solver for CVXPY, like "CLARABEL", "ECOS", etc. by default "CLARABEL".
    output : bool, optional
        Whether to output the progress, by default True.
    eps : float, optional
        A small value to prevent division by zero.

    Returns
    -------
    tuple[np.ndarray]
        The stochastic policy for left and right. Shapes are (num_left, num_right, num_right) and (num_right, num_left, num_left).
    """
    return alternate_fw(pref_left_to_right, pref_right_to_left, v_left, v_right, maxit, maxtol, lr, solver, output, alpha=1, eps=eps)


def alpha_sw_maximize(
        pref_left_to_right: np.ndarray,
        pref_right_to_left: np.ndarray,
        v_left: np.ndarray,
        v_right: np.ndarray,
        maxit: int = 100,
        maxtol: float = 0.01,
        lr: float = 0.1,
        solver: str = "CLARABEL",
        output: bool = True,
        alpha: float = 1.0,
        eps: float = 1e-6
) -> tuple[np.ndarray]:
    """Returns stochastic policy for left and right computed via SW maximization.

    Parameters
    ----------
    pref_left_to_right : np.ndarray
        The preference of left to right. Shape is (num_left, num_right).
    pref_right_to_left : np.ndarray
        The preference of right to left. Shape is (num_right, num_left).
    v_left : np.ndarray
        The examination vector for left side agents.
    v_right : np.ndarray
        The examination vector for right side agents.
    maxit : int, optional
        Maximum number of iterations, by default 100.
    maxtol : float, optional
        Maximum tolerance for convergence, by default 0.01.
    lr : float, optional
        Learning rate, by default 0.1.
    solver : str, optional
        Solver for CVXPY, like "CLARABEL", "ECOS", etc. by default "CLARABEL".
    output : bool, optional
        Whether to output the progress, by default True.
    alpha : float, optional
        The trade-off parameter for the alpha-SW method.
    eps : float, optional
        A small value to prevent division by zero.

    Returns
    -------
    tuple[np.ndarray]
        The stochastic policy for left and right. Shapes are (num_left, num_right, num_right) and (num_right, num_left, num_left).
    """
    return alternate_fw(pref_left_to_right, pref_right_to_left, v_left, v_right, maxit, maxtol, lr, solver, output, alpha, eps)


def alternate_fw(
        pref_left_to_right: np.ndarray,
        pref_right_to_left: np.ndarray,
        v_left: np.ndarray,
        v_right: np.ndarray,
        maxit: int = 100,
        maxtol: float = 0.01,
        lr: float = 0.1,
        solver: str = "CLARABEL",
        output: bool = True,
        alpha: float = 1.0,
        eps: float = 1e-6
) -> tuple[np.ndarray]:
    """Returns stochastic policy for left and right computed via alternate Frank-Wolfe algorithm.

    Parameters
    ----------
    pref_left_to_right : np.ndarray
        The preference of left to right. Shape is (num_left, num_right).
    pref_right_to_left : np.ndarray
        The preference of right to left. Shape is (num_right, num_left).
    v_left : np.ndarray
        The examination vector for left side agents.
    v_right : np.ndarray
        The examination vector for right side agents.
    maxit : int, optional
        Maximum number of iterations, by default 100.
    maxtol : float, optional
        Maximum tolerance for convergence, by default 0.01.
    lr : float, optional
        Learning rate, by default 0.1.
    solver : str, optional
        Solver for CVXPY, like "CLARABEL", "ECOS", etc. by default "CLARABEL".
    output : bool, optional
        Whether to output the progress, by default True.
    alpha : float, optional
        The trade-off parameter for the alpha-SW method.
    eps : float, optional
        A small value to prevent division by zero.

    Returns
    -------
    tuple[np.ndarray]
        The stochastic policy for left and right. Shapes are (num_left, num_right, num_right) and (num_right, num_left, num_left).
    """

    num_left, num_right = pref_left_to_right.shape
    A = np.full(shape=(num_left, num_right, num_right), fill_value=1.0/num_right)
    B = np.full(shape=(num_right, num_left, num_left), fill_value=1.0/num_left)
    P = pref_left_to_right * pref_right_to_left.T
    sw = 0.0

    start = time.time()

    for t in range(maxit):
        # Update B
        b = __fw_one_step(P, A, B, v_left, v_right, num_left, num_right, solver, alpha, eps)
        B = (1-lr) * B + lr * b.reshape((num_right, num_left, num_left))

        # Update A
        a = __fw_one_step(P.T, B, A, v_right, v_left, num_right, num_left, solver, alpha, eps)
        A = (1-lr)*A + lr * a.reshape((num_left, num_right, num_right))

        # Calculate social welfare
        sw_new = np.sum(pref_left_to_right * pref_right_to_left.T * (A @ v_left) * (B @ v_right).T)
        updates = np.abs(sw - sw_new)
        sw = sw_new

        current_time = time.time()

        if output:
            print(f"Step:{t+1:03}  SW:{sw:.5f}  UPDATE:{updates:.5f}  TIME:{current_time-start:.5f}")

        if updates < maxtol:
            if output:
                print(f"Converged in {t+1} iterations.")
            break
    else:
        if output:
            print(f"Stopped because it has been passed {t+1} steps.")
    
    return A, B


def __fw_one_step(P, A, B, v_left, v_right, num_left, num_right, solver, alpha, eps):
    """One step of the alternate frank-wolfe algorithm.

    Parameters
    ----------
    P : np.ndarray
        The preference matrix. Shape is (num_left, num_right).
    A : np.ndarray
        Current stochastic policy for left side agents.
    B : np.ndarray
        Current stochastic policy for right side agents.
    v_left : np.ndarray
        The examination vector for left side agents.
    v_right : np.ndarray
        The examination vector for right side agents.
    num_left : int
        The number of left side agents.
    num_right : int
        The number of right side agents.
    solver : str
        Solver for CVXPY, like "CLARABEL", "ECOS", etc.
    alpha : float
        The trade-off parameter for the alpha-SW method.
    eps : float
        A small value to prevent division by zero.
    """
    b = cp.Variable(shape=(num_right, num_left**2))
    ones_num_left = np.ones(num_left)
    constraints = [b[:, num_left * m : num_left * (m+1)] @ ones_num_left == 1 for m in range(num_left)] + [b[:, m : num_left**2 : num_left] @ ones_num_left == 1 for m in range(num_left)] + [0 <= b]

    obj = 0.0
    for m in range(num_left):
        coeff = (P[m, :] * (A[m, :, :] @ v_left)).reshape(num_right, 1) @ v_right.reshape(1, num_left)
        coeff *= max(np.sum(coeff * B[:, m, :]), eps) ** (alpha - 1.0)
        obj += cp.sum(cp.multiply(coeff, b[:, num_left * m : num_left * (m + 1)]))
    cp.Problem(objective=cp.Maximize(obj), constraints=constraints).solve(solver=solver, verbose=False)

    return b.value
