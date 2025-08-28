import numpy as np
import torch
import time
from typing import Union

def nsw_sinkhorn(
        pref_left_to_right: np.ndarray,
        pref_right_to_left: np.ndarray,
        v_left: np.ndarray,
        v_right: np.ndarray,
        device: str = "cpu",
        max_iter: int = 100,
        lr: float = 0.1,
        eps: float = 1e-6,
        sinkhorn_lambda: float = 200.0,
        sinkhorn_max_iter: int = 100,
        sinkhorn_tol: float = 1e-6,
        output: bool = True,
        maxtol: float = 1e-3
) -> tuple[np.ndarray]:
    """
    Returns stochastic policies for left and right users computed via the approximate NSW maximization with Sinkhorn updates.

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
    device : str, optional
        The device to run the computations on, by default "cpu".
    max_iter : int, optional
        The maximum number of iterations, by default 100.
    lr : float, optional
        The learning rate, by default 0.1.
    eps : float, optional
        A small value to prevent division by zero, by default 1e-6.
    sinkhorn_lambda : float, optional
        The regularization parameter for Sinkhorn updates, by default 200.0.
    sinkhorn_max_iter : int, optional
        The maximum number of iterations for Sinkhorn updates, by default 100.
    sinkhorn_tol : float, optional
        The tolerance for Sinkhorn convergence, by default 1e-6.
    output : bool, optional
        Whether to output the progress, by default True.
    maxtol : float, optional
        The maximum tolerance for convergence, by default 1e-3.

    Returns
    -------
    tuple[np.ndarray]
        The stochastic policies for left and right users. Shapes are (num_left, num_right, num_right) and (num_right, num_left, num_left).
    """
    return alternate_fw_sinkhorn(
        pref_left_to_right=pref_left_to_right,
        pref_right_to_left=pref_right_to_left,
        v_left=v_left,
        v_right=v_right,
        device=device,
        objective="NSW",
        max_iter=max_iter,
        lr=lr,
        eps=eps,
        sinkhorn_lambda=sinkhorn_lambda,
        sinkhorn_max_iter=sinkhorn_max_iter,
        sinkhorn_tol=sinkhorn_tol,
        output=output,
        maxtol=maxtol
    )


def sw_sinkhorn(
        pref_left_to_right: np.ndarray,
        pref_right_to_left: np.ndarray,
        v_left: np.ndarray,
        v_right: np.ndarray,
        device: str = "cpu",
        max_iter: int = 100,
        lr: float = 0.1,
        eps: float = 1e-6,
        sinkhorn_lambda: float = 200.0,
        sinkhorn_max_iter: int = 100,
        sinkhorn_tol: float = 1e-6,
        output: bool = True,
        maxtol: float = 1e-3
) -> tuple[np.ndarray]:
    """
    Returns stochastic policies for left and right users computed via the approximate SW maximization with Sinkhorn updates.

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
    device : str, optional
        The device to run the computations on, by default "cpu".
    max_iter : int, optional
        The maximum number of iterations, by default 100.
    lr : float, optional
        The learning rate, by default 0.1.
    eps : float, optional
        A small value to prevent division by zero, by default 1e-6.
    sinkhorn_lambda : float, optional
        The regularization parameter for Sinkhorn updates, by default 200.0.
    sinkhorn_max_iter : int, optional
        The maximum number of iterations for Sinkhorn updates, by default 100.
    sinkhorn_tol : float, optional
        The tolerance for Sinkhorn convergence, by default 1e-6.
    output : bool, optional
        Whether to output the progress, by default True.
    maxtol : float, optional
        The maximum tolerance for convergence, by default 1e-3.

    Returns
    -------
    tuple[np.ndarray]
        The stochastic policies for left and right users. Shapes are (num_left, num_right, num_right) and (num_right, num_left, num_left).
    """
    return alternate_fw_sinkhorn(
        pref_left_to_right=pref_left_to_right,
        pref_right_to_left=pref_right_to_left,
        v_left=v_left,
        v_right=v_right,
        device=device,
        objective="SW",
        max_iter=max_iter,
        lr=lr,
        eps=eps,
        sinkhorn_lambda=sinkhorn_lambda,
        sinkhorn_max_iter=sinkhorn_max_iter,
        sinkhorn_tol=sinkhorn_tol,
        output=output,
        maxtol=maxtol
    )


def alternate_fw_sinkhorn(
        pref_left_to_right: np.ndarray,
        pref_right_to_left: np.ndarray,
        v_left: np.ndarray,
        v_right: np.ndarray,
        device: str = "cpu",
        objective: str = 'SW',
        max_iter: int = 100,
        lr: float = 0.1,
        eps: float = 1e-6,
        sinkhorn_lambda: float = 200.0,
        sinkhorn_max_iter: int = 100,
        sinkhorn_tol: float = 1e-6,
        output: bool = True,
        maxtol: float = 1e-3
) -> tuple[np.ndarray]:
    """
    Perform alternate Frank-Wolfe updates with Sinkhorn iterations.

    Parameters
    ----------
    pref_left_to_right : np.ndarray
        The preference of left to right. Shape is (num_left, num_right).
    pref_right_to_left : np.ndarray
        The preference of right to left. Shape is (num_right, num_left).
    v_left : np.ndarray
        Examination vector for left agents.
    v_right : np.ndarray
        Examination vector for right agents.
    device : str, optional
        Device for torch computations, 'cpu' or 'cuda'. By default 'cpu'.
    objective : str, optional
        Objective to optimize ("SW" or "NSW"), by default 'SW'.
    max_iter : int, optional
        Maximum number of outer iterations, by default 100.
    lr : float, optional
        Learning rate for updates, by default 0.1.
    eps : float, optional
        Small constant to avoid numerical issues, by default 1e-6.
    sinkhorn_lambda : float, optional
        Regularization parameter for Sinkhorn updates, by default 200.0.
    sinkhorn_max_iter : int, optional
        Maximum iterations for Sinkhorn updates, by default 100.
    sinkhorn_tol : float, optional
        Tolerance for Sinkhorn convergence, by default 1e-6.
    output : bool, optional
        Whether to print progress messages, by default True.
    maxtol : float, optional
        Maximum tolerance for convergence, by default 1e-3.

    Returns
    -------
    tuple[np.ndarray]
        The stochastic policy for left and right. Shapes are (num_left, num_right, num_right) and (num_right, num_left, num_left).
    """
    num_left, num_right = pref_left_to_right.shape
    P = pref_left_to_right * pref_right_to_left.T
    A = torch.full(size=(num_left, num_right, num_right), fill_value=1.0/num_right, device=device, dtype=torch.float64)
    B = torch.full(size=(num_right, num_left, num_left), fill_value=1.0/num_left, device=device, dtype=torch.float64)
    V = np.outer(v_left, v_right)

    sw = 0.0

    start = time.time()

    for t in range(max_iter):
        grad_A = compute_partial_A_matrix(A, B, P, V, eps, mode='torch', device=device, objective=objective, return_tensor=True)
        max_vals_A = torch.amax(grad_A.reshape(num_left, -1), dim=1).reshape(num_left, 1, 1)
        cost_A = max_vals_A - grad_A
        A_new = sinkhorn_update_batched_torch(cost_A, sinkhorn_lambda, sinkhorn_max_iter, sinkhorn_tol)
        A = (1-lr)*A + lr*A_new
        A_np = A.cpu().numpy().copy()

        grad_B = compute_partial_B_matrix(A, B, P, V, eps, mode='torch', device=device, objective=objective, return_tensor=True)
        max_vals_B = torch.amax(grad_B.reshape(num_right, -1), dim=1).reshape(num_right, 1, 1)
        cost_B = max_vals_B - grad_B
        B_new = sinkhorn_update_batched_torch(cost_B, sinkhorn_lambda, sinkhorn_max_iter, sinkhorn_tol)
        B = (1-lr)*B + lr*B_new
        B_np = B.cpu().numpy().copy()

        sw_new = np.sum(pref_left_to_right * pref_right_to_left.T * (A_np @ v_left) * (B_np @ v_right).T)
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

    return A_np, B_np


def compute_partial_A_matrix(
        A: Union[np.ndarray, torch.Tensor],
        B: Union[np.ndarray, torch.Tensor],
        P: Union[np.ndarray, torch.Tensor],
        V: Union[np.ndarray, torch.Tensor],
        eps: float = 0.001,
        mode: str = 'torch',
        device: str = 'cpu',
        objective: str = "NSW",
        return_tensor: bool = False
) -> Union[torch.Tensor, np.ndarray]:
    """
    Compute the partial derivative of the objective function with respect to matrix A.

    Parameters
    ----------
    A : Union[np.ndarray, torch.Tensor]
        Current stochastic policy for left side agents.
    B : Union[np.ndarray, torch.Tensor]
        Current stochastic policy for right side agents.
    P : Union[np.ndarray, torch.Tensor]
        The preference matrix. Shape is (num_left, num_right).
    V : Union[np.ndarray, torch.Tensor]
        An outer product of v_left and v_right.
    eps : float, optional
        Small constant to avoid division by zero, by default 0.001.
    mode : str, optional
        Computation mode ("torch" or otherwise NumPy), by default 'torch'.
    device : str, optional
        Device for torch computations, 'cpu' or 'cuda'.
    objective : str, optional
        Objective type ("NSW" or "SW"), by default "NSW".
    return_tensor : bool, optional
        Whether to return a torch.Tensor (if mode is 'torch'). If False, returns a NumPy array.
    
    Returns
    -------
    Union[torch.Tensor, np.ndarray]
        The partial derivative of the objective with respect to A.
    """
    if mode == 'torch':
        A_t = A if isinstance(A, torch.Tensor) else torch.tensor(A, device=device, dtype=torch.float64)
        B_t = B if isinstance(B, torch.Tensor) else torch.tensor(B, device=device, dtype=torch.float64)
        P_t = P if isinstance(P, torch.Tensor) else torch.tensor(P, device=device, dtype=torch.float64)
        V_t = V if isinstance(V, torch.Tensor) else torch.tensor(V, device=device, dtype=torch.float64)
        V_j = torch.einsum('ij,kl,ijk,jil->j', P_t, V_t, A_t, B_t)
        V_j = torch.clamp(V_j, min=eps)
        numerator = torch.einsum('ij,kl,jil->ijk', P_t, V_t, B_t)
        if objective == "NSW":
            partial_A = numerator / V_j.unsqueeze(0).unsqueeze(-1)
        elif objective == "SW":
            partial_A = numerator
        else:
            raise ValueError(f"Unknown method")
        return partial_A if return_tensor else partial_A.cpu().numpy()
    else:
        V_j = np.einsum('ij,kl,ijk,jil->j', P, V, A, B)
        V_j = np.maximum(V_j, eps)
        numerator = np.einsum('ij,kl,jil->ijk', P, V, B)
        if objective == "NSW":
            partial_A = numerator / V_j[None, :, None]
        elif objective == "SW":
            partial_A = numerator
        else:
            raise ValueError(f"Unknown method")
        return partial_A


def compute_partial_B_matrix(
        A: Union[np.ndarray, torch.Tensor],
        B: Union[np.ndarray, torch.Tensor],
        P: Union[np.ndarray, torch.Tensor],
        V: Union[np.ndarray, torch.Tensor],
        eps: float = 0.001,
        mode: str = 'torch',
        device: str = 'cpu',
        objective: str = "NSW",
        return_tensor: bool = False
) -> Union[torch.Tensor, np.ndarray]:
    """
    Compute the partial derivative of the objective function with respect to matrix B.

    Parameters
    ----------
    A : Union[np.ndarray, torch.Tensor]
        Current stochastic policy for left side agents.
    B : Union[np.ndarray, torch.Tensor]
        Current stochastic policy for right side agents.
    P : Union[np.ndarray, torch.Tensor]
        The preference matrix. Shape is (num_left, num_right).
    V : Union[np.ndarray, torch.Tensor]
        An outer product of v_left and v_right.
    eps : float, optional
        Small constant to avoid division by zero, by default 0.001.
    mode : str, optional
        Computation mode ("torch" or otherwise NumPy), by default 'torch'.
    device : str, optional
        Device for torch computations, 'cpu' or 'cuda'.
    objective : str, optional
        Objective type ("NSW" or "SW"), by default "NSW".
    return_tensor : bool, optional
        Whether to return a torch.Tensor (if mode is 'torch'). If False, returns a NumPy array.
    
    Returns
    -------
    Union[torch.Tensor, np.ndarray]
        The partial derivative of the objective with respect to B.
    """
    if mode == 'torch':
        A_t = A if isinstance(A, torch.Tensor) else torch.tensor(A, device=device, dtype=torch.float64)
        B_t = B if isinstance(B, torch.Tensor) else torch.tensor(B, device=device, dtype=torch.float64)
        P_t = P if isinstance(P, torch.Tensor) else torch.tensor(P, device=device, dtype=torch.float64)
        V_t = V if isinstance(V, torch.Tensor) else torch.tensor(V, device=device, dtype=torch.float64)
        U_i = torch.einsum('ij,kl,ijk,jil->i', P_t, V_t, A_t, B_t)
        U_i = torch.clamp(U_i, min=eps)
        numerator = torch.einsum('ij,kl,ijk->jil', P_t, V_t, A_t)
        if objective == "NSW":
            partial_B = numerator / U_i.unsqueeze(0).unsqueeze(-1)
        elif objective == "SW":
            partial_B = numerator
        else:
            raise ValueError(f"Unknown objective")
        return partial_B if return_tensor else partial_B.cpu().numpy()
    else:
        U_i = np.einsum('ij,kl,ijk,jil->i', P, V, A, B)
        U_i = np.maximum(U_i, eps)
        numerator = np.einsum('ij,kl,ijk->jil', P, V, A)
        if objective == "NSW":
            partial_B = numerator / U_i[None, :, None]
        elif objective == "SW":
            partial_B = numerator
        else:
            raise ValueError(f"Unknown objective")
        return partial_B


def sinkhorn_update_batched_torch(
        cost: torch.Tensor,
        lam: float,
        max_iter: int = 100,
        tol: float = 1e-6
) -> torch.Tensor:
    """
    Perform a batched Sinkhorn update using PyTorch.

    Parameters
    ----------
    cost : torch.Tensor
        Cost matrix of shape (batch, m, m).
    lam : float
        Regularization parameter (lambda).
    max_iter : int, optional
        Maximum number of Sinkhorn iterations, by default 100.
    tol : float, optional
        Tolerance for convergence, by default 1e-6.
    
    Returns
    -------
    torch.Tensor
        The optimal transport (scaling) matrix.
    """
    K = torch.exp(-lam * cost)
    batch, m, _ = K.shape
    u = torch.ones(batch, m, dtype=K.dtype, device=K.device)
    v = torch.ones(batch, m, dtype=K.dtype, device=K.device)
    for iteration in range(max_iter):
        v_new = 1.0 / torch.matmul(K.transpose(-2, -1), u.unsqueeze(-1)).squeeze(-1)
        u_new = 1.0 / torch.matmul(K, v_new.unsqueeze(-1)).squeeze(-1)
        diff_u = torch.max(torch.abs(u_new - u))
        diff_v = torch.max(torch.abs(v_new - v))
        if max(diff_u, diff_v) < tol:
            u, v = u_new, v_new
            break
        u, v = u_new, v_new
    transport_matrix = u.unsqueeze(-1) * K * v.unsqueeze(1)
    return transport_matrix
