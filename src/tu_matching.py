import numpy as np
from utils import to_deterministic_policy, to_stochastic_policy

def tu_matching(
        pref_left_to_right: np.ndarray,
        pref_right_to_left: np.ndarray,
        beta: float = 1.0,
        maxit: int = 10000,
        maxvaltol: float = 1e-9,
        maxsteptol: float = 1e-9,
        output: bool = True
) -> tuple[np.ndarray]:
    """Returns the TU matching policy computed via IPFP, in the form of a stochastic policy [Tomita et al. 2023].

    Parameters:
    -----------
    pref_left_to_right : np.ndarray
        The preference of left to right. Shape is (num_left, num_right).
    pref_right_to_left : np.ndarray
        The preference of right to left. Shape is (num_right, num_left).
    beta : float, optional
        Hyperparameter for IPFP, by default 1.0.
    maxit : int, optional
        Maximum number of iterations for IPFP, by default 10000.
    maxvaltol : float, optional
        Tolerance for the value difference in IPFP, by default 1e-9.
    maxsteptol : float, optional
        Tolerance for the step difference in IPFP, by default 1e-9.
    output : bool, optional
        Whether to print the convergence message, by default True.
    
    Returns:
    --------
    tuple[np.ndarray]
        The stochastic policy for left and right. Shapes are (num_left, num_right, num_right) and (num_right, num_left, num_left).

    References:
        Yoji Tomita, Riku Togashi, Yuriko Hashizume and Naoto Ohsaka. "Fast and Examination-agnostic Reciprocal Recommendation in Matching Markets," RecSys, 2023.
    """
    num_left, num_right = pref_left_to_right.shape

    # Initialize mu_c0, mu_0j, mu
    mu_c0 = np.ones(num_left)
    mu_0j = np.ones(num_right)
    mu = np.ones(shape=(num_left, num_right))

    K = np.exp((pref_left_to_right + pref_right_to_left.T) / (2.0 * beta))
    A = np.sqrt(mu_c0)
    B = np.sqrt(mu_0j)

    # Main loop
    for i in range(maxit):
        # Update A
        KBhalf = (K @ B) / 2.0
        new_A = np.sqrt(np.ones(num_left) + KBhalf * KBhalf) - KBhalf
        update = np.max(np.abs(new_A - A))
        A = new_A.copy()

        if 0.0 in A:
            raise RuntimeError("Zero value in A.")

        # Update B
        KAhalf = (K.T @ A) / 2.0
        new_B = np.sqrt(np.ones(num_right) + KAhalf * KAhalf) - KAhalf
        update = np.max([update, np.max(np.abs(new_B - B))])
        B = new_B.copy()

        if 0.0 in B:
            raise RuntimeError("Zero value in B.")

        # Update mu_c0, mu_0j, mu and check convergence
        mu_c0 = A * A
        mu_0j = B * B
        mu = (K
              * (np.tile(A.reshape(num_left, 1), reps=(1, num_right)))
              * (np.tile(B.reshape(1, num_right), reps=(num_left, 1))))

        valdiff = np.max([
            np.max(np.abs(np.ones(num_left) - mu.sum(axis=1) - mu_c0)),
            np.max(np.abs(np.ones(num_right) - mu.sum(axis=0) - mu_0j))
        ])

        if update < maxsteptol and valdiff < maxvaltol:
            if output:
                print(
                    f"IPFP converged in {i+1} iterations. Update={update}, Valdiff={valdiff}."
                )
            break
    else:
        raise RuntimeError("Not converged.")
    
    tu_deterministic_policy_for_left = to_deterministic_policy(mu)
    tu_deterministic_policy_for_right = to_deterministic_policy(mu.T)

    return (to_stochastic_policy(tu_deterministic_policy_for_left), to_stochastic_policy(tu_deterministic_policy_for_right))
