import numpy as np
from utils import to_deterministic_policy, to_stochastic_policy

def naive(pref_left_to_right: np.ndarray) -> np.ndarray:
    """Naive method that recommends depending only on the preference of left to right.

    Parameters
    ----------
    pref_left_to_right : np.ndarray
        The preference of left to right.
    
    Returns
    -------
    np.ndarray
        The stochastic policy. Shape is (num_left, num_right, num_right).
    """
    naive_deterministic_policy = to_deterministic_policy(pref_left_to_right)
    return to_stochastic_policy(naive_deterministic_policy)
