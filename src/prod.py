import numpy as np
from utils import to_deterministic_policy, to_stochastic_policy

def prod(pref_left_to_right: np.ndarray, pref_right_to_left: np.ndarray) -> np.ndarray:
    """Product method that recommends depending on the preference of both sides.
    
    Parameters
    ----------
    pref_left_to_right : np.ndarray
        The preference of left to right. Shape is (num_left, num_right).
    pref_right_to_left : np.ndarray
        The preference of right to left. Shape is (num_right, num_left).
    
    Returns
    -------
    np.ndarray
        The stochastic policy. Shape is (num_left, num_right, num_right).
    """
    assert pref_left_to_right.shape == pref_right_to_left.T.shape
    prod_deterministic_policy = to_deterministic_policy(pref_left_to_right * pref_right_to_left.T)
    return to_stochastic_policy(prod_deterministic_policy)
