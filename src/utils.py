import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Encode numpy types to json."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def to_deterministic_policy(array: np.ndarray) -> np.ndarray:
    """Convert 2d score array to deterministic policy.
    
    Parameters
    ----------
    array : np.ndarray
        2d score array.
    
    Returns
    -------
    np.ndarray
        Deterministic policy.
    """
    return np.argsort(-array, axis=1)


def to_stochastic_policy(array: np.ndarray) -> np.ndarray:
    """Convert deterministic policy to stochastic policy.
    
    Parameters
    ----------
    array : np.ndarray
        Deterministic policy.
    
    Returns
    -------
    np.ndarray
        Stochastic policy.
    """
    return np.array([np.eye(arr1d.size)[arr1d].T for arr1d in array])


def generate_examination_vector(v_type: str, num_rank: int) -> np.ndarray:
    """Generate examination vector.

    Parameters
    ----------
    v_type : str
        Type of examination vector. One of ["inv", "log", "exp", "k_1", "uni", "lin"].
    num_rank : int
        Number of ranks to be examined. (i.e., number of opposite side agents)

    Returns
    -------
    np.ndarray
        Examination vector.
    """
    if v_type == "inv":
        return 1 / np.arange(1, num_rank + 1)
    elif v_type == "log":
        return 1 / np.log2(np.arange(2, num_rank + 2))
    elif v_type == "exp":
        return 1 / np.exp(np.arange(num_rank))
    elif v_type == "k_1":
        return np.array([1.0]+[0.0]*(num_rank-1))
    elif v_type == "uni":
        return np.array([1.0] * num_rank)
    elif v_type == "lin":
        return np.linspace(1.0, 0.0, num=num_rank)
    else:
        raise ValueError("Invalid v_type.")
