import numpy as np


def l1(a: np.ndarray, b: np.ndarray) -> float:
    """Count l1 (Manhattan) metric"""

    return np.sum(np.abs(a - b))


def l2(a: np.ndarray, b: np.ndarray) -> float:
    """Count l2 (Euclidean) metric"""

    return np.sum((a - b) ** 2)
