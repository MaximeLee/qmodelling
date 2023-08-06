from scipy import special
import numpy as np

pi = np.pi


def boys(x):
    """boys function used for one-two electron integrals"""
    n = 0
    if x == 0:
        return 1.0 / (2 * n + 1)
    return (
        special.gammainc(n + 0.5, x)
        * special.gamma(n + 0.5)
        * (1.0 / (2 * x ** (n + 0.5)))
    )

