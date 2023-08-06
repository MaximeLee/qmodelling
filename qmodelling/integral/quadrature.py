"""Quadrature module to get quadrature points"""
import numpy as np

pi = np.pi


def get_quadrature_points(n, quadrature_type="gauss_chebyshev_2"):
    """quadrature points :
    - first column : weights
    - second column : abscissa on the interval

    Supported quadrature types :
    - gauss_chebyshev_2 : Gauss Chebyshev quadrature of the second kind
    """
    if quadrature_type == "gauss_chebyshev_2":
        i = np.arange(1.0, n + 1.0)
        sin_i = np.sin(i * pi / (n + 1.0))
        cos_i = np.cos(i * pi / (n + 1.0))

        abscissa = (n + 1.0 - 2.0 * i) / (n + 1.0) + 2.0 / pi * (
            1.0 + 2.0 / 3.0 * sin_i**2
        ) * cos_i * sin_i

        weights = 16.0 / 3.0 / (n + 1.0) * sin_i**4.0

    else:
        raise ValueError("Quadrature type not recognized/supported.")
    return np.stack([weights, abscissa],axis=1)
