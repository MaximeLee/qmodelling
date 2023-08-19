"""Quadrature module to get quadrature points"""
import os
import warnings
import numpy as np

pi = np.pi


def get_quadrature_points(n, quadrature_type="gauss_chebyshev_2"):
    """quadrature points :
    - first column : weights
    - second and next columns : abscissa on the interval/domain

    Supported quadrature types :
    - gauss_chebyshev_2 : Gauss Chebyshev quadrature of the second kind
    - lebedev : https://people.sc.fsu.edu/~jburkardt/datasets/sphere_lebedev_rule/sphere_lebedev_rule.html
    """
    if quadrature_type == "gauss_chebyshev_2":
        i = np.arange(1.0, n + 1.0)
        sin_i = np.sin(i * pi / (n + 1.0))
        cos_i = np.cos(i * pi / (n + 1.0))

        abscissa = (n + 1.0 - 2.0 * i) / (n + 1.0) + 2.0 / pi * (
            1.0 + 2.0 / 3.0 * sin_i**2
        ) * cos_i * sin_i
        abscissa = abscissa.reshape(-1, 1)

        weights = 16.0 / 3.0 / (n + 1.0) * sin_i**4.0
        weights = weights.reshape(-1, 1)

    elif quadrature_type == "lebedev":
        data = np.loadtxt(f"{os.path.dirname(__file__)}/lebedev_5810.txt")

        abscissa = data[:, :2] * pi / 180.0
        weights = data[:, 2:3]

        warnings.warn(f"Default number of points for Levedev quadrature is {len(data)}")

    else:
        raise ValueError("Quadrature type not recognized/supported.")
    return np.concatenate([weights, abscissa], axis=1)
