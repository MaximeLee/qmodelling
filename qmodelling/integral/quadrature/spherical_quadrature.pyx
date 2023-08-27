"""module containing quadrature points for R³ in spherical coordinates :
- Gauss Chebyshev quadrature for radial component
- Lebedev quadrature for angular components (theta, phi)
"""
from copy import deepcopy as dcp
import numpy as np
from qmodelling.integral.quadrature.get_quadrature import get_quadrature_points


def tensor_prod(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """Cartesian/tensor product of two arrays"""
    n1 = len(X1)
    n2 = len(X2)

    Xprod = np.concatenate([np.tile(X1, (n2, 1)), np.repeat(X2, n1, axis=0)], axis=1)

    return Xprod


###################################
# Get Chebyshev quadrature points on [-1,1]
###################################
Chebyshev_quadrature = get_quadrature_points(50, "gauss_chebyshev_2")
Chebyshev_weights = Chebyshev_quadrature[:, 0:1]
Chebyshev_abscissa = Chebyshev_quadrature[:, 1:2]

# variable substitution from [-1,1] to [0, +infty[
# Briggs-slater radius for Hydrogen
rm = 0.35
# quadrature radius
R_quadrature = rm * (1.0 + Chebyshev_abscissa) / (1.0 - Chebyshev_abscissa)

###################################
# Get Lebedev quadrature points
###################################
Lebedev_quadrature = get_quadrature_points(None, "lebedev")
Lebedev_weights = Lebedev_quadrature[:, 0:1]
Lebedev_abscissa = Lebedev_quadrature[:, 1:]

######################################################
# Make the tensor product to have 3d integration points
######################################################
R3_weights_quadrature = np.prod(
    tensor_prod(Chebyshev_weights, Lebedev_weights), axis=1, keepdims=True
)

del (Chebyshev_weights, Lebedev_weights)

R3_points_quadrature = tensor_prod(R_quadrature, Lebedev_abscissa)
R = dcp(R3_points_quadrature[:, 0])
Theta = dcp(R3_points_quadrature[:, 1])
Phi = dcp(R3_points_quadrature[:, 2])

X = R * np.cos(Theta) * np.sin(Phi)
Y = R * np.sin(Theta) * np.sin(Phi)
Z = R * np.cos(Phi)

R3_points_quadrature[:, 0] = X
R3_points_quadrature[:, 1] = Y
R3_points_quadrature[:, 2] = Z

Mu_int = (R - rm) / (R + rm)

# to multiply the integrand with because of the variable substitution
subs = (R**2 * 2.0 * rm / (1.0 - Mu_int) ** 2).reshape(-1, 1)

del (Chebyshev_abscissa, Lebedev_abscissa)
del (R, Theta, Phi)
del (X, Y, Z)
