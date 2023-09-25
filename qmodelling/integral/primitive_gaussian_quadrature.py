import copy as cp
import numpy as np
from qmodelling.integral.quadrature import get_quadrature_points

nquad = 20

# pts on [-1,1]
Chebyshev_quadrature_points = get_quadrature_points(
    n=nquad, quadrature_type="gauss_chebyshev_2"
)

# pts on [0,1]
Chebyshev_quadrature_points_01 = cp.deepcopy(Chebyshev_quadrature_points)
Chebyshev_quadrature_points_01[:, 1] = (
    Chebyshev_quadrature_points_01[:, 1] + 1.0
) / 2.0
# sum weights = 1
Chebyshev_quadrature_points_01[:, 0] /= 2.0

# Generating points for R³ integration with variable substitution on [-1,1]³
weights_1d = Chebyshev_quadrature_points[:, 0]
weights_3d = np.meshgrid(weights_1d, weights_1d, weights_1d, indexing="ij")

Chebyshev_weights_3d = (weights_3d[0] * weights_3d[1] * weights_3d[2]).flatten("A")

abscissa_1d = Chebyshev_quadrature_points[:, 1]
abscissa_3d = np.meshgrid(abscissa_1d, abscissa_1d, abscissa_1d, indexing="ij")
Chebyshev_abscissa_3d = np.stack([p.flatten("A") for p in abscissa_3d], axis=1).reshape(
    -1, 1, 3
)

del weights_1d
del weights_3d
del abscissa_1d
del abscissa_3d
