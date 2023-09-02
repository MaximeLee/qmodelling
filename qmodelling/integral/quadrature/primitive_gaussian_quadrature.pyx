import numpy as np
from cython cimport tuple
from qmodelling.constants cimport dtype
from qmodelling.integral.quadrature.get_quadrature import get_quadrature_points

ngauss = 20
ngauss_3d = ngauss**3

cdef cnp.ndarray[dtype, ndim=2] get_Chebyshev_01():
    # pts on [-1,1]
    Chebyshev_quadrature_points = get_quadrature_points(
        n=ngauss, quadrature_type="gauss_chebyshev_2"
    )
    
    # pts on [0,1]
    Chebyshev_01 = Chebyshev_quadrature_points
    Chebyshev_01[:, 1] = (
        Chebyshev_01[:, 1] + 1.0
    ) / 2.0
    # sum weights = 1
    Chebyshev_01[:, 0] /= 2.0

    return Chebyshev_01

cdef tuple get_Chebyshev_3d():
    # pts on [-1,1]
    Chebyshev_quadrature_points = get_quadrature_points(
        n=ngauss, quadrature_type="gauss_chebyshev_2"
    )
    
    # Generating points for R³ integration with variable substitution on [-1,1]³
    weights_1d = Chebyshev_quadrature_points[:, 0]
    weights_3d = np.meshgrid(weights_1d, weights_1d, weights_1d, indexing="ij")
    cdef double[:] Chebyshev_weights_3d = (weights_3d[0] * weights_3d[1] * weights_3d[2]).flatten("A")
    
    abscissa_1d = Chebyshev_quadrature_points[:, 1]
    abscissa_3d = np.meshgrid(abscissa_1d, abscissa_1d, abscissa_1d, indexing="ij")
    cdef double[:,:] Chebyshev_abscissa_3d = np.stack([p.flatten("A") for p in abscissa_3d], axis=1).reshape(
       -1, 3
    )

    return Chebyshev_weights_3d, Chebyshev_abscissa_3d
