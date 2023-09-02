cimport numpy as cnp
from qmodelling.constants cimport dtype

cdef short ngauss
cdef short ngauss_3d

cdef cnp.ndarray[dtype, ndim=2] get_Chebyshev_01()
cdef tuple get_Chebyshev_3d()
