cimport numpy as cnp
import numpy as np

from qmodelling.constants cimport dtype
from qmodelling.c_ops.math cimport sqrt, atanh, pow

cdef double[:] sub(double[:] v1, double[:] v2):

    cdef short n1 = v1.shape[0]

    cdef cnp.ndarray[dtype, ndim=1] v3 = np.zeros(n1)
    cdef double[:] v3_ = v3

    for i in range(n1):
        v3_[i] = v1[i] - v2[i]
    return v3

cdef short[:] add_short(short[:] v, short val):

    cdef short n = v.shape[0]
    cdef cnp.ndarray v_ = np.zeros(n, dtype = np.short)
    cdef short[:] v_view = v_

    for i in range(n):
        v_view[i] = v_view[i] + val
    return v_

cdef double[:] add_double(double[:] v, double val):

    cdef short n = v.shape[0]
    cdef cnp.ndarray v_ = np.zeros(n, dtype = np.float64)
    cdef double[:] v_view = v_

    for i in range(n):
        v_view[i] = v[i] + val
    return v_

cdef double[:] power_double(double[:] v, double e):

    cdef short n = v.shape[0]
    cdef cnp.ndarray v_ = np.zeros(n)
    cdef double[:] v_view = v_

    for i in range(n):
        v_view[i] = v_view[i] ** e
    return v_

cdef double[:] power_array(double[:] v, short[:] e):

    cdef short n = v.shape[0]
    cdef cnp.ndarray v_ = np.zeros(n)
    cdef double[:] v_view = v_

    for i in range(n):
        v_view[i] = pow(v_view[i],  <double>e[i])
    return v_

cdef double[:] elementwise_prod(double[:] v1, double[:] v2):

    cdef short n = v1.shape[0]
    cdef cnp.ndarray v_ = np.zeros(n)
    cdef double[:] v_view = v_

    for i in range(n):
        v_view[i] = v1[i] * v2[i]
    return v_

cdef double prod(double[:] v):

    cdef short n = v.shape[0]

    p = 1.0
    for i in range(n):
        p *= v[i]
    return p

cdef double norm(double[:] v):

    cdef short n = v.shape[0]

    s = 0.0

    for i in range(n):
        s += v[i] ** 2.0
    return sqrt(s)

cdef double dot(double[:] v1, double[:] v2):

    cdef short n1 = v1.shape[0]

    d = 0.0

    for i in range(n1):
        d += v1[i] * v2[i]
    return d

cpdef double[:] linear_combination(double[:] v1, double a1, double[:] v2, double a2):
    cdef short n1 = v1.shape[0]
    cdef cnp.ndarray v_ = np.zeros(n1)
    cdef double[:] v_view = v_

    for i in range(n1):
        v_view[i] = a1 * v1[i] + a2 * v2[i]
    return v_

cpdef double[:] array_atanh(double[:] v):
    cdef short n = v.shape[0]

    for i in range(n):
        v[i] = atanh(v[i])
    return v

