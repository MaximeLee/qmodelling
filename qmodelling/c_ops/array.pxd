cdef double norm(double[:])

cdef double[:] sub(double[:] v1, double[:] v2)
cdef short[:] add_short(short[:] , short)
cdef double[:] add_double(double[:] , double)
cdef double dot(double[:], double[:])
cdef double prod(double[:])
cdef double[:] power_double(double[:] v, double e)
cdef double[:] power_array(double[:], short[:])
cdef double[:] elementwise_prod(double[:], double[:])

cpdef double[:] linear_combination(double[:] v1, double a1, double[:] v2, double a2)

cpdef double[:] array_atanh(double[:])
