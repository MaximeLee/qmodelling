cdef double odd_prod(short n):
    cdef double p = 1.0
    cdef short i 
    for i in range(n):
        p *= 2.0 * <double>i + 1.0
    return p

