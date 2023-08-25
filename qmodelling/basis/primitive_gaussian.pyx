# cython: cdivision=True
from cpython cimport array
import array
import numpy as np

cimport numpy as cnp
cnp.import_array()

from cython.cimports.libc.math import sqrt, pow, remainder, exp

ctypedef cnp.float64_t F64
cdef double pi = np.pi

cdef class PrimitiveGaussian:

    cdef double alpha
    cdef double A
    cdef cnp.ndarray atom_position
    cdef short[:] angular_exponents

    cdef PrimitiveGaussian basis2

    def __cinit__(self, double alpha, cnp.ndarray[F64, ndim=2] atom_position, short ex, short ey, short ez):
        self.alpha = alpha
        self.atom_position = atom_position
        self.angular_exponents = array.array('h', [ex, ey, ez])

    def __init__(self, alpha, atom_position, short ex, short ey, short ez):
        self.A = self.normalization_constant()

    cdef get_attributes(self):
        return self.alpha, self.atom_position, self.angular_exponents, self.A

    cpdef double normalization_constant(self):
        alpha = self.alpha
        ex = self.angular_exponents[0]
        ey = self.angular_exponents[1]
        ez = self.angular_exponents[2]

        Ix = gaussian_integral(2.0*alpha, 2*ex)
        Iy = gaussian_integral(2.0*alpha, 2*ey)
        Iz = gaussian_integral(2.0*alpha, 2*ez)

        return 1.0 / sqrt(Ix * Iy * Iz)

    cpdef double overlap_int(self, basis2):
        """overlap integral over each coordinate"""
        cdef double Ea, A1, A2, a1, a2, dX12
        cdef double I = 1.0
        cdef short coo

        a1, X1, E1, A1 = self.get_attributes()
        a2, X2, E2, A2 = basis2.get_attributes()

        a1p2 = a1 + a2

        dX12 = np.linalg.norm(X1 - X2)

        Ea = exp(-a1 * a2 / a1p2 * pow(dX12, 2.0))

        for coo in range(3):
            I *= overlap_int_coo(X1, X2, a1, a2, E1, E2, coo)
        return A1 * A2 * Ea * I

    cpdef double kinetic_int(self, basis2):
        a1, X1, E1, A1 = self.get_attributes()
        a2, X2, E2, A2 = basis2.get_attributes()

        a1p2 = a1 + a2

        Ea = m.exp(-a1 * a2 / a1p2 * np.linalg.norm(X1 - X2) ** 2.0)

        I = 0.0
        for coo in range(3):
            I += kinetic_int_coo(X1, X2, a1, a2, E1, E2, coo)
        return A1 * A2 * Ea * I

#############
# HELPER FUNCTIONS
#############
cpdef double gaussian_integral(double alpha, short n):
    """integral over R of x^n exp(-alpha x^2)"""
    cdef double odd_prod = 1.0
    cdef short nn = n // 2
    cdef double ii
    cdef short i

    if n % 2 == 1:
        return 0.0

    for i in range(nn):    
        ii = i
        odd_prod *= 2.0 * ii + 1.0

    return sqrt(pi/alpha) / pow(2.0 * alpha, nn) * odd_prod

cdef double custom_comb(short n, short k):
    """custom combinatory coefficient"""
    cdef short result = 1
    cdef short i 
    if k > n - k:
        k = n - k
    for i in range(k):
        result *= n - i
        result //= i + 1
    return <double>result

cpdef double overlap_int_coo(cnp.ndarray[F64, ndim=2] X1,cnp.ndarray[F64, ndim=2] X2, double  a1, double a2, short[:] E1, short[:] E2, short coo):
    """overlap integral over a coordinate"""
    cdef double a1p2 = a1 + a2
    cdef double x1 = X1[0, coo]
    cdef double x2 = X2[0, coo]
    cdef double x_bar = (x1 * a1 + x2 * a2) / a1p2
    cdef double I = 0.0
    cdef short e1 = E1[coo]
    cdef short e2 = E2[coo]

    cdef short i, j

    for i in range(e1 + 1):
        for j in range(e2 + 1):
            I += (
                custom_comb(e1, i)
                * custom_comb(e2, j)
                * (x_bar - x1) ** (e1 - i)
                * (x_bar - x2) ** (e2 - j)
                * gaussian_integral(a1 + a2, i + j)
            )
    return I

