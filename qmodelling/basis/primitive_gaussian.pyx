# cython: cdivision=True
from cpython cimport array
import array
import numpy as np

cimport numpy as cnp
cnp.import_array()

from qmodelling.c_ops.math cimport sqrt, pow, exp, odd_prod
from qmodelling.c_ops.array cimport sub, add_short, add_double, norm, dot, power_double, linear_combination, array_atanh, power_array, elementwise_prod, prod
from qmodelling.constants cimport pi, dtype
from qmodelling.integral.quadrature.primitive_gaussian_quadrature cimport ngauss as ng
from qmodelling.integral.quadrature.primitive_gaussian_quadrature cimport ngauss_3d as ng3d
from qmodelling.integral.quadrature.primitive_gaussian_quadrature cimport get_Chebyshev_01, get_Chebyshev_3d

from libc.math cimport isnan

cdef short ngauss    = ng   
cdef short ngauss_3d = ng3d  

cdef Chebyshev_01 = get_Chebyshev_01()
data_3d = get_Chebyshev_3d()
cdef Chebyshev_weights_3d = data_3d[0]
cdef Chebyshev_abscissa_3d = data_3d[1]

#Chebyshev_01 : np.array() = get_Chebyshev_01()
#Chebyshev_weights_3d : np.array() = data_3d[0]
#Chebyshev_abscissa_3d : np.array() = data_3d[1]

cdef class PrimitiveGaussian:

    cdef double[:] atom_position
    cdef short[3] angular_exponents

    cdef double alpha
    cdef double A

    #cdef cnp.ndarray atom_position
    #cdef cnp.ndarray Chebyshev_abscissa_3d
    #cdef cnp.ndarray angular_exponents

    def __cinit__(self, double alpha, double[:] atom_position, short ex, short ey, short ez):
        self.alpha = alpha
        self.atom_position = atom_position
        self.angular_exponents = array.array('h', [ex, ey, ez])

    def __init__(self, alpha, atom_position, short ex, short ey, short ez):
        self.A = self.normalization_constant()

    cpdef get_attributes(self):
        cdef double a = self.alpha
        cdef double[:] pos = self.atom_position
        cdef short[:] ang = self.angular_exponents
        cdef double AA = self.A
        return a, pos, ang, AA

    cpdef double normalization_constant(self):
        cdef double alpha_2 = 2.0*self.alpha
        cdef short ex = self.angular_exponents[0]
        cdef short ey = self.angular_exponents[1]
        cdef short ez = self.angular_exponents[2]

        Ix = gaussian_integral(alpha_2, 2*ex)
        Iy = gaussian_integral(alpha_2, 2*ey)
        Iz = gaussian_integral(alpha_2, 2*ez)

        return 1.0 / sqrt(Ix * Iy * Iz)

    cpdef double overlap_int(self, PrimitiveGaussian basis2):
        """overlap integral over each coordinate"""
        cdef double[:] X1, X2
        cdef short[:] E1, E2
        cdef double Ea, A1, A2, a1, a2
        cdef double I = 1.0
        cdef short coo

        a1, X1, E1, A1 = self.get_attributes()
        a2, X2, E2, A2 = basis2.get_attributes()

        a1p2 = a1 + a2

        dX12 = sub(X1, X2)

        Ea = exp(-a1 * a2 / a1p2 * dot(dX12, dX12))

        for coo in range(3):
            I *= overlap_int_coo(X1, X2, a1, a2, E1, E2, coo)
        return A1 * A2 * Ea * I

    cpdef double kinetic_int(self, PrimitiveGaussian basis2):
        cdef double[:] X1
        cdef double[:] X2
        cdef short[:] E2
        cdef short[:] E1
        cdef double A1, A2, a1, a2
        cdef short coo, e2

        a1, X1, E1, A1 = self.get_attributes()
        a2, X2, E2, A2 = basis2.get_attributes()

        a1p2 = a1 + a2

        dX12 = sub(X1, X2)

        Ea = exp(-a1 * a2 / a1p2 * dot(dX12, dX12))

        I = 0.0
        for coo in range(3):
            e2 = E2[coo]
            if e2 == 0:
                I1 = -a2 * (
                    2.0 * a2 * overlap_int_coo(X1, X2, a1, a2, E1, add_short(E2, 2), coo)
                    - overlap_int_coo(X1, X2, a1, a2, E1, E2, coo)
                )
        
            elif e2 == 1:
                I1 = -a2 * (
                    -3.0 * overlap_int_coo(X1, X2, a1, a2, E1, E2, coo)
                    + 2.0 * a2 * overlap_int_coo(X1, X2, a1, a2, E1, add_short(E2, 2), coo)
                )
            else:
                Fkp2 = 4.0 * a2 ** 2.0 * overlap_int_coo(X1, X2, a1, a2, E1, add_short(E2,  2), coo)
                Fkm2 = e2 * (e2 - 1.0) * overlap_int_coo(X1, X2, a1, a2, E1, add_short(E2, -2), coo)
                Fk = -2.0 * a2 * (2.0 * e2 + 1.0) * overlap_int_coo(X1, X2, a1, a2, E1, E2, coo)
        
                I1 = -0.5 * (Fkp2 + Fkm2 + Fk)
        
            I2 = overlap_int_coo(X1, X2, a1, a2, E1, E2, (coo + 1) % 3)
            I3 = overlap_int_coo(X1, X2, a1, a2, E1, E2, (coo + 2) % 3)
            I += I1 * I2 * I3

        return A1 * A2 * Ea * I

    cpdef double electron_proton_int(self, basis2, double[:] R, short Z):
        """electron-proton integral via Chebyshev-Gauss Quadrature
        quadrature points are scaled to the interval [0,1]
        """
        cdef double A1, A2, a1, a2, a1p2, Ea
        cdef double tk, wk 

        a1, X1, E1, A1 = self.get_attributes()
        a2, X2, E2, A2 = basis2.get_attributes()

        a1p2 = a1 + a2
    
        dX12 = sub(X1, X2)

        Ea = exp(-a1 * a2 / a1p2 * dot(dX12, dX12))

        # barycenter of the two Gaussians
        #P = (a1 * X1 + a2 * X2) / (a1p2)
        P = linear_combination(X1, a1/a1p2, X2, a2/a1p2)

        # squared distance between P and R (proton position)
        PR = sub(P, R)
        PR2 = dot(PR, PR)

        # quadrature loop
        I = 0.0
        #for wk, tk in Chebyshev_01:
        for i in range(ngauss):
            wk = Chebyshev_01[i, 0]
            tk = Chebyshev_01[i, 1]
            tk2 = pow(tk,2.0)
            I_tmp = 1.0
            for coo in range(3):
                I_tmp *= overlap_int_coo(
                    P, R, a1=a1p2, a2=tk2 * a1p2 / (1.0 - tk2), E1=E1, E2=E2, coo=coo
                )
            I += wk / pow(1.0 - tk2, (3.0 / 2.0)) * exp(-a1p2 * tk2 * PR2) * I_tmp

        return -2.0 * A1 * A2 * Ea * Z * sqrt(a1p2 / pi) * I

    cpdef double electron_electron_int(self, PrimitiveGaussian basis2, PrimitiveGaussian basis3, PrimitiveGaussian basis4):
        """electron-electron integral in chemist notation (ab|cd) = integral a(R1) b(R1) |R1-R2|^(-1) c(R2) d(R2)
        """

        cdef double[::1] r2
        cdef double[:] X1, X2, X3, X4
        cdef short[:] E1, E2, E3, E4
        cdef double Ea, AA, p, E12, E34
        cdef double A1, A2, a1, a2, a1p2
        cdef double A3, A4, a3, a4, a3p2
        cdef double X12_34
        cdef double tk, wkt 
        cdef double tk2, wk2
        cdef double I, I_tmp_t, I_tmp_R2, I_tmp_R1
        #cdef short k, k2
        cdef Py_ssize_t k, k2

        # gaussian parameters
        a1, X1, E1, A1 = self.get_attributes()
        a2, X2, E2, A2 = basis2.get_attributes()
        a3, X3, E3, A3 = basis3.get_attributes()
        a4, X4, E4, A4 = basis4.get_attributes()

        AA = A1 * A2 * A3 * A4
    
        a1p2 = a1 + a2
        a3p4 = a3 + a4
        p = a1p2 * a3p4 / (a1p2 + a3p4)
    
        dX12 = sub(X1, X2)
        dX34 = sub(X3, X4)

        E12 = exp(-a1 * a2 / a1p2 * dot(dX12, dX12))
        E34 = exp(-a3 * a4 / a3p4 * dot(dX34, dX34))
    
        #X12 = (a1 * X1 + a2 * X2) / (a1p2)
        X12 = linear_combination(X1, a1/a1p2, X2, a2/a1p2)
        #X34 = (a3 * X3 + a4 * X4) / (a3p4)
        X34 = linear_combination(X3, a3/a3p4, X4, a4/a3p4)
        dX1234 = sub(X12, X34)
        X12_34 = dot(dX1234, dX1234)
    
        I = 0.0
    
        # quadrature loop over Gauss transformation integration variable: t
        for k in range(ngauss):
            wkt = Chebyshev_01[k, 0]
            tk  = Chebyshev_01[k, 1]
    
            tk2 = pow(tk,2.0)
            I_tmp_t = 0.0
            I_tmp_R2 = 0.0

            denom = a3p4 + (a1p2 * tk2 * p) / (a1p2 + tk2 * (p - a1p2))
            R2_bar = linear_combination(X34, a3p4/denom, X12, (a1p2 * tk2 * p) / (a1p2 + tk2 * (p - a1p2))/denom)

            # quadrature loop over coordinates R2 = (x2 y2 z2)
            for k2 in range(ngauss_3d):
                r2  = Chebyshev_abscissa_3d[k2,:]
                wk2 = Chebyshev_weights_3d[k2]

                R2 = array_atanh(r2)
                I_tmp_R1 = 1.0
                for coo in range(3):
                    I_tmp_R1 *= overlap_int_coo(
                        X12,
                        R2,
                        a1=a1p2,
                        a2=tk2 * p / (1.0 - tk2),
                        E1=E1,
                        E2=E2,
                        coo=coo,
                    )

                R2_X3 = power_array(sub(R2, X3), E3)
                R2_X4 = power_array(sub(R2, X4), E4)
                R2_X3_X4 = elementwise_prod(R2_X3, R2_X4)

                I_tmp_R2 += wk2 * prod(R2_X3_X4) * exp(
                        -a1p2
                        * a3p4
                        / (a1p2 + tk2 * (p - a1p2))
                        * dot(sub(R2, R2_bar), sub(R2, R2_bar))
                    ) / (-prod(add_double(power_double(r2,2.0), - 1.0))) * I_tmp_R1
                """
                I_tmp_R2 += wk2 * prod(R2_X3_X4) * exp(
                        -a1p2
                        * a3p4
                        / (a1p2 + tk2 * (p - a1p2))
                        * dot(sub(R2, R2_bar), sub(R2, R2_bar))
                    ) / (-prod(add_double(power_double(r2,2.0), - 1.0))) * I_tmp_R1
                """
                
            I_tmp_t += I_tmp_R2
    
            I += wkt / pow(1.0 - tk2, 3.0 / 2.0) * exp(-p * tk2 * X12_34) * I_tmp_t
    
        return 2.0 * AA * E12 * E34 * sqrt(p / pi) * I

#############
# HELPER FUNCTIONS
#############
cpdef double gaussian_integral(double alpha, short n):
    """integral over R of x^n exp(-alpha x^2)"""
    cdef short nn = n // 2
    cdef double ii
    cdef short i

    if n % 2 == 1:
        return 0.0

    return sqrt(pi/alpha) / pow(2.0 * alpha, nn) * odd_prod(nn)

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

cpdef double overlap_int_coo(double[:] X1, double[:] X2, double a1, double a2, short[:] E1, short[:] E2, short coo):
    """overlap integral over a coordinate"""
    cdef double a1p2 = a1 + a2
    cdef double x1 = X1[coo]
    cdef double x2 = X2[coo]
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

#######################
# quadrature test functions 
#######################
cpdef double integrate_3d(fun):

    I = 0.0
    for k in range(ngauss_3d):
        rk = Chebyshev_abscissa_3d[k,:]
        wk = Chebyshev_weights_3d[k]

        Rk = array_atanh(rk)

        I += wk * fun(Rk) / (-prod(add_double(power_double(rk,2.0), - 1.0)))

    return I

