from cpython cimport array
import array
import numpy as np

cimport numpy as cnp
cnp.import_array()

from qmodelling.c_ops.math cimport sqrt, pow, exp
from qmodelling.constants cimport pi, dtype
from qmodelling.integral.quadrature.primitive_gaussian_quadrature cimport ngauss, ngauss_3d, get_Chebyshev_01, get_Chebyshev_3d

cdef class PrimitiveGaussian:

    #cdef double[:, :] atom_position
    #cdef double[:, :, :] Chebyshev_abscissa_3d
    #cdef short[3] angular_exponents

    cdef double[:, :] Chebyshev_01
    cdef double[:] Chebyshev_weights_3d

    cdef cnp.ndarray atom_position
    cdef cnp.ndarray Chebyshev_abscissa_3d
    cdef cnp.ndarray angular_exponents

    cdef double alpha
    cdef double A

    cdef short ngauss
    cdef short ngauss_3d

    def __cinit__(self, double alpha, cnp.ndarray[dtype, ndim = 2] atom_position, short ex, short ey, short ez):
        self.alpha = alpha
        self.atom_position = atom_position
        #self.angular_exponents = array.array('h', [ex, ey, ez])
        self.angular_exponents = np.array([ex, ey, ez])
        self.Chebyshev_01 = get_Chebyshev_01()

        data_3d = get_Chebyshev_3d()
        self.Chebyshev_weights_3d = data_3d[0]
        self.Chebyshev_abscissa_3d = data_3d[1]

        self.ngauss = ngauss
        self.ngauss_3d = ngauss_3d

    def __init__(self, alpha, atom_position, short ex, short ey, short ez):
        self.A = self.normalization_constant()

    cpdef get_attributes(self):
        return self.alpha, self.atom_position, self.angular_exponents, self.A

    cpdef double normalization_constant(self):
        cdef double alpha_2 = 2.0*self.alpha
        cdef short ex = self.angular_exponents[0]
        cdef short ey = self.angular_exponents[1]
        cdef short ez = self.angular_exponents[2]

        Ix = gaussian_integral(alpha_2, 2*ex)
        Iy = gaussian_integral(alpha_2, 2*ey)
        Iz = gaussian_integral(alpha_2, 2*ez)

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
        cdef double A1, A2, a1, a2, a1p2, Ea
        cdef short coo, e2

        a1, X1, E1, A1 = self.get_attributes()
        a2, X2, E2, A2 = basis2.get_attributes()

        a1p2 = a1 + a2

        Ea = exp(-a1 * a2 / a1p2 * pow(np.linalg.norm(X1 - X2), 2.0))

        I = 0.0
        for coo in range(3):
            e2 = E2[coo]
            if e2 == 0:
                I1 = -a2 * (
                    2.0 * a2 * overlap_int_coo(X1, X2, a1, a2, E1, E2 + 2, coo)
                    - overlap_int_coo(X1, X2, a1, a2, E1, E2, coo)
                )
        
            elif e2 == 1:
                I1 = -a2 * (
                    -3.0 * overlap_int_coo(X1, X2, a1, a2, E1, E2, coo)
                    + 2.0 * a2 * overlap_int_coo(X1, X2, a1, a2, E1, E2 + 2, coo)
                )
            else:
                Fkp2 = 4.0 * a2 ** 2.0 * overlap_int_coo(X1, X2, a1, a2, E1, E2 + 2, coo)
                Fkm2 = e2 * (e2 - 1.0) * overlap_int_coo(X1, X2, a1, a2, E1, E2 - 2, coo)
                Fk = -2.0 * a2 * (2.0 * e2 + 1.0) * overlap_int_coo(X1, X2, a1, a2, E1, E2, coo)
        
                I1 = -0.5 * (Fkp2 + Fkm2 + Fk)
        
            I2 = overlap_int_coo(X1, X2, a1, a2, E1, E2, (coo + 1) % 3)
            I3 = overlap_int_coo(X1, X2, a1, a2, E1, E2, (coo + 2) % 3)
            I += I1 * I2 * I3

        return A1 * A2 * Ea * I

    cpdef double electron_proton_int(self, basis2, cnp.ndarray[dtype, ndim=2] R, short Z):
        """electron-proton integral via Chebyshev-Gauss Quadrature
        quadrature points are scaled to the interval [0,1]
        """
        cdef double A1, A2, a1, a2, a1p2, Ea
        cdef double tk, wk 

        a1, X1, E1, A1 = self.get_attributes()
        a2, X2, E2, A2 = basis2.get_attributes()

        a1p2 = a1 + a2

        Ea = exp(-a1 * a2 / a1p2 * pow(np.linalg.norm(X1 - X2), 2.0))

        # barycenter of the two Gaussians
        P = (a1 * X1 + a2 * X2) / (a1p2)

        # squared distance between P and R (proton position)
        PR2 = pow(np.linalg.norm(P - R), 2.0)

        # quadrature loop
        I = 0.0
        #for wk, tk in Chebyshev_01:
        for i in range(self.ngauss):
            wk = self.Chebyshev_01[i, 0]
            tk = self.Chebyshev_01[i, 1]
            tk2 = pow(tk,2.0)
            I_tmp = 1.0
            for coo in range(3):
                I_tmp *= overlap_int_coo(
                    P, R, a1=a1p2, a2=tk2 * a1p2 / (1.0 - tk2), E1=E1, E2=E2, coo=coo
                )
            I += wk / pow(1.0 - tk2, (3.0 / 2.0)) * exp(-a1p2 * tk2 * PR2) * I_tmp

        return -2.0 * A1 * A2 * Ea * Z * sqrt(a1p2 / pi) * I

    cpdef double electron_electron_int(self, basis2, basis3, basis4):
        """electron-electron integral in chemist notation (ab|cd) = integral a(R1) b(R1) |R1-R2|^(-1) c(R2) d(R2)
        """

        cdef cnp.ndarray[dtype, ndim = 2] r2
        cdef double Ea, AA, p, E12, E34
        cdef double A1, A2, a1, a2, a1p2
        cdef double A3, A4, a3, a4, a3p2
        cdef double X12_34
        cdef double tk, wkt 
        cdef double tk2, wk2
        cdef double I, I_tmp_t, I_tmp_R2, I_tmp_R1
        cdef short k, k2

        # gaussian parameters
        a1, X1, E1, A1 = self.get_attributes()
        a2, X2, E2, A2 = basis2.get_attributes()
        a3, X3, E3, A3 = basis3.get_attributes()
        a4, X4, E4, A4 = basis4.get_attributes()

        AA = A1 * A2 * A3 * A4
    
        a1p2 = a1 + a2
        a3p4 = a3 + a4
        p = a1p2 * a3p4 / (a1p2 + a3p4)
    
        E12 = exp(-a1 * a2 / a1p2 * np.linalg.norm(X1 - X2) ** 2.0)
        E34 = exp(-a3 * a4 / a3p4 * np.linalg.norm(X3 - X4) ** 2.0)
    
        X12 = (a1 * X1 + a2 * X2) / (a1p2)
        X34 = (a3 * X3 + a4 * X4) / (a3p4)
        X12_34 = np.linalg.norm(X12 - X34) ** 2.0
    
        I = 0.0
    
        # quadrature loop over Gauss transformation integration variable: t
        # for wkt, tk in Chebyshev_quadrature_points_01:
    
        for k in range(self.ngauss):
            wkt = self.Chebyshev_01[k, 0]
            tk  = self.Chebyshev_01[k, 1]
    
            tk2 = tk**2.0
            I_tmp_t = 0.0
            I_tmp_R2 = 0.0
            R2_bar = (a3p4 * X34 + (a1p2 * tk2 * p) / (a1p2 + tk2 * (p - a1p2)) * X12) / (
                a3p4 + (a1p2 * tk2 * p) / (a1p2 + tk2 * (p - a1p2))
            )
    
            # quadrature loop over coordinates R2 = (x2 y2 z2)
            for k2 in range(self.ngauss_3d):
                r2  = self.Chebyshev_abscissa_3d[k2]
                wk2 = self.Chebyshev_weights_3d[k2]
    
                R2 = np.arctanh(r2)
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
    
                I_tmp_R2 += wk2 * np.prod((R2 - X3) ** E3 * (R2 - X4) ** E4) * exp(
                        -a1p2
                        * a3p4
                        / (a1p2 + tk2 * (p - a1p2))
                        * pow(np.linalg.norm(R2 - R2_bar), 2.0)
                    ) / np.prod(1.0 - r2**2.0) * I_tmp_R1
                
    
            I_tmp_t += I_tmp_R2
    
            I += wkt / (1.0 - tk2) ** (3.0 / 2.0) * exp(-p * tk2 * X12_34) * I_tmp_t
    
        return 2.0 * AA * E12 * E34 * sqrt(p / pi) * I

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

cpdef double overlap_int_coo(cnp.ndarray[dtype, ndim=2] X1, cnp.ndarray[dtype, ndim=2] X2, double a1, double a2, E1, E2, short coo):
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

