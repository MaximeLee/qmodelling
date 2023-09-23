"""Primitive Gaussian module"""
import copy as cp
import math as m
import numpy as np
from numba import jit, prange
from numba.experimental import jitclass

from qmodelling.integral.primitive_gaussian_quadrature import Chebyshev_quadrature_points_01, Chebyshev_abscissa_3d, Chebyshev_weights_3d
from qmodelling.constants import pi
from qmodelling.config import dtype_real, dtype_int, dtype_int_numba, dtype_real_numba

spec = [
    ('atom_position', dtype_real_numba[:, :]),
    ('alpha', dtype_real_numba), 
    ('angular_exponents', dtype_int_numba[:]), 
    ('A', dtype_real_numba), 
]

@jitclass(spec)
class PrimitiveGaussian:
    """Primitive Gaussian class

    only considering 1s orbital atm

    alpha (float) : parameter of the gaussian function
    atom_position (np.array)
    ex, ey, ez (integer): exponent of angular momentum l = ex + ey + ez
    """

    def __init__(self, alpha, atom_position=None, ex=0, ey=0, ez=0):
        self.atom_position = atom_position

        # Gaussian parameters
        self.alpha = alpha

        # exponents of the angular momentum
        self.angular_exponents = np.array([ex, ey, ez], dtype = dtype_int)

        # normalizing constant
        self.A = normalization_constant(alpha, ex, ey, ez)

    def __eq__(self, other):
        self_attributes = self.get_attributes()
        other_attributes = other.get_attributes()

        is_equal = np.all(self_attributes == other_attributes)
        return is_equal

    def get_attributes(self):
        """get attributes"""
        return (
            self.alpha,
            self.atom_position,
            self.angular_exponents,
            self.A,
        )

    #def __call__(self, x):
    def call(self, x):
        """forward method needed for integral quadrature"""
        dxx2 = np.sum((x - self.atom_position) ** 2, axis=1, keepdims=True)
        return (
            self.A
            * np.exp(-self.alpha * dxx2)
            * np.prod(x**self.angular_exponents, axis=1, keepdims=True)
        )

    def overlap_int(self, basis2):
        """double overlap integral"""
        a1, X1, angular_exponents1, A1 = self.get_attributes()
        a2, X2, angular_exponents2, A2 = basis2.get_attributes()

        a1p2 = a1 + a2

        Ea = m.exp(-a1 * a2 / a1p2 * np.linalg.norm(X1 - X2) ** 2.0)

        # computing integral over x, y, z
        I = 1.0
        for coo in range(3):
            E1 = angular_exponents1
            E2 = angular_exponents2
            I *= overlap_int_coo(X1, X2, a1, a2, E1, E2, coo)

        return A1 * A2 * Ea * I

    def kinetic_int(self, basis2):
        """kinetic integral"""
        a1, X1, E1, A1 = self.get_attributes()
        a2, X2, E2, A2 = basis2.get_attributes()

        a1p2 = a1 + a2

        Ea = m.exp(-a1 * a2 / a1p2 * np.linalg.norm(X1 - X2) ** 2.0)

        I = 0.0
        for coo in range(3):
            I += kinetic_int_coo(X1, X2, a1, a2, E1, E2, coo)
        return A1 * A2 * Ea * I

    def electron_proton_int(self, basis2, R, Z):
        """electron-proton integral via Chebyshev-Gauss Quadrature
        quadrature points are scaled to the interval [0,1]
        """
        a1, X1, E1, A1 = self.get_attributes()
        a2, X2, E2, A2 = basis2.get_attributes()

        a1p2 = a1 + a2

        Ea = m.exp(-a1 * a2 / a1p2 * np.linalg.norm(X1 - X2) ** 2.0)

        # barycenter of the two Gaussians
        P = (a1 * X1 + a2 * X2) / (a1p2)

        # squared distance between P and R (proton position)
        PR2 = np.linalg.norm(P - R) ** 2.0

        # quadrature loop
        I = 0.0
        for wk, tk in Chebyshev_quadrature_points_01:
            tk2 = tk**2.0
            I_tmp = 1.0
            for coo in range(3):
                I_tmp *= overlap_int_coo(
                    P, R, a1=a1p2, a2=tk2 * a1p2 / (1.0 - tk2), E1=E1, E2=E2, coo=coo
                )
            I += wk / (1.0 - tk2) ** (3.0 / 2.0) * m.exp(-a1p2 * tk2 * PR2) * I_tmp

        return -2.0 * A1 * A2 * Ea * Z * m.sqrt(a1p2 / pi) * I

    def electron_electron_int(self, basis2, basis3, basis4):
        """electron-electron integral in chemist notation (ab|cd) = integral a(R1) b(R1) |R1-R2|^(-1) c(R2) d(R2)
        wrapper to use jit
        """

        # gaussian parameters
        a1, X1, E1, A1 = self.get_attributes()
        a2, X2, E2, A2 = basis2.get_attributes()
        a3, X3, E3, A3 = basis3.get_attributes()
        a4, X4, E4, A4 = basis4.get_attributes()

        return electron_electron_int_jit(
            a1,
            X1,
            E1,
            A1,
            a2,
            X2,
            E2,
            A2,
            a3,
            X3,
            E3,
            A3,
            a4,
            X4,
            E4,
            A4,
        )


##################################
# Numba : function to be compiled just in time (JIT)
##################################
@jit("float64(float64, int64)",cache=True)
def integral(alpha: float, n: int):
    """integral over R of x^n exp(-alpha x^2)"""
    # odd exponent
    if n % 2 == 1:
        return 0.0

    # even exponent
    nn = float(n // 2)

    return m.sqrt(pi / alpha) / (2.0 * alpha) ** nn * np.prod(2.0 * np.arange(nn) + 1)


@jit(cache=True)
def normalization_constant(alpha, ex, ey, ez):
    """computing the normalisation constant of the primitive Gaussian"""

    # integral in x
    int_x = integral(2.0 * alpha, 2 * ex)

    # integralgral in y
    int_y = integral(2.0 * alpha, 2 * ey)

    # integralgral in z
    int_z = integral(2.0 * alpha, 2 * ez)

    return 1.0 / m.sqrt(int_x * int_y * int_z)


@jit(cache=True)
def custom_comb(n, k):
    if k > n - k:
        k = n - k
    result = 1
    for i in range(k):
        result *= n - i
        result //= i + 1
    return result


@jit(cache=True)
def overlap_int_coo(X1, X2, a1, a2, E1, E2, coo):
    """overlap integral comouted with the Binomial theorem for each coordinate
    coo (int) : coordinate index to integrate over (x:0, y:1, z:2)
    """

    e1 = E1[coo]
    e2 = E2[coo]
    x1 = X1[0, coo]
    x2 = X2[0, coo]
    a1p2 = a1 + a2
    x_bar = (x1 * a1 + x2 * a2) / a1p2
    I = 0.0
    for i in range(e1 + 1):
        for j in range(e2 + 1):
            I += (
                custom_comb(e1, i)
                * custom_comb(e2, j)
                * (x_bar - x1) ** (e1 - i)
                * (x_bar - x2) ** (e2 - j)
                * integral(a1 + a2, i + j)
            )
    return I


@jit(cache=True)
def kinetic_int_coo(X1, X2, a1, a2, E1, E2, coo):
    """kinetic integral comouted with the Binomial theorem for each coordinate
    coo (int) : coordinate index to integrate over (x:0, y:1, z:2)
    """

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
        Fkp2 = 4.0 * a2**2.0 * overlap_int_coo(X1, X2, a1, a2, E1, E2 + 2, coo)
        Fkm2 = e2 * (e2 - 1.0) * overlap_int_coo(X1, X2, a1, a2, E1, E2 - 2, coo)
        Fk = -2.0 * a2 * (2.0 * e2 + 1.0) * overlap_int_coo(X1, X2, a1, a2, E1, E2, coo)

        I1 = -0.5 * (Fkp2 + Fkm2 + Fk)

    I2 = overlap_int_coo(X1, X2, a1, a2, E1, E2, (coo + 1) % 3)
    I3 = overlap_int_coo(X1, X2, a1, a2, E1, E2, (coo + 2) % 3)

    return I1 * I2 * I3


#@jit(parallel=True, cache=True)
@jit(cache=True, nopython=True, nogil=True)
def electron_electron_int_jit(
    a1,
    X1,
    E1,
    A1,
    a2,
    X2,
    E2,
    A2,
    a3,
    X3,
    E3,
    A3,
    a4,
    X4,
    E4,
    A4,
):
    AA = A1 * A2 * A3 * A4

    a1p2 = a1 + a2
    a3p4 = a3 + a4
    p = a1p2 * a3p4 / (a1p2 + a3p4)

    E12 = m.exp(-a1 * a2 / a1p2 * np.linalg.norm(X1 - X2) ** 2.0)
    E34 = m.exp(-a3 * a4 / a3p4 * np.linalg.norm(X3 - X4) ** 2.0)

    X12 = (a1 * X1 + a2 * X2) / (a1p2)
    X34 = (a3 * X3 + a4 * X4) / (a3p4)
    X12_34 = np.linalg.norm(X12 - X34) ** 2.0

    I = 0.0

    # quadrature loop over Gauss transformation integration variable: t
    # for wkt, tk in Chebyshev_quadrature_points_01:
    nk = len(Chebyshev_quadrature_points_01)
    nk2 = len(Chebyshev_weights_3d)

    for k in range(nk):
        wkt, tk = Chebyshev_quadrature_points_01[k]

        tk2 = tk**2.0
        I_tmp_t = 0.0
        I_tmp_R2 = 0.0
        R2_bar = (a3p4 * X34 + (a1p2 * tk2 * p) / (a1p2 + tk2 * (p - a1p2)) * X12) / (
            a3p4 + (a1p2 * tk2 * p) / (a1p2 + tk2 * (p - a1p2))
        )

        # quadrature loop over coordinates R2 = (x2 y2 z2)
        #for k2 in prange(nk2):
        for k2 in range(nk2):
            wk2 = Chebyshev_weights_3d[k2]
            r2  = Chebyshev_abscissa_3d[k2]

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

            I_tmp_R2 += (
                wk2
                * np.prod((R2 - X3) ** E3 * (R2 - X4) ** E4)
                * m.exp(
                    -a1p2
                    * a3p4
                    / (a1p2 + tk2 * (p - a1p2))
                    * np.linalg.norm(R2 - R2_bar) ** 2.0
                )
                / np.prod(1.0 - r2**2.0)
                * I_tmp_R1
            )

        I_tmp_t += I_tmp_R2

        I += wkt / (1.0 - tk2) ** (3.0 / 2.0) * m.exp(-p * tk2 * X12_34) * I_tmp_t

    return 2.0 * AA * E12 * E34 * m.sqrt(p / pi) * I
