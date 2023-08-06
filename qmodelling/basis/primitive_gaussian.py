"""Primitive Gaussian module"""
import copy as cp
import math as m
import numpy as np
from scipy import special
from qmodelling.basis.basis_function import BasisFunction
from qmodelling.integral.quadrature import get_quadrature_points

pi = np.pi


def boys(x):
    """boys function used for one-two electron integrals"""
    n = 0
    if x == 0:
        return 1.0 / (2 * n + 1)
    return (
        special.gammainc(n + 0.5, x)
        * special.gamma(n + 0.5)
        * (1.0 / (2 * x ** (n + 0.5)))
    )


class PrimitiveGaussian(BasisFunction):
    """Primitive Gaussian class

    only considering 1s orbital atm

    alpha (float) : parameter of the gaussian function
    atom_position (np.array)
    ex, ey, ez (integer): exponent of angular momentum l = ex + ey + ez
    """

    # pts on [-1,1]
    Chebyshev_quadrature_points = get_quadrature_points(
        n=30, quadrature_type="gauss_chebyshev_2"
    )

    # pts on [0,1]
    Chebyshev_quadrature_points_01 = cp.deepcopy(Chebyshev_quadrature_points)
    Chebyshev_quadrature_points_01[:, 1] = (
        Chebyshev_quadrature_points_01[:, 1] + 1.0
    ) / 2.0
    # sum weights = 1
    Chebyshev_quadrature_points_01[:, 0] /= 2.0

    def __init__(self, alpha, atom_position, ex=0, ey=0, ez=0):
        super().__init__()

        # Gaussian parameters
        self.alpha = alpha
        self.atom_position = atom_position

        # exponents of the angular momentum
        self.ex = ex
        self.ey = ey
        self.ez = ez
        self.angular_exponents = np.array([ex, ey, ey])

        # normalizing constant
        self.A = PrimitiveGaussian.normalization_constant(alpha, ex, ey, ez)

    def get_attributes(self):
        """get attributes"""
        return (
            self.alpha,
            self.atom_position,
            self.angular_exponents,
            self.A,
        )

    @classmethod
    def integral(cls, alpha, n):
        """integral over R of x^n exp(-alpha x^2)"""
        # odd exponent
        if n % 2 == 1:
            return 0.0

        # even exponent
        nn = float(n // 2)

        return (
            m.sqrt(pi / alpha)
            / (2.0 * alpha) ** nn
            * np.prod(np.prod(2.0 * np.arange(nn) + 1))
        )

    @classmethod
    def normalization_constant(cls, alpha, ex, ey, ez):
        """computing the normalisation constant of the primitive Gaussian"""

        # integral in x
        int_x = PrimitiveGaussian.integral(2.0 * alpha, 2 * ex)

        # PrimitiveGaussian.integralgral in y
        int_y = PrimitiveGaussian.integral(2.0 * alpha, 2 * ey)

        # PrimitiveGaussian.integralgral in z
        int_z = PrimitiveGaussian.integral(2.0 * alpha, 2 * ez)

        return 1.0 / m.sqrt(int_x * int_y * int_z)

    def __call__(self, x):
        """forward method needed for integral quadrature"""
        dxx2 = np.sum((x - self.atom_position) ** 2, axis=1, keepdims=True)
        return (
            self.A
            * np.exp(-self.alpha * dxx2)
            * np.prod(x**self.angular_exponents, axis=1, keepdims=True)
        )

    @classmethod
    def overlap_int_coo(cls, X1, X2, a1, a2, E1, E2, coo):
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
                    m.comb(e1, i)
                    * m.comb(e2, j)
                    * (x_bar - x1) ** (e1 - i)
                    * (x_bar - x2) ** (e2 - j)
                    * PrimitiveGaussian.integral(a1 + a2, i + j)
                )
        return I

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
            I *= PrimitiveGaussian.overlap_int_coo(X1, X2, a1, a2, E1, E2, coo)

        return A1 * A2 * Ea * I

    @classmethod
    def kinetic_int_coo(cls, X1, X2, a1, a2, E1, E2, coo):
        """kinetic integral comouted with the Binomial theorem for each coordinate
        coo (int) : coordinate index to integrate over (x:0, y:1, z:2)
        """

        e2 = E2[coo]
        if e2 == 0:
            I1 = -a2 * (
                2.0
                * a2
                * PrimitiveGaussian.overlap_int_coo(X1, X2, a1, a2, E1, E2 + 2, coo)
                - PrimitiveGaussian.overlap_int_coo(X1, X2, a1, a2, E1, E2, coo)
            )

        elif e2 == 1:
            I1 = -a2 * (
                -3.0 * PrimitiveGaussian.overlap_int_coo(X1, X2, a1, a2, E1, E2, coo)
                + 2.0
                * a2
                * PrimitiveGaussian.overlap_int_coo(X1, X2, a1, a2, E1, E2 + 2, coo)
            )
        else:
            Fkp2 = (
                4.0
                * a2**2.0
                * PrimitiveGaussian.overlap_int_coo(X1, X2, a1, a2, E1, E2 + 2, coo)
            )
            Fkm2 = (
                e2
                * (e2 - 1.0)
                * PrimitiveGaussian.overlap_int_coo(X1, X2, a1, a2, E1, E2 - 2, coo)
            )
            Fk = (
                -2.0
                * a2
                * (2.0 * e2 + 1.0)
                * PrimitiveGaussian.overlap_int_coo(X1, X2, a1, a2, E1, E2, coo)
            )

            I1 = -0.5 * (Fkp2 + Fkm2 + Fk)

        I2 = PrimitiveGaussian.overlap_int_coo(X1, X2, a1, a2, E1, E2, (coo + 1) % 3)
        I3 = PrimitiveGaussian.overlap_int_coo(X1, X2, a1, a2, E1, E2, (coo + 2) % 3)

        return I1 * I2 * I3

    def kinetic_int(self, basis2):
        """kinetic integral"""
        a1, X1, E1, A1 = self.get_attributes()
        a2, X2, E2, A2 = basis2.get_attributes()

        a1p2 = a1 + a2

        Ea = m.exp(-a1 * a2 / a1p2 * np.linalg.norm(X1 - X2) ** 2.0)

        I = 0.0
        for coo in range(3):
            I += PrimitiveGaussian.kinetic_int_coo(X1, X2, a1, a2, E1, E2, coo)
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
        for wk, tk in PrimitiveGaussian.Chebyshev_quadrature_points_01:
            tk2 = tk**2.0
            I_tmp = 1.0
            for coo in range(3):
                I_tmp *= PrimitiveGaussian.overlap_int_coo(
                    P, R, a1=a1p2, a2=tk2 * a1p2 / (1.0 - tk2), E1=E1, E2=E2, coo=coo
                )
            I += wk / (1.0 - tk2) ** (3.0 / 2.0) * m.exp(-a1p2 * tk2 * PR2) * I_tmp

        return -2.0 * A1 * A2 * Ea * Z * m.sqrt(a1p2 / pi) * I

    def electron_electron_int(self, basis2, basis3, basis4):
        """electron-electron integral in chemist notation (ab|cd) = integral a(R1) b(R1) |R1-R2|^(-1) c(R2) d(R2)"""

        a1, X1, E1, A1 = self.get_attributes()
        a2, X2, E2, A2 = basis2.get_attributes()
        a3, X3, E3, A3 = basis3.get_attributes()
        a4, X4, E4, A4 = basis4.get_attributes()

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
        for wkt, tk in PrimitiveGaussian.Chebyshev_quadrature_points_01:
            tk2 = tk**2.0
            I_tmp_t = 0.0
            I_tmp_R2 = 0.0
            R2_bar = (
                a3p4 * X34 + (a1p2 * tk2 * p) / (a1p2 + tk2 * (p - a1p2)) * X12
            ) / (a3p4 + (a1p2 * tk2 * p) / (a1p2 + tk2 * (p - a1p2)))

            # quadrature loop over coordinates R2 = (x2 y2 z2)
            for wkX2, x2 in PrimitiveGaussian.Chebyshev_quadrature_points:
                for wkY2, y2 in PrimitiveGaussian.Chebyshev_quadrature_points:
                    for wkZ2, z2 in PrimitiveGaussian.Chebyshev_quadrature_points:
                        r2 = np.array([[x2,y2,z2]])
                        R2 = np.arctanh(r2)
                        I_tmp_R1 = 1.0
                        for coo in range(3):
                            I_tmp_R1 *= PrimitiveGaussian.overlap_int_coo(
                                X12,
                                R2,
                                a1=a1p2,
                                a2=tk2 * p / (1.0 - tk2),
                                E1=E1,
                                E2=E2,
                                coo=coo,
                            )
        
                        I_tmp_R2 += (
                            wkX2 * wkY2 * wkZ2
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


#        A1234 = self.A * basis2.A * basis3.A * basis4.A
#
#        x1 = self.atom_position
#        x2 = basis2.atom_position
#        x3 = basis3.atom_position
#        x4 = basis4.atom_position
#
#        a1 = self.alpha
#        a2 = basis2.alpha
#        a3 = basis3.alpha
#        a4 = basis4.alpha
#
#        a12 = a1 * a2
#        ap12 = a1 + a2
#        x12 = x1 - x2
#        x12_2 = np.inner(x12, x12)
#        e12 = m.exp(-a12 / ap12 * x12_2)
#
#        a43 = a4 * a3
#        ap43 = a4 + a3
#        x43 = x4 - x3
#        x43_2 = np.inner(x43, x43)
#        e43 = m.exp(-a43 / ap43 * x43_2)
#
#        ap1234 = ap12 + ap43
#
#        x43_mean = (a4 * x4 + a3 * x3) / (ap43)
#        x12_mean = (a1 * x1 + a2 * x2) / (ap12)
#        x12_43 = x12_mean - x43_mean
#        Q2 = np.inner(x12_43, x12_43)
#
#        return (
#            A1234
#            * 2.0
#            * pi**2
#            / (ap12 * ap43)
#            * e43
#            * e12
#            * m.sqrt(pi / ap1234)
#            * boys((ap43 * ap12 / ap1234) * Q2)
#        )
