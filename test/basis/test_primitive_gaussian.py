"""Testing primitive Gaussians"""
from qmodelling.basis.primitive_gaussian import PrimitiveGaussian, integral, normalization_constant, Chebyshev_quadrature_points_01
import math as m
import numpy as np
import time
from constants import *
from utils import *

PG1_1s = PrimitiveGaussian(a1, X1)
A1 = PG1_1s.A
E1 = PG1_1s.angular_exponents

PG2_1s = PrimitiveGaussian(a2, X2)
A2 = PG2_1s.A
exponents2 = E2 = PG2_1s.angular_exponents

PG3_1s = PrimitiveGaussian(a3, X3)
A3 = PG3_1s.A
exponents3 = E3 = PG3_1s.angular_exponents

PG4_1s = PrimitiveGaussian(a4, X4)
A4 = PG4_1s.A
exponents4 = E4 = PG4_1s.angular_exponents

class TestPrimitiveGaussian:
    """testing functions of the PrimitiveGaussian class"""

    def test_quadrature_points(self):
        """test Chebyshev quadrature on function over [0,1]"""
        data = Chebyshev_quadrature_points_01

        assert np.all(data[:,1]<1.0)
        assert m.isclose(np.sum(data[:,0]),1.0)

        for i in range(10):
            I_true = 1.0/(i+1.0)

            I_quad = np.dot(data[:,0], data[:,1]**i)
            assert m.isclose(I_true, I_quad)

    def test_integral(self):
        """testing PrimitiveGaussian.integral"""

        # testing odd exponent
        assert m.isclose(integral(alpha, 1), 0.0)

        # testing even exponents 0, 2, 4
        assert m.isclose(integral(alpha, 0), G_alpha)
        assert m.isclose(integral(alpha, 2), G_alpha / (2.0 * alpha))
        assert m.isclose(
            integral(alpha, 4), G_alpha * 3.0 / (2.0 * alpha) ** 2.0
        )

    def test_normalization_constant(self):
        """test normalization constant"""

        # for 1s orbitals
        assert m.isclose(
            normalization_constant(alpha, 0, 0, 0),
            (2.0 * alpha / pi) ** 0.75,
        )

        # testing if it normalizes the integrals for {ex, ey, ez} in {0, 1}
        for ex in [0, 1]:
            for ey in [0, 1]:
                for ez in [0, 1]:
                    # squared norm
                    norm2 = (
                        normalization_constant(alpha, ex, ey, ez)
                        ** 2.0
                    )

                    # squared integral
                    int_x = integral(2.0 * alpha, 2 * ex)
                    int_y = integral(2.0 * alpha, 2 * ey)
                    int_z = integral(2.0 * alpha, 2 * ez)
                    int2 = int_x * int_y * int_z
                    assert m.isclose(1.0, norm2 * int2)

    def test_overlap_int_1s(self):
        """test overlap integral calculation for 1s orbitals"""
        assert m.isclose(PG1_1s.overlap_int(PG2_1s), A1 * A2 * e12 * G12**3.0)

    def test_overlap_int_1s_2p(self):
        """testing overlap integral of 1s and 2p orbitals"""

        ey = ez = 0

        for ex in [0, 1]:
            PG1 = PrimitiveGaussian(a1, X1, ex, ey, ez)
            A1 = PG1.A
            exponents1 = PG1.angular_exponents
        
            exponents12 = exponents1 + exponents2
        
            I = G12**3.0 * np.prod((X_bar-X1)**exponents1 * (X_bar-X2)**exponents2 + (exponents12==2) * 3.0/2.0/a1p2)

            assert m.isclose(PG1.overlap_int(PG2_1s), A1 * A2 * e12 * I)

    def test_kinetic_int_1s(self):
        overlap = PG1_1s.overlap_int(PG2_1s)
        I_int = overlap * a2 * (-3.0 * a2 / a1p2 - 2.0 * a2 * np.linalg.norm(X_bar-X2)**2.0 + 3.0)
        assert m.isclose(PG1_1s.kinetic_int(PG2_1s), I_int)

    def test_electron_proton_int_1s(self):
        """electron attraction integrals on hydrogen type orbitals"""
        I_int = - 2.0 * A1 * A2 * (pi / a1p2) * e12 * boys(a1p2 * np.linalg.norm(R-X_bar)**2.0)
        assert m.isclose(PG1_1s.electron_proton_int(PG2_1s, R, 1), I_int)

    def test_electron_electron_int_1s(self):
        """electron attraction integrals on hydrogen type orbitals"""
        I_int = (
            A1*A2*A3*A4
            * 2.0
            * pi**2
            / (a1p2 * a3p4)
            * e34
            * e12
            * m.sqrt(pi / (a1p2 + a3p4))
            * boys(p * Q2)
        )

        t1 = time.time()
        assert np.isclose(PG1_1s.electron_electron_int(PG2_1s, PG3_1s, PG4_1s), I_int)
        t2 = time.time()
        dt1 = t2 - t1

        t1 = time.time()
        assert np.isclose(PG1_1s.electron_electron_int(PG2_1s, PG3_1s, PG4_1s), I_int)
        t2 = time.time()
        dt2 = t2 - t1

        assert dt2>dt1

