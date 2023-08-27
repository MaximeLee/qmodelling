from qmodelling.basis.primitive_gaussian import PrimitiveGaussian, gaussian_integral
import numpy as np
import time
from config import *

class TestPrimitiveGaussian1s:

    def test_gaussian_integral(self):
        # odd 
        assert isclose(gaussian_integral(a1, 1), 0.0)
        assert isclose(gaussian_integral(a1, 3), 0.0)

        # even
        assert isclose(gaussian_integral(a1, 0), np.sqrt(pi/a1))
        assert isclose(gaussian_integral(a1, 2), np.sqrt(pi/a1)/2/a1)

    def test_normalization_constant(self):
        Ix = gaussian_integral(2.0 * a1, 0)
        Iy = gaussian_integral(2.0 * a1, 0)
        Iz = gaussian_integral(2.0 * a1, 0)
        I = Ix * Iy * Iz
        one = PG.normalization_constant()**2 * I
        assert isclose(one, 1)
        assert isclose(Ix, np.sqrt(pi/2/a1))
        assert isclose(Iy, np.sqrt(pi/2/a1))
        assert isclose(Iz, np.sqrt(pi/2/a1))

    def test_overlap_int(self):
        II = PrimitiveGaussian.overlap_int(PG, PG2)
        assert isclose(II, A1 * A2 * Ea12 * G12**3)

    def test_kinetic_int(self):
        II = PrimitiveGaussian.kinetic_int(PG, PG2)
        II_true = PrimitiveGaussian.overlap_int(PG, PG2) * a2 * (-3.0 * a2 / a1p2 - 2.0 * a2 * np.linalg.norm(X_bar12-X2)**2.0 + 3.0)
        assert isclose(II, II_true)

    def test_electron_proton(self):
        I_int = - 2.0 * A1 * A2 * (pi / a1p2) * Ea12 * boys(a1p2 * np.linalg.norm(R-X_bar12)**2.0)
        assert isclose(PrimitiveGaussian.electron_proton_int(PG, PG2, R, 1), I_int)


    def test_electron_electron(self):
        I_int = (
            A1*A2*A3*A4
            * 2.0
            * pi**2
            / (a1p2 * a3p4)
            * Ea34
            * Ea12
            * np.sqrt(pi / (a1p2 + a3p4))
            * boys(p * Q2)
        )
        t1 = time.time()
        assert isclose(PrimitiveGaussian.electron_electron_int(PG, PG2, PG3, PG4), I_int)
        t2 = time.time()
        print(f't = {t2-t1} s')
