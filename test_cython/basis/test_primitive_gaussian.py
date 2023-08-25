from qmodelling.basis.primitive_gaussian import PrimitiveGaussian, gaussian_integral
import numpy as np

alpha = 0.6
atom_position = np.random.uniform(size=[1,3])
ex = ey = ez = 0
PG = PrimitiveGaussian(alpha, atom_position, ex, ey, ez)
pi = np.pi
isclose = np.isclose

class TestPrimitiveGaussian:

    def test_gaussian_integral(self):
        # odd 
        assert isclose(gaussian_integral(alpha, 1), 0.0)
        assert isclose(gaussian_integral(alpha, 3), 0.0)

        # even
        assert isclose(gaussian_integral(alpha, 0), np.sqrt(pi/alpha))
        assert isclose(gaussian_integral(alpha, 2), np.sqrt(pi/alpha)/2/alpha)

    def test_normalization_constant(self):
        Ix = gaussian_integral(2.0 * alpha, 2 * ex)
        Iy = gaussian_integral(2.0 * alpha, 2 * ey)
        Iz = gaussian_integral(2.0 * alpha, 2 * ez)
        I = Ix * Iy * Iz
        one = PG.normalization_constant()**2 * I
        assert isclose(one, 1)
        assert isclose(Ix, np.sqrt(pi/2/alpha))
        assert isclose(Iy, np.sqrt(pi/2/alpha))
        assert isclose(Iz, np.sqrt(pi/2/alpha))
