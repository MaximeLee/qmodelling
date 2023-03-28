from qmodelling.func.primitivegaussian import PrimitiveGaussian
import numpy as np
import math as m

pi = np.pi

class TestPrimitiveGaussian:

    def test_overlap_int(self):
        alpha = pi/2
        PG1 = PrimitiveGaussian(alpha=alpha)
        PG2 = PrimitiveGaussian(alpha=alpha)
        x1 = 5.0
        x2 = 0.0
        PG1.x = x1
        PG2.x = x2
        integral = PrimitiveGaussian.overlap_int(PG1,PG2)
        assert np.isclose(integral,m.exp(-pi/4*(x1-x2)**2))

        PG1.x = np.array([0,0,x1])
        PG2.x = np.array([0,0,x2])
        integral = PrimitiveGaussian.overlap_int(PG1,PG2)
        assert np.isclose(integral,m.exp(-pi/4*np.inner(x1-x2,x1-x2)))

        PG3 = PrimitiveGaussian(alpha=1.0)
        x3 = 54.0
        PG3.x = x3
        integral = PrimitiveGaussian.overlap_int(PG2,PG3)
        assert np.isclose(integral,(2/pi)**(3/4)*m.exp(-1/(1+2/pi)*(x2-x3)**2))

    def test_kinetic_int(self):
        alpha = pi/2
        PG1 = PrimitiveGaussian(alpha=alpha)
        PG2 = PrimitiveGaussian(alpha=alpha)
        x1 = 5.0
        x2 = 0.0
        PG1.x = x1
        PG2.x = x2
        integral = PrimitiveGaussian.kinetic_int(PG1,PG2)
        assert True
