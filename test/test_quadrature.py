"""testing integral quadrature points"""
import numpy as np
import math
from qmodelling.integral.quadrature import get_quadrature_points
from qmodelling.basis.primitive_gaussian import PrimitiveGaussian, integral

npts = 50
atol = 1e-16

weights_abscissa = get_quadrature_points(npts, quadrature_type='gauss_chebyshev_2')

weights = weights_abscissa[:,0]
abscissa = weights_abscissa[:,1]

a = 10.0

class TestQuadrature:

    def test_gauss_chebyshev_2_canonic(self):
        """testing function on canonic polynomial basis eg x^n"""
        
        for n in range(20):
            I_true = 0.0 if n%2==1 else 2.0 / (n+1)

            I_quadrature = np.dot(weights, abscissa**n)

            assert math.isclose(I_true, I_quadrature, abs_tol=atol)

    def test_gauss_chebyshev_2_gaussians(self):
        """testing on gaussians function with exponent/angular momentum over R with variable substitution on [-1,1]"""

        def angular_gaussian(x, n, a):
            return x**n * np.exp(-a*x**2.0)

        for n in range(20):

            I_true = integral(a, n)
            integrand = angular_gaussian(np.arctanh(abscissa), n, a) / (1.0 - abscissa**2.0)
            I_quadrature = np.dot(weights, integrand)
            
            assert math.isclose(I_true, I_quadrature, abs_tol=atol)
            
