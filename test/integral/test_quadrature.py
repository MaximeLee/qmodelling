"""testing integral quadrature points"""
import numpy as np
import math
from qmodelling.integral.quadrature import get_quadrature_points
from qmodelling.integral.spherical_quadrature import R3_points_quadrature, R3_weights_quadrature, subs
from qmodelling.basis.primitive_gaussian import PrimitiveGaussian, integral

pi = np.pi
atol = 1e-16
    
class TestQuadratureGauss:

    npts = 50
    weights_abscissa = get_quadrature_points(npts, quadrature_type='gauss_chebyshev_2')
    
    weights = weights_abscissa[:,0]
    abscissa = weights_abscissa[:,1]
    
    a = 10.0

    def test_gauss_chebyshev_2_canonic(self):
        """testing function on canonic polynomial basis eg x^n"""
        
        for n in range(20):
            I_true = 0.0 if n%2==1 else 2.0 / (n+1)

            I_quadrature = np.dot(self.weights, self.abscissa**n)

            assert math.isclose(I_true, I_quadrature, abs_tol=atol)

    def test_gauss_chebyshev_2_gaussians(self):
        """testing on gaussians function with exponent/angular momentum over R with variable substitution on [-1,1]"""

        def angular_gaussian(x, n, a):
            return x**n * np.exp(-a*x**2.0)

        for n in range(20):

            I_true = integral(self.a, n)
            integrand = angular_gaussian(np.arctanh(self.abscissa), n, self.a) / (1.0 - self.abscissa**2.0)
            I_quadrature = np.dot(self.weights, integrand)
            
            assert math.isclose(I_true, I_quadrature, abs_tol=atol)
            
class TestQuadratureLevedev:

    weights_abscissa = get_quadrature_points(None, quadrature_type='lebedev')
    weights = weights_abscissa[:,0]
    theta = weights_abscissa[:,1]
    phi = weights_abscissa[:,2]

    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    def test_shapes(self):
        assert self.weights.shape==(5810,) and self.theta.shape==(5810,) and self.phi.shape==(5810,)
        assert self.x.shape==(5810,) and self.y.shape==(5810,) and self.z.shape==(5810,)

        assert math.isclose(np.max(self.theta), pi, abs_tol=atol) 
        assert math.isclose(np.min(self.theta), -pi, abs_tol=2e-2)

        assert math.isclose(np.max(self.phi), pi, abs_tol=atol) 
        assert math.isclose(np.min(self.phi), 0, abs_tol=atol)

    def test_unit_sphere(self):
        I_quad = 4*pi*np.sum(self.weights)
        assert math.isclose(4*pi, I_quad, abs_tol=atol)


    def test_2(self):
        I_quad = 4*pi*np.dot(self.weights, self.z)
        assert math.isclose(0.0, I_quad, abs_tol=4*atol)

class TestSphericalQuadrature:

    weights = R3_weights_quadrature
    points = R3_points_quadrature
    subs = subs

    def test_shapes(self):
        assert self.weights.shape==(5810*50,1) and np.isclose(np.sum(self.weights),2.0)
        assert self.points.shape==(5810*50,3)
        assert self.subs.shape==(5810*50,1)
    
    def test_gaussian(self):
        """test on 3d gaussian integral"""
        alpha = 10.0
        R2 = np.linalg.norm(self.points, axis = 1, keepdims = True)**2
        I_quad = 4.0*pi*np.einsum('ij,ij', self.weights, self.subs * np.exp(-alpha*R2))
        I_true = (pi/alpha)**(3.0/2.0)
        assert np.isclose(I_true, I_quad)
    

