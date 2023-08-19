import numpy as np
from qmodelling.molecule import Molecule
from qmodelling.utils import get_atomic_orbitals
from qmodelling.dft.lda import *
from qmodelling.integral.spherical_quadrature import R3_weights_quadrature, R3_points_quadrature, subs

N = 50

class TestDFT:
    def test_vxc(self):
        vxc(np.random.uniform(size=[N,1]))
        assert True

    def test_LDA_matrix_H2(self):
        H2 = Molecule()
        H2.add_atom('H', np.random.uniform(size=[1,3]))
        H2.add_atom('H', np.random.uniform(size=[1,3]))
        orbitals = get_atomic_orbitals(H2)
        quadrature = {'W':R3_weights_quadrature, 'X':R3_points_quadrature, 'subs':subs}
        n = 2
        n_quad = len(quadrature['X'])
        rhos = [np.zeros([n,n,n_quad,1])]*2 
        Vxc = LDA_matrix(orbitals, rhos, quadrature)
        assert np.all(np.isclose(0.0,Vxc))
