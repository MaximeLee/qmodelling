import numpy as np
from qmodelling.tensors.overlap import *
from qmodelling.tensors.kinetic import *
from qmodelling.tensors.electron_electron import *
from qmodelling.tensors.electron_proton import *
from qmodelling.molecule import Molecule
from qmodelling.utils import get_atomic_orbitals

class TestOperatorsH2:
    H2 = Molecule()
    
    x1 = np.zeros([1,3]) #np.random.uniform(size=[1,3])
    H2.add_atom('H',x1)
    x2 = np.ones([1,3]) #np.random.uniform(size=[1,3])
    H2.add_atom('H',x2)
    
    CPG = get_atomic_orbitals(H2)

    def test_overlap(self):
        overlap = overlap_matrix(self.CPG)
        assert np.all(overlap>0)
        assert np.all(overlap.T == overlap)
        assert np.all(overlap[0,1]<1)
        assert np.all(np.isclose(np.diag(overlap), 1.0))

    def test_kinetic(self):
        kinetic = kinetic_matrix(self.CPG)
        assert np.all(kinetic.T == kinetic)

    def test_electron_proton(self):
        elec_prot = electron_proton_matrix(self.CPG, self.H2)
        assert np.all(elec_prot<0)
        assert np.all(np.isclose(elec_prot.T,elec_prot))

        abs_elec_prot = np.abs(elec_prot)

        assert np.all(np.diag(abs_elec_prot)>=abs_elec_prot)

#    def test_coulomb(self):
#        coulomb = electron_electron_matrix(self.CPG)
#        assert np.all(coulomb.T==coulomb)
#        assert np.all(coulomb>0.0)

