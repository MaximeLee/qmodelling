import numpy as np
from qmodelling.molecule import Molecule
from qmodelling.utils import *

class TestUtilGetAtomicOrbitals:

    def test_get_atomic_orbitals_of_h2(self):
        molec = Molecule()

        atoms = ['H', 'H']

        for atom in atoms:
            molec.add_atom(atom, np.random.uniform(size=[1,3]))

        assert len(get_atomic_orbitals(molec)) == 2

    def test_get_atomic_orbitals_of_water(self):
        molec = Molecule()

        atoms = ['H', 'H', 'O']

        for atom in atoms:
            molec.add_atom(atom, np.random.uniform(size=[1,3]))

        assert len(get_atomic_orbitals(molec)) == 7
