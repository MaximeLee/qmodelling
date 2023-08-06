from qmodelling.molecule import Molecule
import numpy as np
import pytest

dim = 3
molec = Molecule()

class TestMolecule:

    def test_add_atom(self):
        
        atoms = ['H', 'He', 'Li']

        for atom in atoms:
            molec.add_atom(atom, np.random.uniform(size=[1,dim]))

    def test_add_atom_exception(self):
        with pytest.raises(ValueError):
            molec.add_atom('HHH', [0, 0, 0])
