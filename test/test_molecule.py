from qmodelling.molecule import Molecule, Atom
from qmodelling.func.orbital import Orbital
from qmodelling.func.primitivegaussian import PrimitiveGaussian

class TestMolecule:

    def testAtom(self):
        Z = 5
        coo = [5.0 , 6.0]
        PG = PrimitiveGaussian(5.6)
        basis = [PG]
        orbital = Orbital(None,basis)
        Atom(Z,coo,orbital)

    def testMolecule(self):
        Z = 1
        coo1 = [0.0 , 0.0]
        PG = PrimitiveGaussian(5.6)
        basis = [PG]
        orbital = Orbital(None,basis)
        H1 = Atom(Z,coo1,orbital)
        coo2 = [0.0 , 1.0]
        H2 = Atom(Z,coo2,orbital)
        Molecule([H1, H2])
