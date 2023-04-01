class Atom:
    """Atom class
    Z       : atomic number
    x0      : position of the atom
    orbital : atomic orbital as a list of basis functions
    """
    def __init__(self,Z,x,orbital):
        self.Z = Z
        self.orbital = orbital
        self.orbital.x = x

class Molecule:
    def __init__(self,atoms):
        self.atoms = atoms
        self.n = len(atoms)
