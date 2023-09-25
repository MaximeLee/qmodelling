"""Molecule class"""
from mendeleev import element
import sqlalchemy


class Molecule:
    """
    atomic_numbers (list) :
    atomic_positions (list) : stores atomic positions as numpy arrays ?
    """

    def __init__(self):
        self.natoms = 0
        self.atomic_symbols = []
        self.atomic_numbers = []
        self.atomic_charges = []
        self.atomic_positions = []

    def add_atom(self, symbol, atom_position, atomic_charge=0):
        """add an Atom to the molecule
        Symbol (str)
        atom_position ([], tuple, np.array())
        """
        try:
            Z = element(symbol).atomic_number
            self.atomic_numbers.append(Z)
        except sqlalchemy.exc.NoResultFound as AtomNotFound:
            raise ValueError("Unknown atom") from AtomNotFound
        self.atomic_positions.append(atom_position)
        self.atomic_symbols.append(symbol)
        self.atomic_charges.append(atomic_charge)
        self.natoms += 1
