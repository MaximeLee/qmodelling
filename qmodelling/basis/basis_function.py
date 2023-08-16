"""abstract class for every basis functions"""
from abc import ABC, abstractmethod


class BasisFunction(ABC):
    """parent class for every basis functions"""

    def __init__(self, atom_position=None):
        self.atom_position = atom_position

    @abstractmethod
    def overlap_int(self, basis2):
        """overlap integral"""

    @abstractmethod
    def kinetic_int(self, basis2):
        """kinetic integral"""

    @abstractmethod
    def electron_proton_int(self, basis2, R, Z):
        """electron-proton integral"""

    @abstractmethod
    def electron_electron_int(self, basis2, basis3, basis4):
        """electron-electron integral"""
