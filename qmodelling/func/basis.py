from abc import ABC, abstractmethod

class Basis(ABC):

    def __init__(self,**kwargs):
        """constructor"""
        self.x = None

    @classmethod
    @abstractmethod
    def kinetic_int(basis1,basis2):
        """kinetic integral"""

    @classmethod
    @abstractmethod
    def electron_electron_int(cls,basis1,basis2):
        """electron-electron integral"""

    @classmethod
    @abstractmethod
    def electron_proton_int(cls,basis1,basis2):
        """electron-proton integral"""

    @classmethod
    @abstractmethod
    def overlap_int(cls,basis1,basis2):
        """overlap integral"""

