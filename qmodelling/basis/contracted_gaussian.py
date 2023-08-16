"""Contracted Gaussian module"""
import numpy as np
from qmodelling.basis.primitive_gaussian import PrimitiveGaussian

class ContractedGaussian:

    def __init__(self, coeff, PG_list):
        self.coeff = coeff
        self.PG_list = PG_list

    def __eq__(self, other):
        return np.all(self.coeff==other.coeff) and self.PG_list==other.PG_list

    def __len__(self):
        return len(self.PG_list)
