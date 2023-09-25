"""compute coulomb matrix from list of Contracted Gaussians"""
from typing import List
import numpy as np
from numba import jit
import multiprocessing as mp
from qmodelling.basis.primitive_gaussian import PrimitiveGaussian
from qmodelling.basis.contracted_gaussian import ContractedGaussian

def electron_electron_matrix(CPG_list: List[ContractedGaussian]):
    """coulomb matrix"""

    n = len(CPG_list)

    coulomb = np.zeros([n, n, n, n])
    n_i = n_j = n_k = n_l = 3

    for i in range(n):
        CPG_i = CPG_list[i]

        for j in range(n):
            CPG_j = CPG_list[j]

            for k in range(n):
                CPG_k = CPG_list[k]

                for l in range(n):
                    CPG_l = CPG_list[l]

                    for ii in range(n_i):
                        PG_i = CPG_i.PG_list[ii]
                        c_i = CPG_i.coeff[ii]

                        for jj in range(n_j):
                            PG_j = CPG_j.PG_list[jj]
                            c_j = CPG_j.coeff[jj]

                            for kk in range(n_k):
                                PG_k = CPG_k.PG_list[kk]
                                c_k = CPG_k.coeff[kk]

                                for ll in range(n_l):
                                    PG_l = CPG_l.PG_list[ll]
                                    c_l = CPG_l.coeff[ll]

                                    coulomb[i, j, k, l] += c_i * c_j * c_k * c_l * PrimitiveGaussian.electron_electron_int(
                                            PG_i, PG_j, PG_k, PG_l
                                    )

    return coulomb

@jit
def exchange(Vee, P):
    Vex = np.einsum("kl,iklj->ij", P, Vee)
    return -0.5*Vex

@jit
def Coulomb(Vee, P):
    """Coulomb interaction

    Vee : for electron integrals in AO basis
    P   : density matrix
    """
    #Vcoulomb = np.einsum("ijkl,kl->ij", Vee, P)
    Vcoulomb = np.einsum("kl,ijlk->ij", P, Vee)
    return Vcoulomb

