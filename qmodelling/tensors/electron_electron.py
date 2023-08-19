"""compute coulomb matrix from list of Contracted Gaussians"""
import numpy as np
from numba import jit
from qmodelling.basis.primitive_gaussian import PrimitiveGaussian


def electron_electron_matrix(CPG_list):
    """coulomb matrix"""

    n = len(CPG_list)

    coulomb = np.zeros([n, n, n, n])
    n_i = n_j = n_k = n_l = 6

    for i in range(n):
        CPG_i = CPG_list[i].PG_list
        CPG_i_coeff = CPG_list[i].coeff

        for j in range(n):
            CPG_j = CPG_list[j].PG_list
            CPG_j_coeff = CPG_list[j].coeff

            for k in range(n):
                CPG_k = CPG_list[k].PG_list
                CPG_k_coeff = CPG_list[k].coeff

                for l in range(n):
                    CPG_l = CPG_list[l].PG_list
                    CPG_l_coeff = CPG_list[l].coeff

                    for ii in range(n_i):
                        PG_i = CPG_i[ii]
                        c_i = CPG_i_coeff[ii]

                        for jj in range(n_j):
                            PG_j = CPG_j[jj]
                            c_j = CPG_j_coeff[jj]

                            for kk in range(n_k):
                                PG_k = CPG_k[kk]
                                c_k = CPG_k_coeff[kk]

                                for ll in range(n_l):
                                    PG_l = CPG_l[ll]
                                    c_l = CPG_l_coeff[ll]

                                    coulomb[i, j, k, l] += c_i * c_j * c_k * c_l * PrimitiveGaussian.electron_electron_int(
                                            PG_i, PG_j, PG_k, PG_l
                                    )
    return coulomb

@jit
def Coulomb(Vee, P):
    """Coulomb interaction

    Vee : for electron integrals in AO basis
    P   : density matrix
    """
    Vcoulomb = np.einsum("ijkl,kl->ij", Vee, P)
    return Vcoulomb

