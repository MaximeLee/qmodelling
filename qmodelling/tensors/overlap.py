"""compute overlap matrix from list of Contracted Gaussians"""
import numpy as np
from qmodelling.basis.primitive_gaussian import PrimitiveGaussian

def overlap_matrix(CPG_list):
    """overlap matrix"""

    n = len(CPG_list)

    overlap = np.zeros([n, n])

    for i in range(n):  
        CPG_i = CPG_list[i]
        n_i = len(CPG_i)

        for j in range(n):
            CPG_j = CPG_list[j]
            n_j = len(CPG_j)

            for k in range(n_i):
                PG_i = CPG_i.PG_list[k]
                c_i = CPG_i.coeff[k]

                for l in range(n_j):
                    PG_j = CPG_j.PG_list[l]
                    c_j = CPG_j.coeff[l]

                    overlap[i,j] += c_i * c_j * PrimitiveGaussian.overlap_int(PG_i, PG_j)

    return overlap
