"""compute coulomb matrix from list of Contracted Gaussians"""
from typing import List
import numpy as np
from numba import jit
import multiprocessing as mp
from qmodelling.basis.primitive_gaussian import PrimitiveGaussian
from qmodelling.basis.contracted_gaussian import ContractedGaussian

electron_electron_int = PrimitiveGaussian.electron_electron_int

def task_function(i, j, k, l, CPG_list):

    val = 0.0

    CPG_i = CPG_list[i].PG_list
    CPG_i_coeff = CPG_list[i].coeff

    CPG_j = CPG_list[j].PG_list
    CPG_j_coeff = CPG_list[j].coeff

    CPG_k = CPG_list[k].PG_list
    CPG_k_coeff = CPG_list[k].coeff

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

                    val += c_i * c_j * c_k * c_l * electron_electron_int(
                            PG_i, PG_j, PG_k, PG_l
                    )

    return val

def electron_electron_matrix(CPG_list: List[ContractedGaussian], nb: int=6):
    """coulomb matrix"""

    n = len(CPG_list)

    n_i = n_j = n_k = n_l = nb
    """
    coulomb = np.zeros([n, n, n, n])
    for i in prange(n):
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
    """

    num_cores = mp.cpu_count()
    pool = mp.Pool(processes=num_cores)
    
    task_args = [(i,j,k,l, CPG_list) for i in range(n) for j in range(n) for k in range(n) for l in range(n)] 
    
    results = pool.starmap(task_function, task_args)
    pool.close()
    pool.join()

    coulomb = np.array(results).reshape(n,n,n,n)

    return coulomb

@jit
def Coulomb(Vee, P):
    """Coulomb interaction

    Vee : for electron integrals in AO basis
    P   : density matrix
    """
    Vcoulomb = np.einsum("ijkl,kl->ij", Vee, P)
    return Vcoulomb

