"""density related computation"""
import numpy as np
from numba import jit

@jit
def density_matrix(c, N_alpha, N_beta):
    """computes the density matrix"""

    cca = c[:, 0:N_alpha]
    ccb = c[:, 0:N_beta]

    # zeros padding in case there are more alpha e-
    if N_alpha > N_beta:
        ccb = np.hstack([ccb, np.zeros(shape=[len(ccb), 1])])

    P_alpha = cca @ cca.T
    P_beta = ccb @ ccb.T

    return P_alpha + P_beta


def density(rho_atoms, orbitals, P, BB):
    """computing density at each points of discretized grid"""
    n = len(orbitals)
    n_atoms = len(rho_atoms)

    # computing volumic functionnal potential beforehand
    for a in range(n_atoms):
        for ii in range(n):
            for jj in range(ii + 1, n):
                rho_atoms[a][ii, jj] = np.einsum("ij,nij->n", P, BB[a][ii, jj]).reshape(
                    -1, 1
                )

    return rho_atoms
