"""compute different element of the total energy"""
import numpy as np
from qmodelling.integral.becke_partitionning import normalized_cell_function
from qmodelling.dft.lda import exc

pi = np.pi


def energies(orbitals, rho_atoms, Hcore, Vcoulomb, P, quadrature):
    """function"""
    Ecore = np.sum(P * Hcore)
    Ecoulomb = np.sum(P * Vcoulomb) / 2.0

    X = quadrature["X"]
    Wquad = quadrature["W"]
    subs = quadrature["subs"]

    # loop over fuzzy cells
    n = len(orbitals)
    Exc = 0.0

    for ii in range(n):
        R1 = orbitals[ii].position
        dx1 = X + R1

        for jj in range(ii + 1, n):
            R2 = orbitals[jj].position
            dx2 = X + R2

            # 1 -> 2
            Wcell12 = normalized_cell_function(dx1, R1, R2)
            Exc += (
                4.0
                * pi
                * np.eisum(
                    "ij,ij",
                    Wquad,
                    Wcell12 * rho_atoms[0][ii, jj] * exc(rho_atoms[0][ii, jj]) * subs,
                )
            )

            # 2 -> 1
            Wcell21 = normalized_cell_function(dx2, R2, R1)
            Exc += (
                4.0
                * pi
                * np.eisum(
                    "ij,ij",
                    Wquad,
                    Wcell21 * rho_atoms[1][ii, jj] * exc(rho_atoms[1][ii, jj]) * subs,
                )
            )

    return Ecore, Ecoulomb, Exc
