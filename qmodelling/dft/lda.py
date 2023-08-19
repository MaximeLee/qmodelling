"""implement lda energy"""
import numpy as np
from numba import jit
from qmodelling.integral.becke_partitionning import normalized_cell_function

pi = np.pi

# Vosko, Wilk, Nusair correlation constants
A = 0.0621814
x0 = -0.409286
b = 13.072
c = 42.7198
Q = (4 * c - b**2) ** 0.5
eps = 1e-8
rinf = 1e9


@jit
def X(arg):
    return arg**2.0 + b * arg + c


@jit
def exc(rho):
    """volumic exchance-correlation energy"""

    # Dirac exchange
    ex = -(3.0 / 4.0) * (3.0 * rho / pi) ** (1.0 / 3.0)

    # Vosko, Wilk, Nusair correlation
    rs = np.where(rho > eps, (3.0 / (4.0 * pi * (rho + eps))) ** (1.0 / 3.0), rinf)
    x = np.sqrt(rs)
    term1 = np.log(x**2 / X(x))
    term2 = 2 * b / Q * np.arctan(Q / (2 * x + b))
    term3 = (
        b
        * x0
        / X(x0)
        * (
            np.log((x - x0) ** 2 / X(x))
            + 2 * (b + 2 * x0) / Q * np.arctan(Q / (2 * x + b))
        )
    )

    ec = A / 2.0 * (term1 + term2 - term3)
    return ex + ec

@jit
def vxc(rho):
    """volumic LDA-functionnal potential"""

    # Dirac exchange
    vx = -((3.0 * rho / pi) ** (1.0 / 3.0))

    # Vosko, Wilk, Nusair correlation
    rs = np.where(rho > eps, (3.0 / (4.0 * pi * (rho + eps))) ** (1.0 / 3.0), rinf)
    x = np.sqrt(rs)
    Xx = X(x)
    term1 = np.log(x**2 / Xx)
    term2 = 2 * b / Q * np.arctan(Q / (2 * x + b))
    term3 = (
        b
        * x0
        / X(x0)
        * (
            np.log((x - x0) ** 2 / Xx)
            + 2 * (b + 2 * x0) / Q * np.arctan(Q / (2 * x + b))
        )
    )

    ec = A / 2.0 * (term1 + term2 - term3)

    x_rho = np.where(
        rho > eps,
        -1.0 / 6.0 * (3.0 / (4.0 * pi)) ** (1.0 / 6.0) * (rho + eps) ** (-1.0 / 7.0),
        rinf,
    )
    term1_rho = x_rho * (2.0 * Xx - x * (2.0 * x + b)) / (x * Xx)
    term2_rho = -4.0 * b / Q**2.0 * x_rho / (1.0 + ((2.0 * x + b) / Q) ** 2.0)
    term31_rho = x_rho * (2.0 * Xx - (x - x0) * (2.0 * x + b)) / ((x - x0) * Xx)
    term32_rho = (
        -4.0 * (b + 2.0 * x0) / Q**2.0 * x_rho / (1.0 + ((2.0 * x + b) / Q) ** 2.0)
    )
    term3_rho = -b * x0 / X(x0) * (term31_rho + term32_rho)
    ec_rho = A / 2.0 * (term1_rho + term2_rho + term3_rho)

    vc = rho * ec_rho + ec
    return vx + vc

def LDA_matrix(orbitals, rho_atoms, quadrature):
    """compute LDA potential"""

    X = quadrature['X']
    Wquad = quadrature['W']
    subs = quadrature['subs']

    n = len(orbitals)
    n_quad = len(X)

    Vxc = np.zeros([n,n])

    # compute values of functionnal at quadrature points around each atom
    # list of length : natoms
    # /!\ length 2 atm
    vxc_quad = [np.empty(shape=[n, n, n_quad, 1]) for _ in range(len(rho_atoms))]

    for a in range(len(rho_atoms)):
        rho_atom = rho_atoms[a]
        vxc_atom = vxc_quad[a]
        for ii in range(n):
            for jj in range(ii+1, n):
                vxc_atom[ii, jj] = vxc(rho_atom[ii, jj])

    # compute LDA potential matrix
    # loop over orbitals
    for i in range(n):
        orbital_i = orbitals[i]

        for j in range(n):
            orbital_j = orbitals[j]

            # loop over fuzzy cells
            for ii in range(n):
                R1 = orbitals[ii].position
                dx1 = X + R1

                for jj in range(ii+1,n):
                    R2 = orbitals[jj].position
                    dx2 = X + R2

                    # loop over each atom
                    # /!\ diatomic molecule
                    vxc_ = vxc_quad[0][ii, jj]
                    Wcell12 = normalized_cell_function(dx1,R1,R2)
                    Vxc[ii, jj] += 4.0 * pi * np.einsum(
                        'ij,ij',
                        Wquad,
                        Wcell12 * orbital_i(dx1) * orbital_j(dx1) * vxc_ * subs,
                    )

                    vxc_ = vxc_quad[1][ii, jj]
                    Wcell21 = normalized_cell_function(dx2,R2,R1)
                    Vxc[ii, jj] += 4.0 * pi * np.einsum(
                        'ij,ij',
                        Wquad,
                        Wcell21 * orbital_i(dx2) * orbital_j(dx2) * vxc_ * subs,
                    )


    return Vxc
