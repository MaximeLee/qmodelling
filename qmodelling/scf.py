"""SCF convergence loop"""
import numpy as np
from scipy import linalg
from qmodelling.tensors.electron_electron import electron_electron_matrix, Coulomb
from qmodelling.tensors.electron_proton import electron_proton_matrix
from qmodelling.tensors.kinetic import kinetic_matrix
from qmodelling.tensors.overlap import overlap_matrix
from qmodelling.tensors.proton_proton import proton_proton_potential
from qmodelling.tensors.density import density_matrix
from qmodelling.dft.lda import LDA_matrix
from qmodelling.integral.spherical_quadrature import (
    subs,
    R3_weights_quadrature,
    R3_points_quadrature,
)
from qmodelling.energy import energies


def get_total_energy(orbitals, molecule, niter=200, tol=1e-8):
    """optimize MO with Lagragian optimization of the Roothan-Hall system"""

    # compute matrices
    T = kinetic_matrix(orbitals)
    Vee = electron_electron_matrix(orbitals)
    Vep = electron_proton_matrix(orbitals, molecule)
    Vpp = proton_proton_potential(molecule)

    # core Hamiltonian
    Hcore = T + Vep

    S = overlap_matrix(orbitals)

    # S**(-1/2)
    S_12 = np.linalg.inv(linalg.sqrtm(S))

    # get quadrature points
    X = R3_points_quadrature
    W = R3_weights_quadrature
    quadrature = {"X": X, "W": W, "subs": subs}

    ##################
    # SCF loop
    ##################

    n = len(orbitals)
    n_quad = len(X)
    n_atoms = molecule.natoms

    # occupation number
    N_alpha = n // 2 + n % 2
    N_beta = n // 2

    # initial density matrix
    P = np.zeros([n, n])

    # initial energy
    E_elec = np.inf

    # Initial density
    # values of density around every atom in orbital basis at quadrature points (Lebedev)
    # length : natoms
    # shape : (n_orbital, n_orbitals, n_quadrature)
    rho_atoms = [np.zeros(shape=[n, n, n_quad, 1]) for _ in range(n_atoms)]

    # Product of atomic orbitals of each quadrature points centered around each atom
    # length : natoms
    # shape : (n_orbital, n_orbitals, n_quadrature, n_orbitals, n_orbitals)
    BB = [np.zeros(shape=[n, n, n_quad, n, n]) for _ in range(n_atoms)]

    for ii in range(n):
        R1 = orbitals[ii].position
        dx1 = X + R1

        for jj in range(ii+1,n):
            R2 = orbitals[jj].position
            dx2 = X + R2

            B = np.hstack([orbital(dx1) for orbital in orbitals])
            BB[0][ii, jj] = np.einsum("ni,nj->nij", B, B)

            B = np.hstack([orbital(dx2) for orbital in orbitals])
            BB[1][ii, jj] = np.einsum("ni,nj->nij", B, B)

    del(B, dx1, dx2)

    for it in range(niter):
        # energy update
        E_elec_old = E_elec

        # e-e Coulomb interaction
        Vcoulomb = Coulomb(Vee, P)

        # density functionnal potential
        Vxc = LDA_matrix(orbitals, rho_atoms, quadrature)

        # Fock operator
        F = Hcore + Vcoulomb + Vxc

        # MO coefficients of LCAO approximation
        Fu = np.dot(S_12, np.dot(F, S_12))
        _, eigvec = linalg.eigh(Fu)
        C = np.dot(S_12, eigvec)

        # density matrix
        P = density_matrix(C, N_alpha, N_beta)

        # new electron density
        rho_atoms = density(rho_atoms, orbitals, P, BB)

        # new electronic energy
        Ecore, Ecoulomb, Exc = energies(
            orbitals, rho_atoms, Hcore, Vcoulomb, P, quadrature
        )
        E_elec = Ecore + Ecoulomb + Exc


        if abs(E_elec_old - E_elec) < tol:
            return E_elec + Vpp, True

    return E_elec + Vpp, False
