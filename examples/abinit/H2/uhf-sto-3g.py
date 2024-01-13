from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
import matplotlib.pyplot as plt
import scipy as scp

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import numpy as np
from qmodelling.molecule import Molecule
from qmodelling.utils import get_atomic_orbitals
from qmodelling.tensors.overlap import overlap_matrix
#from qmodelling.tensors.electron_electron import electron_electron_matrix, exchange_matrix, Coulomb_matrix
from qmodelling.tensors.electron_electron import *
from qmodelling.tensors.electron_proton import electron_proton_matrix
from qmodelling.tensors.kinetic import kinetic_matrix
from qmodelling.tensors.proton_proton import proton_proton_potential
from qmodelling.tensors.density import density_matrix
#from qmodelling.energy import total_energy, energies
from qmodelling.energy import energies
from qmodelling.numba_ops import *
from qmodelling.utils import timeit

seed = 10
np.random.seed(seed)

# SCF loop
@timeit
def scf_loop(molecule, eps = 1e-9, n_loop=500, theta = np.pi/2):
    # Define basis functions
    AO = get_atomic_orbitals(molecule)

    # compute operators
    S = overlap_matrix(AO)

    S_1 = np.linalg.inv(S)
    S_12  = scp.linalg.sqrtm(S_1)
    Vep = electron_proton_matrix(AO, molecule)
    Vpp = proton_proton_potential(molecule)
    T = kinetic_matrix(AO)

    Vee = electron_electron_matrix(AO)
    
    Hcore = T + Vep
    
    n = len(AO)

    # initial density/guess
    #"""
#    C_alpha = np.array([
#        [1, 1],
#        [-1, 1]
#    ])/np.sqrt(2) 
    C_alpha = np.array([
        [0.56512635, 0.58934771],
        [0.19810066, 0.43611826]
    ])
    C_beta  = np.asarray(C_alpha)
    #"""

    # mixing to break symetry
    """
    C1 = np.asarray(C_beta[:, 0:1])
    C2 = np.asarray(C_beta[:, 1:2])

    C_beta[:, 0:1] =     C1 + k*C2
    C_beta[:, 1:2] = - k*C1 +   C2
    C_beta = C_beta/np.sqrt(1+k**2)
    #"""

    # rotation to break symetry
    R = np.array([
        [np.cos(theta), - np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    C_beta = R @ C_beta

    P_alpha = density_matrix(C_alpha, 1)/2
    P_beta  = density_matrix(C_beta , 1)/2

    P_beta = P_alpha = np.zeros([n, n])

    E_old = np.inf
    
    # SCF loop
    for i in range(n_loop):
    
        # compute electron integrals : Coulomb + Exchange
        Vexchange_alpha = exchange_matrix(Vee, P_alpha)
        Vexchange_beta  = exchange_matrix(Vee, P_beta )

        Vcoulomb  = Coulomb_matrix(Vee, P_alpha + P_beta)
        #Vcoulomb_alpha  = Coulomb_matrix(Vee, P_alpha)
        #Vcoulomb_beta   = Coulomb_matrix(Vee, P_beta )
        
        # Fock operator
        F_alpha = Hcore + Vcoulomb + Vexchange_alpha
        F_beta  = Hcore + Vcoulomb + Vexchange_beta 

        Fu_alpha = S_12 @ F_alpha @ S_12
        Fu_beta  = S_12 @ F_beta  @ S_12

        # diagonalization of the operator
        _, Cu_alpha = np.linalg.eigh(Fu_alpha)
        _, Cu_beta  = np.linalg.eigh(Fu_beta )
        
        # reverse basis change
        C_alpha = S_12 @ Cu_alpha
        C_beta  = S_12 @ Cu_beta 

        # new density matrix
        P_alpha = density_matrix(C_alpha, 1)/2
        P_beta  = density_matrix(C_beta , 1)/2
        
        # Compute energy with new density matrix
        Ecore_alpha, Ecoulomb_alpha, Eexchange_alpha = energies(Hcore, Vcoulomb, Vexchange_alpha, P_alpha)
        Ecore_beta , Ecoulomb_beta , Eexchange_beta  = energies(Hcore, Vcoulomb, Vexchange_beta , P_beta )
    
        Ecore = Ecore_alpha + Ecore_beta
        Ecoulomb = Ecoulomb_alpha + Ecoulomb_beta
        Eexchange = (Eexchange_alpha + Eexchange_beta)*2

        E = Ecore + Ecoulomb + Eexchange

        if abs(E-E_old)<eps:
            print(f'Converged in {i+1} iterations!')
            return Ecore, Ecoulomb, Eexchange, Vpp
    
        # else continue
        E_old = E
    print(f'Not Converged!')
    return Ecore, Ecoulomb, Eexchange, Vpp

n_occ = 1
d_vec = np.linspace(0.4, 10, 100)
E_list = []
E_core_list = []
E_coulomb_list = []
E_exchange_list = []

for i, d in enumerate(d_vec):
    print('-'*5)
    print(i)
    print('-'*5)
    x1 = np.array([[0.0, 0.0, 0.0]])
    x2 = np.array([[d  , 0.0, 0.0]])
    
    # Define molecule
    H2 = Molecule()
    H2.add_atom('H', x1)
    H2.add_atom('H', x2)

    #Ecore, Ecoulomb, Eexchange, Vpp = scf_loop(H2)
    Ecore, Ecoulomb, Eexchange, Vpp = scf_loop(H2)
    E = Ecore + Ecoulomb + Eexchange

    E_list.append(E + Vpp)

    E_core_list    .append(Ecore)
    E_coulomb_list .append(Ecoulomb)
    E_exchange_list.append(Eexchange)

plt.figure(1)
plt.plot(d_vec, E_list)
plt.xlabel('A')
plt.ylabel('H')
plt.savefig(f'E_uhf.png')
np.savetxt('uhf.txt', np.stack([d_vec, E_list], axis=1))

plt.figure(2)
plt.plot(d_vec, E_core_list)
plt.savefig('E_core_uhf.png')

plt.figure(3)
plt.plot(d_vec, E_coulomb_list)
plt.savefig('E_coulomb_uhf.png')

plt.figure(4)
plt.plot(d_vec, E_exchange_list)
plt.savefig('E_exchange_uhf.png')
