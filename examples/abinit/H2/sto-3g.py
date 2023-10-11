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

# SCF loop
@timeit
def scf_loop(molecule, eps = 1e-8, n_loop=500):
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
    
    # initial density
    n = len(AO)
    P = np.zeros([n, n])
    E_old = np.inf
    
    # SCF loop
    for i in range(n_loop):
    
        # compute electron integrals : Coulomb + Exchange
        Vexchange = exchange_matrix(Vee, P)

        Vcoulomb  = Coulomb_matrix(Vee, P)
        
        # Fock operator
        F = Hcore + Vcoulomb + Vexchange
        Fu = S_12 @ F @ S_12

        # diagonalization of the operator
        #_, Cu = numba_eigh(Fu)
        _, Cu = np.linalg.eigh(Fu)
        #_, Cu = scp.linalg.eigh(Fu)
        
        # reverse basis change
        C = S_12 @ Cu

        # new density matrix
        P = density_matrix(C, n_occ)
        
        # Compute energy with new density matrix
        #E = total_energy(Hcore, Vcoulomb, Vexchange, P)
        Ecore, Ecoulomb, Eexchange = energies(Hcore, Vcoulomb, Vexchange, P)
    
        E = Ecore + Ecoulomb + Eexchange

        if abs(E-E_old)<eps:
            print(f'Converged in {i+1} iterations!')
            return Ecore, Ecoulomb, Eexchange, Vpp
    
        # else continue
        E_old = E
    print(f'Not Converged!')
    return Ecore, Ecoulomb, Eexchange, Vpp

n_occ = 1
d_vec = np.linspace(0.4, 10, 40)
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
plt.savefig('E.png')

plt.figure(2)
plt.plot(d_vec, E_core_list)
plt.savefig('E_core.png')

plt.figure(3)
plt.plot(d_vec, E_coulomb_list)
plt.savefig('E_coulomb.png')

plt.figure(4)
plt.plot(d_vec, E_exchange_list)
plt.savefig('E_exchange.png')
