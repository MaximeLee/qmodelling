"""H2 UKS"""
import numpy as np
import matplotlib.pyplot as plt
from qmodelling.molecule import Molecule
from qmodelling.scf import get_total_energy
from qmodelling.utils import get_atomic_orbitals
from qmodelling.tensors.electron_electron import electron_electron_matrix

# Define multiple interatomic distances
nd = 1
dlin = np.linspace(0.4, 20, nd)

# For Loop over interatomic distances
E_list = []
for l, d in enumerate(dlin):

    # Define molecule
    H2 = Molecule()
    H2.add_atom('H', np.array([[0.0, 0.0, 0.0]]))
    H2.add_atom('H', np.array([[0.0, 0.0, d ]]))

    # get AO
    CPG_list = get_atomic_orbitals(H2)

    # SCF training loop
    electron_electron_matrix(CPG_list)
    sys.exit()

"""
    E = get_total_energy(CPG_list, H2)
    E_list.append(E)

    print(f'{l} : done!')
    print(f'distance {d} : energy = {E}!\n')


plt.title('Total energy of H2')
plt.plot(E)
plt.xlabel('x (Angstrom)')
plt.ylabel('E (H)')
plt.savefig('E_tot_H2.png')
"""
