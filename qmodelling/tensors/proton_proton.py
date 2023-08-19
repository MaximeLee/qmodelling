"""compute proton repulsion interaction"""
import numpy as np


def proton_proton_potential(molecule):

    Vpp = 0.0
    for i in range(molecule.natoms):
        Z_i = molecule.atomic_numbers[i]
        Rpi = molecule.atomic_positions[i]

        for j in range(i+1, molecule.natoms):
            Z_j = molecule.atomic_numbers[j]
            Rpj = molecule.atomic_positions[j]
            
            Vpp += Z_i * Z_j /np.linalg.norm(Rpi-Rpj)

    return Vpp
