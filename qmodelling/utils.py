"""get atomic orbitals from basis and molecule"""
import os

# from itertools import chain
import numpy as np
from mendeleev import element

from qmodelling.basis.primitive_gaussian import PrimitiveGaussian
from qmodelling.basis.contracted_gaussian import ContractedGaussian


def get_atomic_orbitals(molecule, basis="sto-6g"):
    """loop over symbols to get basis coefficients"""

    if basis != "sto-6g":
        raise ValueError("Only sto-6g supported atm")

    # list of atomic orbitals of each atom
    # whose elements are : functions [PG], coefficients [float]
    atomic_orbitals = []

    # loop over atoms
    for i in range(molecule.natoms):
        symbol = molecule.atomic_symbols[i]
        atom_position = molecule.atomic_positions[i]

        CG_atom = get_contracted_gaussians_of_atom(symbol, atom_position)

        atomic_orbitals.extend(CG_atom)
    return atomic_orbitals


def get_contracted_gaussians_of_atom(symbol, position):
    """read file containing parameters of primitive gaussian"""

    #    contracted_gaussian_list_alpha = []
    #    contracted_gaussian_list_beta = []
    contracted_gaussian_list = []

    S_number = 1
    SP_number = 2

    eof = False

    # electroninc configuration of the atom
    elec_config = element(symbol).ec.conf

    # text file containing coefficients
    fn = f"{os.path.dirname(__file__)}/basis/sto-6g/{symbol}.txt"

    # read the orbital type (only atm): S, SP
    with open(fn, "r", encoding="ascii") as data:
        orbital_type = data.readline().strip()

        while orbital_type.strip():
            # storing parameters
            gaussian_exponents = []
            coeffs = []

            # read 6 next rows
            for _ in range(6):
                row = data.readline().strip().split()
                gaussian_exponents.append(float(row[0]))

                # coefficients
                coeffs.append([float(c) for c in row[1:]])

            coeffs = np.array(coeffs).reshape(6, -1)

            # define the Contracted Gaussians
            if orbital_type == "S":
                n_elec = elec_config[(S_number, "s")]

                coeffs_ = coeffs.flatten()
                PG_list = [
                    PrimitiveGaussian(alpha=alpha, atom_position=position)
                    for alpha in gaussian_exponents
                ]
                contracted_gaussians = [ContractedGaussian(coeffs_, PG_list)]
                if n_elec == 1:
                    eof = True

                S_number += 1

            elif orbital_type == "SP":
                # adding s orbital
                PG_list = [
                    PrimitiveGaussian(alpha=alpha, atom_position=position)
                    for alpha in gaussian_exponents
                ]
                contracted_gaussians = [ContractedGaussian(coeffs[:, 0], PG_list)]

                # adding p orbital/s
                try:
                    coeffs_ = coeffs[:, 1]

                    n_elec = elec_config[(S_number, "p")]
                    
                    all_p_orbitals = (n_elec // 3) > 0

                    n_p_orbitals = 3 if all_p_orbitals else n_elec%3

                    for i in range(n_p_orbitals):
                        ex = (i == 0) * 1
                        ey = (i == 1) * 1
                        ez = (i == 2) * 1
    
                        PG_list = [
                            PrimitiveGaussian(
                                alpha=alpha, atom_position=position, ex=ex, ey=ey, ez=ez
                            )
                            for alpha in gaussian_exponents
                        ]
    
                        contracted_gaussians.append(
                            ContractedGaussian(coeffs_, PG_list)
                        )

                    eof = n_elec < 6

                except KeyError:
                    eof = True

                SP_number += 1
            else:
                raise NotImplementedError("d and above orbitals not supported yet")

            contracted_gaussian_list.extend(contracted_gaussians)

            if eof:
                break

            orbital_type = data.readline().strip()

    return contracted_gaussian_list
