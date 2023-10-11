"""get atomic orbitals from basis and molecule"""
import os
import time
from functools import wraps

# from itertools import chain
import numpy as np
from mendeleev import element

from qmodelling.basis.primitive_gaussian import PrimitiveGaussian
from qmodelling.basis.contracted_gaussian import ContractedGaussian


def get_atomic_orbitals(molecule, basis_fn="sto-3g"):
    """loop over symbols to get basis coefficients"""

    if basis_fn != "sto-3g":
        raise ValueError("Only sto-3g supported atm")

    # list of atomic orbitals of each atom
    # whose elements are : functions [PG], coefficients [float]
    atomic_orbitals = []

    # loop over atoms
    for i in range(molecule.natoms):
        symbol = molecule.atomic_symbols[i]
        atom_position = molecule.atomic_positions[i]
        atom_charge = molecule.atomic_charges[i]

        CG_atom = get_contracted_gaussians_of_atom(symbol, atom_position, atom_charge, basis_fn)

        atomic_orbitals.extend(CG_atom)
    return atomic_orbitals


def get_contracted_gaussians_of_atom(symbol, position, atom_charge, basis_fn):
    """read file containing parameters of primitive gaussian"""

    contracted_gaussian_list = []

    S_number = 1
    SP_number = 2

    eof = False

    # electroninc configuration of the atom
    elec_config = element(symbol).ec.conf
    elec_config[next(reversed(elec_config))] -= atom_charge

    if elec_config[next(reversed(elec_config))]<1:
        elec_config.popitem(last=True)

    # text file containing coefficients
    fn = f"{os.path.dirname(__file__)}/basis/coefficients/{basis_fn}/{symbol}.txt"

    # read the orbital type (only atm): S, SP
    with open(fn, "r", encoding="ascii") as data:
        orbital_type = data.readline().strip()

        while orbital_type.strip():
            # storing parameters
            gaussian_exponents = []
            coeffs = []

            # read 3 next rows
            for _ in range(3):
                row = data.readline().strip().split()
                gaussian_exponents.append(float(row[0]))

                # coefficients
                coeffs.append([float(c) for c in row[1:]])

            coeffs = np.array(coeffs).reshape(3, -1)

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

                    eof = n_elec < 3

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

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

