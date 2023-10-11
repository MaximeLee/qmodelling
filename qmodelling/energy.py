import numpy as np
from numba import jit

@jit
def energies(Hcore, Vcoulomb, Vexchange, P):
    Ecore = np.sum(Hcore*P)

    Ecoulomb = np.sum(Vcoulomb*P) * 0.5

    Eexchange = np.sum(Vexchange*P) * 0.5

    return Ecore, Ecoulomb, Eexchange
