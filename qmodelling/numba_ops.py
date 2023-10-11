from numba import jit
import numpy as np

@jit(cache=True, nopython=True, nogil=True)
def numba_basis_change(F, X):
    return X.T @ F @ X

@jit(cache=True, nopython=True, nogil=True)
def numba_eigh(M):
    return np.linalg.eigh(M)

@jit(cache=True, nopython=True, nogil=True)
def numba_eig(M):
    return np.linalg.eig(M)

@jit(cache=True, nopython=True, nogil=True)
def numba_prod(A,B):
    return A@B
