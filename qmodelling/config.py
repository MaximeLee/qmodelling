import numpy as np
from numba import uint16, float64, float32

# numpy types
dtype_real = np.float64
dtype_int = np.uint16

# numba types
dtype_real_numba = float64 if dtype_real == np.float64 else float32
dtype_int_numba = uint16
