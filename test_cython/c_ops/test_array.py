import numpy as np
from qmodelling.c_ops.array import array_atanh as atanh
from qmodelling.c_ops.array import linear_combination as lc

N = 50
X = np.random.uniform(size=N)

X1 = np.random.uniform(size=3)
a1 = 5.0
X2 = np.random.uniform(size=3)
a2 = 7.0

X12_lc = a1 * X1 + a2 * X2

class TestArray:

    def test_atanh(self):
        A1 = np.arctanh(X)
        A2 = atanh(X)
        
        assert np.all(np.isclose(A1, A2))

    def test_linear_combination(self):
        X12_lc_c = lc(X1, a1, X2, a2)
        assert np.all(np.isclose(X12_lc, X12_lc_c))

