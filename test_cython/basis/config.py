import numpy as np
from scipy import special
from qmodelling.basis.primitive_gaussian import PrimitiveGaussian, gaussian_integral

def boys(x):
    """boys function used for one-two electron integrals"""
    n = 0
    if x == 0:
        return 1.0 / (2 * n + 1)
    return (
        special.gammainc(n + 0.5, x)
        * special.gamma(n + 0.5)
        * (1.0 / (2 * x ** (n + 0.5)))
    )


pi = np.pi
isclose = np.isclose

R = np.random.uniform(size=[1,3])

# PG 1
a1 = 0.6
X1 = np.random.uniform(size=[1,3])
PG = PrimitiveGaussian(a1, X1, 0, 0, 0)
A1 = PG.normalization_constant()

# PG 2
a2 = 1.0
X2 = np.random.uniform(size=[1,3])
PG2 = PrimitiveGaussian(a2, X2, 0, 0, 0)
A2 = PG2.normalization_constant()

# 1-2
a1p2 = a1 + a2
X_bar12 = X12 = (X1 * a1 + X2 * a2) / (a1 + a2)
G12 = np.sqrt(pi/a1p2)
Ea12 = np.exp(-a1*a2/a1p2 * np.linalg.norm(X1-X2)**2)

# PG 3
a3 = 1.0
X3 = np.random.uniform(size=[1,3])
PG3 = PrimitiveGaussian(a3, X3, 0, 0, 0)
A3 = PG3.normalization_constant()

# PG 4
a4 = 1.0
X4 = np.random.uniform(size=[1,3])
PG4 = PrimitiveGaussian(a4, X4, 0, 0, 0)
A4 = PG4.normalization_constant()

# 3-4
a3p4 = a3 + a4
Ea34 = np.exp(-a3*a4/a3p4 * np.linalg.norm(X3-X4)**2)
X34 = (X3*a3 + X4*a4)/(a3p4)

# 1-2-3-4
Q2 = np.linalg.norm(X34-X12)**2.0
p = a1p2 * a3p4 / (a1p2 + a3p4)
