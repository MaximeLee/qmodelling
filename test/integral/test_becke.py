import numpy as np
from qmodelling.integral.becke_partitionning import *


r = np.zeros(shape=[1,3])
r[0,0] = 0.0

R1 = np.zeros(shape=[1,3])
R1[0,0] = -1.0

R2 = np.zeros(shape=[1,3])
R2[0,0] =  1.0

class TestBecke:

    def test_confocal_elliptic_coordinates(self):
        assert confocal_elliptic_coordinates(r,R1,R2)

    def test_normalized_cell_function(self):
        w12 = normalized_cell_function(r, R1, R2)
        w21 = normalized_cell_function(r, R2, R1)
        assert np.all(np.isclose(w12+w21,1.0))

