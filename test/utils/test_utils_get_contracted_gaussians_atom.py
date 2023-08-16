import numpy as np
from qmodelling.utils import *

atom_position = np.random.uniform(size=[1,3])

class TestUtilsGetContractedGaussiansOfAtom:
    
    def test_get_contracted_gaussians_of_hydrogen(self):

        symbol = 'H'

        orbital = get_contracted_gaussians_of_atom(symbol, atom_position)

        # testing alpha
        assert len(orbital)==1
        PG_list =  orbital[0].PG_list

        assert len(PG_list) == 6

        H_coeff = np.array([
            0.9163596281E-02,
            0.4936149294E-01,
            0.1685383049E+00,
            0.3705627997E+00,
            0.4164915298E+00,
            0.1303340841E+00
        ])

        assert np.all(orbital[0].coeff==H_coeff)

        assert PG_list[0].alpha==0.3552322122E+02
        assert PG_list[1].alpha==0.6513143725E+01
        assert PG_list[2].alpha==0.1822142904E+01
        assert PG_list[3].alpha==0.6259552659E+00
        assert PG_list[4].alpha==0.2430767471E+00
        assert PG_list[5].alpha==0.1001124280E+00

    def test_get_contracted_gaussians_of_beryllium(self):

        symbol = 'Be'

        orbital = get_contracted_gaussians_of_atom(symbol, atom_position)

        assert len(orbital)==2

        # testing beta orbitals coefficients 2s
        Be_coeff = np.array([
            -0.1325278809E-01,
            -0.4699171014E-01,
            -0.3378537151E-01,
             0.2502417861E+00,
             0.5951172526E+00,
             0.2407061763E+00
        ])

        assert np.all(orbital[1].coeff==Be_coeff)

    def test_get_contracted_gaussians_of_boron(self):

        symbol = 'B'

        orbital = get_contracted_gaussians_of_atom(symbol, atom_position)

        assert len(orbital)==3

        B_2p = orbital[-1]
        assert np.all(B_2p.PG_list[0].angular_exponents == np.array([1, 0, 0]))

        # tester 2p orbital
        B_2p_coeff = np.array([
            0.3759696623E-02,
            0.3767936984E-01,
            0.1738967435E+00,
            0.4180364347E+00,
            0.4258595477E+00,
            0.1017082955E+00
        ])
        assert np.all(B_2p.coeff == B_2p_coeff)

    def test_get_contracted_gaussians_of_oxygen(self):

        symbol = 'O'

        orbital = get_contracted_gaussians_of_atom(symbol, atom_position)

        assert len(orbital)==5

    def test_get_contracted_gaussians_of_neon(self):

        symbol = 'Ne'

        orbital = get_contracted_gaussians_of_atom(symbol, atom_position)

        assert len(orbital)==5

        for i in range(3):
            ex = (i == 0) * 1
            ey = (i == 1) * 1
            ez = (i == 2) * 1
            for j in range(6):
                alpha_ij = orbital[2+i].PG_list[j].angular_exponents
                assert np.all(
                    alpha_ij==np.array([ex, ey, ez])
                    )
