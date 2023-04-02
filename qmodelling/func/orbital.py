class Orbital:
    """
    coeff : list of the coefficients of the linear combination of basis functions
    basis : list of basis functions
    x     : position of the atom
    """
    def __init__(self,coeff,basis):
        self.coeff = coeff
        self.basis = basis
         
