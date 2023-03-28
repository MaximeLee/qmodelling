from qmodelling.func.basis import Basis
import numpy as np
import math as m

pi = np.pi

class PrimitiveGaussian(Basis):
    """Primitive Gaussian class
    
    only considering 1s orbital atm

    alpha : parameter of the gaussian function
    """
    def __init__(self,alpha):
        super().__init__()
        self.alpha = alpha
        # normalizing constant
        self.A     = ( 2.0 * alpha / pi ) ** 0.75 

    @classmethod
    def kinetic_int(cls,basis1,basis2):
        """kinetic integral"""
        overlap = cls.overlap_int(basis1,basis2)
        a1 , a2 = basis1.alpha , basis2.alpha

        x1 , x2 = basis1.x , basis2.x
        a1p2 = a1 + a2
        xmean = (a1*x1+a2*x2)/a1p2
        dr = x2 - xmean
        dr2 = np.inner(dr,dr)

        k_int  = -2 * a2**2 * dr2  * overlap  
        k_int += -3 * a2**2 / a1p2 * overlap 
        k_int +=  3 * a2           * overlap
        return k_int

    @classmethod
    def electron_electron_int(cls,basis1,basis2):
        """electron-electron integral"""
        return

    @classmethod
    def electron_proton_int(cls,basis1,basis2):
        """electron-proton integral"""
        return

    @classmethod
    def overlap_int(cls,basis1,basis2):
        """overlap integral"""
        AA = basis1.A*basis2.A

        alpha1 , alpha2 = basis1.alpha , basis2.alpha
        alpha_1p2 = alpha1 + alpha2
         
        x1 , x2 = basis1.x , basis2.x
        dr = x2-x1
        dr2 = np.inner(dr,dr) 
        e12 = m.exp(-alpha1*alpha2/alpha_1p2*dr2)

        G = (pi/alpha_1p2)**(3/2)
        return AA*e12*G 

