from qmodelling.func.basis import Basis
import numpy as np
import math as m
from scipy import special
 
pi = np.pi

def boys(x,n):
    if x == 0:
        return 1.0/(2*n+1)
    else:
        return special.gammainc(n+0.5,x) * special.gamma(n+0.5) * (1.0/(2*x**(n+0.5)))

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
    def electron_electron_int(cls,basis1,basis2,basis3,basis4):
        """electron-electron integral"""
        A1234 = basis1.A * basis2.A * basis3.A * basis4.A
        
        x1 = basis1.x
        x2 = basis2.x
        x3 = basis3.x
        x4 = basis4.x

        a1 = basis1.alpha
        a2 = basis2.alpha
        a3 = basis3.alpha
        a4 = basis4.alpha

        a14  = a1 * a4
        ap14 = a1 + a4
        x14 = x1 - x4 
        x14_2 = np.inner(x14,x14)
        e14 = m.exp(-a14/ap14*x14_2)
        
        a23  = a2 * a3
        ap23 = a2 + a3
        x23 = x2 - x3 
        x23_2 = np.inner(x23,x23)
        e23 = m.exp(-a23/ap23*x23_2)

        ap1234 = a14 + a23
        
        x23_mean = (a2*x2+a3*x3)/(a2+a3)
        x14_mean = (a1*x1+a4*x4)/(a1+a4)
        x23_14 = x23_mean - x14_mean
        Q2 = np.inner(x23_14,x23_14)

        return A1234*e14*e23*2.0*pi**2*m.sqrt(pi/ap1234)*boys(a23*a14/ap1234*Q2,0)


    @classmethod
    def electron_proton_int(cls,basis1,basis2,Xp):
        """electron-proton integral"""
        A12 = basis1.A*basis2.A
        a1 , a2 = basis1.alpha , basis2.alpha
        x1 , x2 = basis1.x , basis2.x
        p = a1 + a2

        a12 = a1 * a2
        dx1_2   = x1 - x2
        dx1_22  = np.inner(dx1_2,dx1_2)

        x12mean = (a1*x1+a2*x2)/p
        dxp_12 = x12mean - Xp
        dxp_122 = np.inner(dxp_12,dxp_12)
        return 2.0 * A12 * (pi/p) * m.exp(-a12/p*dx1_22) * boys(p*dxp_122,0)

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

