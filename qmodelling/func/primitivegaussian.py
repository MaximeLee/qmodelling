from qmodelling.func.basis import Basis
import numpy as np
import math as m
from scipy import special
 
pi = np.pi

def boys(x):
    n = 0
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

    def __call__(self,x):
        x0 = self.x
        dxx = x-x0 
        dxx2 = np.sum(dxx**2,axis=1)
        return self.A*np.exp(-self.alpha*dxx2)

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

        a12  = a1 * a2
        ap12 = a1 + a2
        x12 = x1 - x2 
        x12_2 = np.inner(x12,x12)
        e12 = m.exp(-a12/ap12*x12_2)
        
        a43  = a4 * a3
        ap43 = a4 + a3
        x43 = x4 - x3 
        x43_2 = np.inner(x43,x43)
        e43 = m.exp(-a43/ap43*x43_2)

        ap1234 = ap12 + ap43
        
        x43_mean = (a4*x4+a3*x3)/(ap43)
        x12_mean = (a1*x1+a2*x2)/(ap12)
        x12_43 = x12_mean - x43_mean
        Q2 = np.inner(x12_43,x12_43)

        # return A1234*e14*e23*2.0*pi**2/(ap23+ap14)*m.sqrt(pi/ap1234)*boys(ap23*ap14/ap1234*Q2,0)
        return A1234*2.0*pi**2/(ap12*ap43)*e43*e12*m.sqrt(pi/ap1234)*boys((ap43*ap12/ap1234)*Q2)


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
        return 2.0 * A12 * (pi/p) * m.exp(-a12/p*dx1_22) * boys(p*dxp_122)

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

