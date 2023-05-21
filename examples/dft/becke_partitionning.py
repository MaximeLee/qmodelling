import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt

def confocal_elliptic_coordinates(r,R1,R2):
    """confocal elliptic coordinates for diatomic molecules
    R1/2 : cartesian position of nuclei 1/2
    r    : cartesian position in space
    """

    # internuclear distance
    R12 = lg.norm(R1-R2,axis=1,keepdims=True)
    # distance to nuclei1
    r1  = lg.norm(R1-r,axis=1,keepdims=True)
    # distance to nuclei2
    r2  = lg.norm(R2-r,axis=1,keepdims=True)

    lbda = (r1 + r2)/R12 # lambda
    nu   = (r1 - r2)/R12
    return lbda, nu

def pk(nu,k=3):
    """antisymetric function respecting the conditions"""
    out = nu
    for _ in range(k):
        #out = 3.0/2.0*out - 0.5*out**3 #p(out)
        out = 0.5*out*(3.0-out**2)
    return out

def s(nu,k=3):
    """cutoff function"""
    return 0.5*(1.0 - pk(nu,k))

def normalized_cell_functions(r,R1,R2,k=3):
    """cell function for diatomic systems"""
    _, nu12 = confocal_elliptic_coordinates(r,R1,R2)
    _, nu21 = confocal_elliptic_coordinates(r,R2,R1)

    s1 = s(nu12,k)
    s2 = s(nu21,k)

    return s1/(s1+s2), s2/(s1+s2)
