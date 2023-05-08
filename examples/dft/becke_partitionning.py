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

#def p(nu):
#    return 3.0/2.0*nu - 0.5*nu**3

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
    #_, nu21 = confocal_elliptic_coordinates(r,R2,R1) 
    nu21 = -nu12
    s1 = s(nu12,k)
    s2 = s(nu21,k)

    return s1/(s1+s2), s2/(s1+s2)
"""
R1 = np.array([0.0, 0.0])
R2 = np.array([1.0, 0.0])

rlin = np.linspace(0.0,1.0)
S1, S2 = [], []
for rr in rlin:
    r = np.array([rr,1.0])

    s1, s2 = normalized_cell_functions(r,R1,R2)
    S1.append(s1)
    S2.append(s2)

plt.plot(rlin,S1)
plt.plot(rlin,S2)
plt.show()
R = np.random.uniform(size=[5,3])
R1 = np.array([0.0, 0.0, 0.0]).reshape(1,3)
R2 = np.array([1.0, 0.0, 0.0]).reshape(1,3)
print(confocal_elliptic_coordinates(R,R1,R2)[1].shape)
"""
