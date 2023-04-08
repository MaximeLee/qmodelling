import cProfile
import pstats
import sys
import numpy as np
import math as m
from copy import deepcopy as dcp
from scipy import special, linalg
from qmodelling.molecule import Atom, Molecule
from qmodelling.func.orbital import Orbital
from qmodelling.func.primitivegaussian import PrimitiveGaussian

pi = np.pi


# compute the matrix operators (add proton-proton interactions and overlap integral)
# H = T + Vee + Vep + Vpp

def overlap_int(basis):
    n = len(basis)
    S = np.zeros([n,n])
    
    # double loops over the atoms
    for i, basis_i in enumerate(basis):
        for j, basis_j in enumerate(basis):

            S[i,j] = PrimitiveGaussian.overlap_int(basis_i,basis_j)

    return S

def T_int(basis):

    n = len(basis)
    T = np.zeros([n,n])
    
    # double loops over the atoms
    for i, basis_i in enumerate(basis):
        for j, basis_j in enumerate(basis):

            T[i,j] = PrimitiveGaussian.kinetic_int(basis_i,basis_j)

    return T

def Vep_int(basis,molec):

    n = len(basis)
    Vep = np.zeros([n,n])
    
    for atom in molec.atoms:
        Xp = atom.x
        Z  = atom.Z

        for i, basis_i in enumerate(basis):
            for j, basis_j in enumerate(basis):

                Vep[i,j] = -Z*PrimitiveGaussian.electron_proton_int(basis_i,basis_j,Xp) 
    return Vep

def Vee_int(basis):

    n = len(basis)
    Vee = np.zeros([n,n,n,n])

    for i, bi in enumerate(basis):
        for j, bj in enumerate(basis):
            for k, bk in enumerate(basis):
                for l, bl in enumerate(basis):
                    vee_ijkl = PrimitiveGaussian.electron_electron_int(bi,bj,bk,bl) 
                    Vee[i,j,k,l] += vee_ijkl 

    return Vee

def Vpp_int(molec):
    n = molec.n
    Vpp = 0.0
    
    for i in range(n):
        atomi = molec.atoms[i]
        Zi = atomi.Z
        xi = atomi.x
        for j in range(i+1,n):
            atomj = molec.atoms[j]
            Zj = atomj.Z
            xj = atomj.x
            
            xij = xi - xj
            rij2 = np.inner(xij,xij)
            Vpp += Zi*Zj/m.sqrt(rij2)

    return Vpp

# Gunnarson and Lundwvist exchange-correlation functionnal
"""
def G(x):
    return 0.5*((1+x)*m.log(1+1/x)-x**2+x/2-1/3)

def G_prime(x):
    return 0.5*(m.log(1+1/x)-1/x-2*x+1/2)

def density(x,basis,P):
    B = np.array([[b1(x)*b2(x) for b1 in basis] for b2 in basis])
    return np.sum(P*B)

def exc(x,basis,P):
    rho =  density(x,basis,P)
    
    if m.isclose(rho,0.0): return 0.0

    rs = (3/4/pi/rho)**(1/3)
    y = -0.458/rs - 0.0666*G(rs/11.4)
    return y

def exc_prime(x,basis,P):
    rho =  density(x,basis,P)

    if m.isclose(rho,0.0): return 0.0

    rs = (3/4/pi/rho)**(1/3)

    dy1 = -0.458*(4*pi/3)**(1/3)/3*rho**(-2/3)
    dy2 = -0.0666/11.4*(3/4/pi)**(1/3)*(-1/3*rho**(-4/3))*G_prime(rs/11.4)
    dy = dy1 + dy2 
    return dy

def Vxc(x,basis,P):
    rho = density(x,basis,P)
    if m.isclose(rho,0.0): return 0.0
    rs = (3/4/pi/rho)**(1/3)
    
    vxc = rho*exc_prime(x,basis,P) + exc(x,basis,P)
    return vxc
"""

def density(x,basis,P):
    B = np.array([[b1(x)*b2(x) for b1 in basis] for b2 in basis])
    return np.sum(P*B)

# simple LDA approximation (only exchange part no correlation)
def Vxc(x,basis,P):
    rho = density(x,basis,P)
    vxc = -(3/pi)**(1/3)*rho #rho*exc_prime(x,basis,P) + exc(x,basis,P)
    return vxc

# functions used in the scf loop
def double_int(basis,P,X):
    """computing two electron operators"""
    n = len(basis)

    # computing coulomb repulsion integrals
    Vee = Vee_int(basis)

#    d1 = np.zeros([n,n])
#    for i in range(n):
#        for j in range(n):
#            for k in range(n):
#                for l in range(n):
#
#                    J = Vee[i,j,k,l]
#                    d1[i,j] += P[k,l]*J
    d1 = np.einsum('ijkl,kl->ij',Vee,P)

    # computing the exchange-repulsion term with Riemann integrals over (0,1]
    dx = linalg.norm(X[1]-X[0])
    d2 = np.zeros([n,n])

    for i,basis_i in enumerate(basis):
        for j,basis_j in enumerate(basis):
            for r in X:

                d2[i,j] += basis_i(r)*Vxc(r,basis,P)*basis_j(r)

    d2 *= dx
    return d1 + d2


def P_mat(c,n):
    c = c[:,0:1]
    P = 2.0 * c@c.T   
    return P

def compute_Eelec(P,Hcore,G,n):
    Eelec = np.sum(P*(Hcore+0.5*G))
    return Eelec

# scf loop
def scf_loop(basis,molec,X,Nmax=1,eps=1e-8):
    n = len(basis)
    P = np.zeros([n,n])

    # compute overlap matrix
    S = overlap_int(basis)
    S12  = linalg.sqrtm(S) 
    S_12 = linalg.inv(S12)

    # integrals
    T   = T_int(basis)
    Vep = Vep_int(basis,molec)
    Hcore = T + Vep
    E = 0.0

    for _ in range(Nmax):
        E_old = E

        # compute Fock operator
        G = double_int(basis,P,X) 
        F = Hcore + G

        # solve secular equation
        Fu = np.dot(S_12,np.dot(F,S_12))
        eigval , eigvec = linalg.eigh(Fu)
        c = np.dot(S_12,eigvec)

        P = P_mat(c,n)
        E = compute_Eelec(P,Hcore,G,n) 

        if abs(E-E_old)<eps:
            print("Converged !")
            return E

    print("Not converged ...")
    return E

# defining the molecule
alpha1 = 0.3425250914E+01
alpha2 = 0.6239137298E+00
alpha3 = 0.1688554040E+00

PG1 = PrimitiveGaussian(alpha1)
PG2 = PrimitiveGaussian(alpha2) 
PG3 = PrimitiveGaussian(alpha3) 

c1 = 0.1543289673E+00 
c2 = 0.5353281423E+00
c3 = 0.4446345422E+00

coeff = None #(c1, c2, c3)

basisH1 = (PG1, PG2, PG3) 
basisH2 = dcp(basisH1) 


# training + results
n = 1 #500
Z = 1
dlin = np.linspace(0.4,10,n)

basis = (*basisH1,*basisH2)
orbit = Orbital(None,basis)

Etot = []

nint = 1000
xlin = np.linspace(-1e6,1e6,nint)
z = np.zeros(nint)
X = np.stack([z,z,xlin],axis=1)

with cProfile.Profile() as profile:
    for d in dlin:
    
        x1 = np.array([0.0, 0.0, 0.0])
        x2 = np.array([0.0, 0.0, d  ])
    
        for b in basisH1:
            b.x = x1
        
        for b in basisH2:
            b.x = x2
    
        H_1 = Atom(Z,x1,orbit) 
        H_2 = Atom(Z,x2,orbit)
        
        HH = (H_1, H_2)
        
        molec = Molecule(HH)
        Eelec = scf_loop(basis,molec,X)
        Vpp = Vpp_int(molec) 
        Etot.append(Eelec+Vpp)

res = pstats.Stats(profile)
res.sort_stats(pstats.SortKey.TIME)
res.dump_stats("result.prof")

# output results
import matplotlib.pyplot as plt

plt.plot(dlin,Etot)
plt.savefig("E.png")
