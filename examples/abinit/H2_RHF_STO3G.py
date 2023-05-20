import sys
import numpy as np
import math as m
from scipy import special, linalg
from qmodelling.molecule import Atom, Molecule
from qmodelling.func.orbital import Orbital
from qmodelling.func.primitivegaussian import PrimitiveGaussian

pi = np.pi


# compute the matrix operators (add proton-proton interactions and overlap integral)
# H = T + Vee + Vep + Vpp

def overlap_int(molec):
    n = molec.n
    S = np.zeros([n,n])
    
    # double loops over the atoms
    for i in range(n):
        atomi = molec.atoms[i]
        ni = atomi.nbasis
        coeffi = atomi.orbital.coeff
        basisi = atomi.orbital.basis

        for j in range(n):
            atomj = molec.atoms[j]
            nj = atomj.nbasis
            coeffj = atomj.orbital.coeff
            basisj = atomj.orbital.basis

            # double loops over basis functions
            for k in range(ni):
                basis_ik = basisi[k]
        
                for l in range(nj):
                    basis_jl = basisj[l]

                    S[i,j] += coeffi[k]*coeffj[l]*PrimitiveGaussian.overlap_int(basis_ik,basis_jl)

    return S

def T_int(molec):

    n = molec.n
    T = np.zeros([n,n])
    
    # double loops over the atoms
    for i in range(n):
        atomi = molec.atoms[i]
        ni = atomi.nbasis
        coeffi = atomi.orbital.coeff
        basisi = atomi.orbital.basis

        for j in range(n):
            atomj = molec.atoms[j]
            nj = atomj.nbasis
            coeffj = atomj.orbital.coeff
            basisj = atomj.orbital.basis

            # double loops over basis functions
            for k in range(ni):
                basis_ik = basisi[k]
        
                for l in range(nj):
                    basis_jl = basisj[l]

                    T[i,j] += coeffi[k]*coeffj[l]*PrimitiveGaussian.kinetic_int(basis_ik,basis_jl)

    return T

def Vep_int(molec):

    n = molec.n
    Vep = np.zeros([n,n])
    
    for atom in molec.atoms:
        Xp = atom.x
        Z  = atom.Z
        for i in range(n):
            atomi = molec.atoms[i]
            orbiti = atomi.orbital
            coeffi = orbiti.coeff
            basisi = orbiti.basis
            ni = atomi.nbasis

            for j in range(n):
                atomj = molec.atoms[j]
                orbitj = atomj.orbital
                coeffj = orbitj.coeff
                basisj = orbitj.basis
                nj = atomj.nbasis

                for k in range(ni):
                    b1 = basisi[k]
                    for l in range(nj):
                        b2 = basisj[l]

                        Vep[i,j] -= Z*coeffi[k]*coeffj[l]*PrimitiveGaussian.electron_proton_int(b1,b2,Xp) 
    return Vep

def Vee_int(molec):

    n = molec.n
    Vee = np.zeros([n,n,n,n])

    for i in range(n):
        atomi = molec.atoms[i]
        orbiti = atomi.orbital
        coeffi = orbiti.coeff
        basisi = orbiti.basis
        ni = atomi.nbasis

        for j in range(n):
            atomj = molec.atoms[j]
            orbitj = atomj.orbital
            coeffj = orbitj.coeff
            basisj = orbitj.basis
            nj = atomj.nbasis

            for k in range(n):
                atomk = molec.atoms[k]
                orbitk = atomk.orbital
                coeffk = orbitk.coeff
                basisk = orbitk.basis
                nk = atomk.nbasis

                for l in range(n):
                    atoml = molec.atoms[l]
                    orbitl = atoml.orbital
                    coeffl = orbitl.coeff
                    basisl = orbitl.basis
                    nl = atoml.nbasis

                    for u in range(ni):
                        c_u = coeffi[u]
                        b1 = basisi[u]

                        for v in range(nj):
                            c_v = coeffj[v]
                            b2 = basisj[v]

                            for w in range(nk):
                                c_w = coeffk[w]
                                b3 = basisk[w]

                                for t in range(nl):
                                    c_t = coeffl[t]
                                    b4 = basisl[t]

                                    c_uvwt = c_u*c_v*c_w*c_t
                                    vee_ijkl = c_uvwt * PrimitiveGaussian.electron_electron_int(b1,b2,b3,b4) 
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

# functions used in the scf loop
def G_int(molec,P):
    """computing two electron operators"""
    n = molec.n
    G = np.zeros([n,n])

    # computing coulomb integrals
    Vee = Vee_int(molec)

    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    density = P[k,l]
                    J = Vee[i,j,k,l]
                    K = Vee[i,j,l,k]

                    G[i,j] += P[k,l]*(J-0.5*K)
    return G


def P_mat(c,n):
    c = c[:,0:1]
    P = 2.0 * c@c.T   
    return P

def compute_Eelec(P,Hcore,G,n):
    Eelec = np.sum(P*(Hcore+0.5*G))
    return Eelec

# scf loop
def scf_loop(molec,Nmax=200,eps=1e-8):
    E_old = 0.0
    n = molec.n
    P = np.zeros([n,n])

    # compute overlap matrix
    S = overlap_int(molec)
    S12  = linalg.sqrtm(S) 
    S_12 = linalg.inv(S12)

    # integrals
    T   = T_int(molec)
    Vep = Vep_int(molec)
    Hcore = T + Vep
    E = 0.0

    for _ in range(Nmax):
        E_old = E

        # compute Fock operator
        G   = G_int(molec,P) 
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

c1 = 0.1543289673E+00 
c2 = 0.5353281423E+00
c3 = 0.4446345422E+00
coeff = (c1, c2, c3)
n = 100
dlin = np.linspace(0.4,20,n)

Etot = []

for d in dlin:

    x1 = np.array([0.0, 0.0, 0.0])
    x2 = np.array([0.0, 0.0, d  ])
    
    PG11 = PrimitiveGaussian(alpha1,x0=x1)
    PG21 = PrimitiveGaussian(alpha2,x0=x1) 
    PG31 = PrimitiveGaussian(alpha3,x0=x1)  

    PG12 = PrimitiveGaussian(alpha1,x0=x2)
    PG22 = PrimitiveGaussian(alpha2,x0=x2) 
    PG32 = PrimitiveGaussian(alpha3,x0=x2)  

    basis1 = (PG11, PG21, PG31)
    basis2 = (PG12, PG22, PG32)

    orbit1 = Orbital(coeff,basis1)
    orbit2 = Orbital(coeff,basis2)

    Z = 1
    H_1 = Atom(Z,x1,orbit1) 
    H_2 = Atom(Z,x2,orbit2)
    
    HH = (H_1, H_2)
    
    molec = Molecule(HH)
    Eelec = scf_loop(molec)
    Vpp = Vpp_int(molec) 
    Etot.append(Eelec+Vpp)

# output results
import matplotlib.pyplot as plt

plt.plot(dlin,Etot)
plt.savefig("E.png")
