import cProfile
import pstats
import sys
import numpy as np
import math as m
from copy import deepcopy as dcp
from scipy import special, linalg
#from qmodelling.molecule import Atom, Molecule
#from qmodelling.func.orbital import Orbital
from qmodelling.func.primitivegaussian import PrimitiveGaussian
import matplotlib.pyplot as plt
from becke_partitionning import normalized_cell_functions

pi = np.pi
seed = 123
np.random.seed(seed)

#############################################
# Defining classes
#############################################
class Atom:
    def __init__(self,Z,x):
        self.x = x
        self.Z = Z

class Molecule:
    def __init__(self,atoms):
        self.atoms = atoms

class Orbital:
    def __init__(self,coeff,basis):
        self.coeff = coeff
        self.basis = basis
        self.n = len(coeff)

    def __call__(self,X):
        n = self.n
        coeff = self.coeff
        basis = self.basis
        y = np.zeros((len(X),1))

        for i in range(n):
            y += coeff[i]*basis[i](X)
        return y

#############################################
# computing integrals for operators
#############################################

def overlap_int(orbitals):
    n = len(orbitals)
    S = np.zeros([n,n])
    
    for i in range(n):
        orbital_i = orbitals[i]
        basis_i = orbital_i.basis
        coeff_i = orbital_i.coeff
        ni = len(coeff_i)

        for j in range(n):
            orbital_j = orbitals[j]
            basis_j = orbital_j.basis
            coeff_j = orbital_j.coeff
            nj = len(coeff_j)

            for k in range(ni):
                ck = coeff_i[k]
                bk = basis_i[k]

                for l in range(nj):
                    cl = coeff_j[l]
                    bl = basis_j[l]

                    S[i,j] += ck*cl*PrimitiveGaussian.overlap_int(bk,bl)

    return S

def T_int(orbitals):
    n = len(orbitals)
    T = np.zeros([n,n])

    # double loops over the atoms
    for i in range(n):
        orbital_i = orbitals[i]
        basis_i = orbital_i.basis
        coeff_i = orbital_i.coeff
        ni = len(coeff_i)

        for j in range(n):
            orbital_j = orbitals[j]
            basis_j = orbital_j.basis
            coeff_j = orbital_j.coeff
            nj = len(coeff_j)

            for k in range(ni):
                ck = coeff_i[k]
                bk = basis_i[k]

                for l in range(nj):
                    cl = coeff_j[l]
                    bl = basis_j[l]

                    ckl = ck*cl
                    T[i,j] += ckl*PrimitiveGaussian.kinetic_int(bk,bl)

    return T

def Vep_int(basis,molecule):
    n = len(orbitals)
    Vep = np.zeros([n,n])
    
    for atom in molecule.atoms:
        Xp = atom.x
        Z  = atom.Z

        for i in range(n):
            orbital_i = orbitals[i]
            basis_i = orbital_i.basis
            coeff_i = orbital_i.coeff
            ni = len(coeff_i)

            for j in range(n):
                orbital_j = orbitals[j]
                basis_j = orbital_j.basis
                coeff_j = orbital_j.coeff
                nj = len(coeff_j)

                for k in range(ni):
                    ck = coeff_i[k] 
                    bk = basis_i[k]

                    for l in range(nj):
                        cl = coeff_j[l]
                        bl = basis_j[l]

                        ckl = ck * cl
                        
                        Vep_kl = -Z*ckl*PrimitiveGaussian.electron_proton_int(
                            bk,
                            bl,
                            Xp
                        ) 

                        Vep[i,j] += Vep_kl

    return Vep

def Vee_int(orbitals):
    n = len(orbitals)
    Vee = np.zeros([n,n,n,n])

    # loops over orbitals
    for i in range(n):
        orbital_i = orbitals[i]
        basis_i = orbital_i.basis
        coeff_i = orbital_i.coeff
        ni = len(coeff_i)

        for j in range(n):
            orbital_j = orbitals[j]
            basis_j = orbital_j.basis
            coeff_j = orbital_j.coeff
            nj = len(coeff_j)

            for k in range(n):
                orbital_k = orbitals[k]
                basis_k = orbital_k.basis
                coeff_k = orbital_k.coeff
                nk = len(coeff_k)

                for l in range(n):
                    orbital_l = orbitals[l]
                    basis_l = orbital_l.basis
                    coeff_l = orbital_l.coeff
                    nl = len(coeff_l)

                    # loops over basis functions
                    for ii in range(ni):
                        cii = coeff_i[ii]
                        bii = basis_i[ii]

                        for jj in range(nj):
                            cjj = coeff_j[jj]
                            bjj = basis_j[jj]

                            for kk in range(nk):
                                ckk = coeff_k[kk]
                                bkk = basis_k[kk]

                                for ll in range(nl):
                                    cll = coeff_l[ll]
                                    bll = basis_l[ll]

                                    cijkl = cii*cjj*ckk*cll
                                    Vee[i,j,k,l] += cijkl * PrimitiveGaussian.electron_electron_int(bii,bjj,bkk,bll) 

    return Vee

def Vpp_int(molec):
    n = len(molec.atoms)
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
            rij = linalg.norm(xij)
            Vpp += Zi*Zj/rij

    return Vpp

####################################
# Function used in the scf loop
####################################

def Coulomb(Vee,P):
    """Coulomb interaction
    
    Vee : for electron integrals in AO basis
    P   : density matrix
    """
    Vcoulomb = np.einsum('ijkl,kl->ij',Vee,P) 
    return Vcoulomb

def density(X,P,S):
    """computing density at each points of discretized grid

    X : Points where to compute density
    P : density matrix
    S : overlap matrix
    """
    B = np.hstack([orbital(X) for orbital in orbitals])
    BB = np.einsum('ni,nj->nij',B,B)
    rho = np.einsum('ij,nij->n',P,BB).reshape(-1,1) 

    # L1-normalization the density with Density and Overlap matrices
    norm = np.einsum('ij,ij',P,S)
    return rho/norm

def vx(rho):
    """volumic LDA-functionnal"""
    return -(3.0*rho/pi)**(1/3)


def LDA(orbitals,P,S):
    """LDA operator in atomic orbital basis

    orbitals : list containing AO objects
    P : density matrix
    S : overlap matrix
    """
    n = len(orbitals)
    Vc = 0.0
    Vx = np.zeros([n,n])

    # loop over AO
    for i in range(n):
        orbital_i = orbitals[i]

        for j in range(n):
            orbital_j = orbitals[j]

            # loop over fuzzy cell functions
            for ii in range(n):
                for jj in range(ii+1,n):
                    # center of the first cell
                    R1 = orbitals[ii].x
                    # center of the secind cell
                    R2 = orbitals[jj].x
                    # 1 -> 2
                    # center integrals at center of cell 1
                    Dx = X_cartesian_int - R1
                    rho = density(Dx,P,S)
                    vx_X = vx(rho)
                    # fuzzy cell weight function centered at 1
                    wcell, _ = normalized_cell_functions(Dx,R1=R1,R2=R2)
                    # Chebyshev + Lebedenev quadrature + variable substitution for radial component (R/Mu)
                    Vx[i,j] += np.einsum(
                        'ij,ij',
                        orbital_i(Dx)*orbital_j(Dx)*vx_X*wcell*np.sin(Phi_int)*R_int**2*2.0*rm/(1.0-Mu_int)**2,
                        Wint
                    )
                    # 2 -> 1
                    Dx = X_cartesian_int - R2
                    rho = density(Dx,P,S)
                    vx_X = vx(rho)
                    wcell, _ = normalized_cell_functions(Dx,R1=R2,R2=R1)
                    Vx[i,j] += np.einsum(
                        'ij,ij',
                        orbital_i(Dx)*orbital_j(Dx)*vx_X*wcell*np.sin(Phi_int)*R_int**2*2.0*rm/(1.0-Mu_int)**2,
                        Wint
                    )

    return Vx,Vc


def density_matrix(c):
    """RHF density matrix"""
    c = c[:,0:1]
    return  2.0 * c@c.T

def exc(rho):
    """volumic exchance-correlation energy"""
    return -(3.0/4.0)*(3.0/pi*rho)**(1/3)

def energy(orbitals,Hcore,Vcoulomb,S,P):
    """computing energy of the system
    orbitals : list of AO objects
    Hcore : Core Hamiltonian in AO basis
    Vcoulomb : Coulomb operator in A0 basis
    S : overlap matrix
    P : density matrix
    """
    Ecore    = np.sum(P*Hcore)
    Ecoulomb = np.sum(P*Vcoulomb)/2.0
    #rho      = density(Xint,orbitals,P)
    #Exc      = np.sum(rho*exc(rho))*dx**3

    Exc = 0.0
    n = len(orbitals)
    # loop over fuzzy cells
    for ii in range(n):
        for jj in range(ii+1,n):
            R1 = orbitals[ii].x
            R2 = orbitals[jj].x
            # 1 -> 2
            Dx = X_cartesian_int - R1
            rho = density(Dx,P,S)
            wcell, _ = normalized_cell_functions(Dx,R1=R2,R2=R1)
            Exc += np.einsum(
                'ij,ij',
                rho*exc(rho)*np.sin(Phi_int)*R_int**2*2.0*rm/(1.0-Mu_int)**2,
                Wint
            )
            # 2 -> 1
            Dx = X_cartesian_int - R2
            rho = density(Dx,P,S)
            wcell, _ = normalized_cell_functions(Dx,R1=R2,R2=R1)
            Exc += np.einsum(
                'ij,ij',
                rho*exc(rho)*np.sin(Phi_int)*R_int**2*2.0*rm/(1.0-Mu_int)**2,
                Wint
            )
    return Ecore , Ecoulomb , Exc

def scf_loop(orbitals,molecule,params):

    tol = params['tol']
    Niter = params['Niter']
    n = len(orbitals)

#    # computing product of AO at each integration points
#    B = np.hstack([orbital(Xint) for orbital in orbitals])
#    BB = np.einsum('ni,nj->nij',B,B)

    # density matrix
    P = np.random.uniform(size=[n,n])#np.zeros([n,n])

    # compute integrals 
    T   = T_int(orbitals)
    Vep = Vep_int(orbitals,molecule)
    Vee = Vee_int(orbitals)
    S = overlap_int(orbitals)
    S12 = linalg.sqrtm(S)
    S_12 = linalg.inv(S12)
    del(S12)

    # core Hamiltonian
    Hcore = T + Vep

    # high enough initial random value
    E = 1e2 

    for _ in range(Niter):

        # save Energy from last iteration
        E_old = E
        
        # coulomb interactions
        Vcoulomb = Coulomb(Vee,P)

        # exchange-correlation functionnal
        Vx, Vc = LDA(orbitals,P,S)

        # Fock operator 
        F = Hcore + Vcoulomb + Vx + Vc

        # solve normalized equation system
        Fu = np.dot(S_12,np.dot(F,S_12))
        eigval, eigvec = linalg.eigh(Fu)
        c = np.dot(S_12,eigvec)

        # density matrix
        P = density_matrix(c)

        # compute new density
        #rho = density(Xint,orbitals,P)
        # rho = density(X_cartesian_int,BB,P)

        # compute energy
        Ecore , Ecoulomb , Exc = energy(orbitals,Hcore,Vcoulomb,S,P)
        E = Ecore + Ecoulomb + Exc

        if abs(E-E_old) < tol:
        #if abs(E-E_old) < tol*abs(E):
            return Ecore , Ecoulomb , Exc, c, eigval

    print('Not converged...')
    return Ecore , Ecoulomb , Exc, c, eigval

####################################
# Training loop over each configuration
####################################
# Quadrature weights/points
# for radial component
Nr = 30
ii  = np.arange(1,Nr+1).reshape(-1,1) 
sin_i = np.sin(ii*pi/(Nr+1))
cos_i = np.cos(ii*pi/(Nr+1))

# Chebyshev quadrature points on [-1,1]
Mu_quadrature = (Nr+1.0-2.0*ii)/(Nr+1.0) + 2.0/pi*(1.0 + 2.0/3.0*sin_i**2)*cos_i*sin_i
# Chebyshev quadrature points on [0, +infty [
# Bragg-slater Radius for Hydrogen
rm = 0.35 # Angstrom
# variable substitution for the radial component 
R_quadrature = rm*(1+Mu_quadrature)/(1-Mu_quadrature)
# Chebyshev quadrature weights
Wr_quadrature = 16.0/3.0/(Nr+1)*sin_i**4

# for solid angle / angular coordinates
# file containing integration points (theta/phi in deg) and associated weights
fn = 'lebedenev_quadrature.txt'
Data = np.loadtxt(fn)
ThetaPhi_quadrature = Data[:,:2]
# convert to rad
ThetaPhi_quadrature = ThetaPhi_quadrature*pi/180.0
Wangular_quadrature = Data[:,2:3]

# make a tensor product of the Quadrature points
X_spherical_int = np.concatenate([
        np.tile(R_quadrature,(len(ThetaPhi_quadrature),1)),
        np.repeat(ThetaPhi_quadrature,len(R_quadrature),axis=0)   
    ],
    axis=1
)
Phi_int = X_spherical_int[:,2:3]
R_int   = X_spherical_int[:,0:1]
Mu_int  = (R_int - rm)/(R_int + rm)
Wint = np.concatenate([
        np.tile(Wr_quadrature,(len(Wangular_quadrature),1)),
        np.repeat(Wangular_quadrature,len(Wr_quadrature),axis=0)   
    ],
    axis=1
)
Wint = np.prod(Wint,axis=1,keepdims=True)

X_cartesian_int = np.zeros_like(X_spherical_int)
R, Theta, Phi = X_spherical_int[:,0], X_spherical_int[:,1], X_spherical_int[:,2]
X_cartesian_int[:,0] = R*np.cos(Theta)*np.sin(Phi)
X_cartesian_int[:,1] = R*np.sin(Theta)*np.sin(Phi)
X_cartesian_int[:,2] = R*np.cos(Phi)

del(ii,sin_i,cos_i)
del(Mu_quadrature,R_quadrature,Wr_quadrature)
del(ThetaPhi_quadrature,Wangular_quadrature)
del(R,Theta,Phi)

# parameters of the simulation
# coefficients of the gaussians
alpha1 = 0.3425250914E+01
alpha2 = 0.6239137298E+00
alpha3 = 0.1688554040E+00
# coefficients of linear combination
c1 = 0.1543289673E+00 
c2 = 0.5353281423E+00
c3 = 0.4446345422E+00
coeff = (c1,c2,c3)

# distance between the Hs
n = 20
dlin = np.linspace(0.4,15,n)

# parameters of the scp loop
params = {'tol':1e-7,'Niter':200}

E_list = []
Ecore_list = []
Ecoulomb_list = []
Exc_list = []
Epp_list = []

#with cProfile.Profile() as profile:
for i,d in enumerate(dlin):

    # defining the molecule
    x1 = np.array([0.0, 0.0, 0.0]).reshape(1,-1)
    x2 = np.array([0.0, 0.0,   d]).reshape(1,-1)
    H1 = Atom(Z=1,x=x1)
    H2 = Atom(Z=1,x=x2)
    atoms = (H1,H2)
    molecule = Molecule(atoms)

    # defining basis functions then orbitals
    # first Hydrogen
    PG_H1_1 = PrimitiveGaussian(alpha=alpha1,x0=x1) 
    PG_H1_2 = PrimitiveGaussian(alpha=alpha2,x0=x1) 
    PG_H1_3 = PrimitiveGaussian(alpha=alpha3,x0=x1) 
    basisH1 = (PG_H1_1, PG_H1_2,PG_H1_3)
    orbitalH1 = Orbital(coeff,basisH1)
    orbitalH1.x = x1

    # second Hydrogen
    PG_H2_1 = PrimitiveGaussian(alpha=alpha1,x0=x2) 
    PG_H2_2 = PrimitiveGaussian(alpha=alpha2,x0=x2) 
    PG_H2_3 = PrimitiveGaussian(alpha=alpha3,x0=x2) 
    basisH2 = (PG_H2_1, PG_H2_2,PG_H2_3)
    orbitalH2 = Orbital(coeff,basisH2)
    orbitalH2.x = x2

    orbitals = (orbitalH1, orbitalH2)

    # scf loop
    Ecore , Ecoulomb , Exc, P, eigval = scf_loop(orbitals,molecule,params)
    Eelec = Ecore + Ecoulomb + Exc

    # proton-proton interaction
    Epp = Vpp_int(molecule)

    E_list.append(Epp+Eelec)
    Ecore_list.append(Ecore)
    Ecoulomb_list.append(Ecoulomb)
    Exc_list.append(Exc)
    Epp_list.append(Epp)
    print(f"Opti {i+1}/{n} done!")
#    print(eigval)
#    print('-'*10)
#    print(P)
#    print('-'*10)
#    print('-'*10)

#results = pstats.Stats(profile)
#results.sort_stats(pstats.SortKey.TIME)
#profile.dump_stats('results.prof')

plt.figure(1)
plt.plot(dlin,E_list,marker='x',label='total')
plt.plot(dlin,Ecore_list,marker='x',label='Core (kinetic + ep)')
plt.plot(dlin,Ecoulomb_list,marker='x',label='Coulomb')
plt.plot(dlin,Exc_list,marker='x',label='xc')
plt.plot(dlin,Epp_list,marker='x',label='pp')
plt.legend()
plt.savefig('Es.png')

plt.figure(2)
plt.plot(dlin,E_list,marker='x')
plt.xlabel('x (Angstrom)')
plt.ylabel('E (Hartree)')
plt.savefig('E.png')
