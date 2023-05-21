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
seed = 12
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

        return y.reshape(-1,1)

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

def density(rho,orbitals,P,S,BB):
    """computing density at each points of discretized grid

    X : Points where to compute density
    P : density matrix
    S : overlap matrix
    """
    n = len(orbitals)

    # computing volumic functionnal potential beforehand
    for ii in range(n):
        for jj in range(ii+1,n):
            # center of the first cell
            R1 = orbitals[ii].x
            # center of the secind cell
            R2 = orbitals[jj].x

            # 1 -> 2
            # centering integrals at center of cell 1
            Dx = X_cartesian_int + R1
            rho[ii,jj,0] = np.einsum('ij,nij->n',P,BB[ii,jj,0]).reshape(-1,1) 

            # 2 -> 1
            # centering integrals at center of cell2
            Dx = X_cartesian_int + R2
            rho[ii,jj,1] = np.einsum('ij,nij->n',P,BB[ii,jj,1]).reshape(-1,1) 

    # L1-normalization the density with Density and Overlap matrices
    norm = np.einsum('ij,ij',P,S)
    return rho/norm

def LDA(orbitals,rho):
    """LDA operator in atomic orbital basis

    orbitals : list containing AO objects
    P : density matrix
    S : overlap matrix
    """
    n = len(orbitals)
    Vxc = np.zeros([n,n])

    # storing values of exchange-correlation function evaluated at 
    # quadrature points centered at each fuzz cell centers
    Vxc_X = np.empty([n,n,2,len(X_cartesian_int),1])

    # computing volumic functionnal potential beforehand
    for ii in range(n):
        for jj in range(ii+1,n):
            # 1 -> 2
            # centering integrals at center of cell 1
            vxc_X = vxc(rho[ii,jj,0])
            Vxc_X[ii,jj,0] = vxc_X

            # 2 -> 1
            # centering integrals at center of cell2
            vxc_X = vxc(rho[ii,jj,1])
            Vxc_X[ii,jj,1] = vxc_X

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
                    vxc_X = Vxc_X[ii,jj,0]
                    # centering integrals
                    Dxx = X_cartesian_int + R1
                    # fuzzy cell weight function centered at 1
                    wcell, _ = normalized_cell_functions(Dxx,R1,R2)
                    # Chebyshev + Lebedenev quadrature + variable substitution for radial component (R/Mu)
                    Vxc[i,j] += 4.0*pi*np.einsum(
                        'ij,ij',
                        wcell*orbital_i(Dxx)*orbital_j(Dxx)*vxc_X*subs,
                        Wint
                    )

                    # 2 -> 1
                    vxc_X = Vxc_X[ii,jj,1]
                    Dxx = X_cartesian_int + R2
                    _, wcell = normalized_cell_functions(Dxx,R1,R2)
                    Vxc[i,j] += 4.0*pi*np.einsum(
                        'ij,ij',
                        wcell*orbital_i(Dxx)*orbital_j(Dxx)*vxc_X*subs,
                        Wint
                    )

    return Vxc


def density_matrix(c):
    """RHF density matrix"""
    c = c[:,0:1]
    return  2.0 * c@c.T

# Vosko, Wilk, Nusair correlation constants
A = 0.0621814
x0 = -0.409286
b = 13.072
c = 42.7198
Q = (4*c-b**2)**0.5
eps = 1e-6
rinf = 1e6

def exc(rho):
    """volumic exchance-correlation energy"""
    # Dirac exchange
    ex  = -(3.0/4.0)*(3.0/pi*rho)**(1/3)
    ex *= 9.0/8.0
    #ex = -(3.0)*(3.0/4.0/pi*rho)**(1.0/3.0)

    # Vosko, Wilk, Nusair correlation
    def X(arg):
        return arg**2.0 + b*arg + c

    rs = np.where(rho>eps,(3.0/(4.0*pi*(rho+1e-8)))**(1.0/3.0),rinf)
    x  = np.sqrt(rs)
    term1 = np.log(x**2/X(x))
    term2 = 2*b/Q*np.arctan(Q/(2*x+b))
    term3 = b*x0/X(x0)*(np.log((x-x0)**2/X(x)) + 2*(b+2*x0)/Q*np.arctan(Q/(2*x+b)) )

    ec = A/2.0*(term1 + term2 - term3 )
    return ex + ec 

def vxc(rho):
    """volumic LDA-functionnal"""
    # Dirac exchange
    vx = -9.0/8.0*(3.0*rho/pi)**(1.0/3.0)
    #vx = -4.0*(3.0/4.0*rho/pi)**(1.0/3.0)

    # Vosko, Wilk, Nusair correlation
    def X(arg):
        return arg**2.0 + b*arg + c

    rs = np.where(rho>eps,(3.0/(4.0*pi*(rho+1e-8)))**(1.0/3.0),rinf)
    x  = np.sqrt(rs)
    Xx = X(x)
    term1 = np.log(x**2/Xx)
    term2 = 2*b/Q*np.arctan(Q/(2*x+b))
    term3 = b*x0/X(x0)*(np.log((x-x0)**2/Xx) + 2*(b+2*x0)/Q*np.arctan(Q/(2*x+b)) )

    ec = A/2.0*(term1 + term2 - term3 )

    term1 = (2*Xx-x*(2*x+b))/(Xx*x) 
    term2 = -4*b/((2*x+b)**2 + Q**2) 
    term31 = (2*Xx-(2*x+b)*(x-x0))/(Xx*(x-x0))
    term32 = -4*(2*x0+b)/((2*x+b)**2 + Q**2)
    term3 = b*x0/X(x0)*(term31 + term32) 
    ec_rho = A/2.0*(term1 + term2 - term3 )

    vc = rho*ec_rho + ec
#    vc = 0.0
    return vx + vc

def energy(rho,orbitals,Hcore,Vcoulomb,P):
    """computing energy of the system
    orbitals : list of AO objects
    Hcore : Core Hamiltonian in AO basis
    Vcoulomb : Coulomb operator in A0 basis
    S : overlap matrix
    P : density matrix
    """
    Ecore    = np.sum(P*Hcore)
    Ecoulomb = np.sum(P*Vcoulomb*0.5)

    Exc = 0.0

    # loop over fuzzy cells
    n = len(orbitals)
    for ii in range(n):
        for jj in range(ii+1,n):
            R1 = orbitals[ii].x
            R2 = orbitals[jj].x

            # 1 -> 2
            Dx = X_cartesian_int + R1
            wcell, _ = normalized_cell_functions(Dx,R1,R2)
            Exc += 4.0*pi*np.einsum(
                'ij,ij',
                wcell*rho[ii,jj,0]*exc(rho[ii,jj,0])*subs,
                Wint
            )

            # 2 -> 1
            Dx = X_cartesian_int + R2
            _, wcell = normalized_cell_functions(Dx,R1,R2)
            Exc += 4.0*pi*np.einsum(
                'ij,ij',
                wcell*rho[ii,jj,1]*exc(rho[ii,jj,1])*subs,
                Wint
            )
    return Ecore , Ecoulomb , Exc

def scf_loop(orbitals,molecule,params):

    tol = params['tol']
    Niter = params['Niter']
    n = len(orbitals)
    # damping constant
    #alpha = 0.5

    # values of product of the AO over space
    BB = np.zeros([n,n,2,len(X_cartesian_int),n,n])
    for ii in range(n):
        for jj in range(ii+1,n):
            # center of the first cell
            R1 = orbitals[ii].x
            # center of the secind cell
            R2 = orbitals[jj].x

            # 1 -> 2
            # centering integrals at center of cell 1
            Dx = X_cartesian_int + R1
            B = np.hstack([orbital(Dx) for orbital in orbitals])
            BB[ii,jj,0] = np.einsum('ni,nj->nij',B,B)

            # 2 -> 1
            # centering integrals at center of cell2
            Dx = X_cartesian_int + R2
            B = np.hstack([orbital(Dx) for orbital in orbitals])
            BB[ii,jj,1] = np.einsum('ni,nj->nij',B,B)

    # density matrix
    #P = np.random.uniform(size=[n,n])
    P = np.zeros([n,n])

    # values of density at quadrature points around each Hydrogen atom 
    # initial density at zero
    rho = np.zeros([n,n,2,len(X_cartesian_int),1]) 

    # compute integrals 
    T   = T_int(orbitals)
    Vep = Vep_int(orbitals,molecule)
    Vee = Vee_int(orbitals)

    S = overlap_int(orbitals)
    S12 = linalg.sqrtm(S)
    S_12 = linalg.inv(S12)

    # core Hamiltonian
    Hcore = T + Vep

    # high enough initial random value
    E = np.inf

    for _ in range(Niter):

        # save Energy from last iteration
        #P_old = dcp(P)
        E_old = E
        
        # coulomb interactions
        Vcoulomb = Coulomb(Vee,P)

        # exchange-correlation functionnal
        Vx, Vc = LDA(orbitals,rho)
    
        # Fock operator 
        F = Hcore + Vcoulomb + Vx + Vc

        # solve normalized equation system
        Fu = np.dot(S_12,np.dot(F,S_12))
        eigval, eigvec = linalg.eigh(Fu)
        c = np.dot(S_12,eigvec)

        # density matrix
        P = density_matrix(c)
#        P = alpha*P + (1.0-alpha)*P_old

        # compute new density
        #rho = density(Xint,orbitals,P)
        rho = density(rho,orbitals,P,S,BB)

        # compute energy
        Ecore , Ecoulomb , Exc = energy(rho,orbitals,Hcore,Vcoulomb,P)
        E = Ecore + Ecoulomb + Exc

        if abs(E-E_old) < tol:
            return Ecore , Ecoulomb , Exc, c, eigval

    print('Not converged...')
    return Ecore , Ecoulomb , Exc, c, eigval

####################################
# Training loop over each configuration
####################################
# Quadrature weights/points
# for radial component
Nr = 50
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
R_int   = X_spherical_int[:,0:1]
Phi_int = X_spherical_int[:,2:3]
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

subs = R_int**2*2.0*rm/(1.0-Mu_int)**2

# parameters of the simulation
# coefficients of the gaussians
alpha1 = 0.3425250914E+01
alpha2 = 0.6239137298E+00
alpha3 = 0.1688554040E+00
# coefficients of linear combination
c1 = 0.1543289673E+00 
c2 = 0.5353281423E+00
c3 = 0.4446345422E+00
#coeff = (c1,c2,c3)
coeff = np.array([c1,c2,c3])

# distance between the Hs
n = 100
dlin = np.linspace(0.4,20,n)

# parameters of the scp loop
params = {'tol':1e-6,'Niter':200}

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
    Ecore , Ecoulomb , Exc, _, _ = scf_loop(orbitals,molecule,params)
    Eelec = Ecore + Ecoulomb + Exc

    # proton-proton interaction
    Epp = Vpp_int(molecule)

    E_list.append(Epp+Eelec)
    Ecore_list.append(Ecore)
    Ecoulomb_list.append(Ecoulomb)
    Exc_list.append(Exc)
    Epp_list.append(Epp)

    print(f"Opti {i+1}/{n} done!")

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
plt.xlabel('x (Angstrom)')
plt.savefig('Es_becke.png')

plt.figure(2)
plt.plot(dlin,E_list,marker='x')
plt.xlabel('x (Angstrom)')
plt.ylabel('E (Hartree)')
plt.savefig('E_becke.png')

plt.figure(3)
plt.title('Core energy (kinetic + proton-electron interaction) energy')
plt.plot(dlin,Ecore_list,marker='x')
plt.xlabel('x (Angstrom)')
plt.ylabel('E (Hartree)')
plt.savefig('Ecore.png')

plt.figure(4)
plt.title('Electron Coulomb interaction energy')
plt.plot(dlin,Ecoulomb_list,marker='x')
plt.xlabel('x (Angstrom)')
plt.ylabel('E (Hartree)')
plt.savefig('Ecoulomb.png')

plt.figure(5)
plt.title('Exchange energy')
plt.plot(dlin,Exc_list,marker='x')
plt.xlabel('x (Angstrom)')
plt.ylabel('E (Hartree)')
plt.savefig('Exc_VWN.png')

plt.figure(6)
plt.title('Proton repulsion energy')
plt.plot(dlin,Epp_list,marker='x')
plt.xlabel('x (Angstrom)')
plt.ylabel('E (Hartree)')
plt.savefig('Epp.png')
