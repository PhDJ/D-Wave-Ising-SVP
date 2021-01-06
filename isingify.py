import numpy as np
import scipy.sparse as spsp

from pauli import sigma_x, sigma_y, sigma_z, sigma_identity

X, Y, Z, I = sigma_x, sigma_y, sigma_z, sigma_identity


def svp_isingcoeffs_bin(Gram, sitek):
    """
Turns an n-dimensional SVP problem defined by a Grammian matrix and
a coefficient-range parameter k into an Ising model specification on
n*(ceil(log2(k))+1) spins, where eigenvalues correspond to squared-norms
of vectors in the lattice defined by the SVP problem.

Parameters
----------
Gram  : Grammian matrix as array of shape (n, n)
sitek : a parameter specifying the range of coefficients
   that will multiply each basis vector. The available
   coefficients will be in [-k,-k+1,...0,...,k-2,k-1].

Returns
-------
Jmat                 : array of shape (m, m), where m = n*(ceil(log2(k))+1),
                  containing the coupling (ZZ) coefficients of the Ising 
                  Hamiltonian
hvec                 : array of shape (m,) containg the field (Z) coefficients
                  of the Ising Hamiltonian
identity_coefficient : a float representing the scalar that multiplies the
                  identity term that is added to the Ising Hamiltonian
    """
    nqudits = Gram.shape[0]
    qubits_per_qudit = int(np.ceil(np.log2(sitek))) + 1
    nqubits = nqudits * qubits_per_qudit
    Jmat = np.zeros((nqubits, nqubits))
    hvec = np.zeros(nqubits)
    identity_coefficient = 0

    cn = 0.5

    for m in range(nqudits):
        for l in range(nqudits):
            for j in range(qubits_per_qudit):
                mj_qubit = (m * qubits_per_qudit) + j
                for k in range(qubits_per_qudit):
                    lk_qubit = (l * qubits_per_qudit) + k
                    coeff = Gram[m, l] * (2 ** (j + k - 2))
                    # Jmat can only be used for qubit-qubit interactions
                    if mj_qubit == lk_qubit:
                        identity_coefficient += coeff
                    # same qubit begets constant shift
                    else:
                        Jmat[mj_qubit, lk_qubit] += coeff
                lj_qubit = (l * qubits_per_qudit) + j
                # both linear sums go over same range so can sum qudits l, m over the indices j (i.e. don't need to duplicate with k)
                hvec[mj_qubit] += cn * Gram[m, l] * (2 ** (j - 1))
                hvec[lj_qubit] += cn * Gram[m, l] * (2 ** (j - 1))

            # After multiplying two qudits together, left with a (cn)**2 term, i.e. constant shift
            identity_coefficient += cn * cn * Gram[m, l]

    return Jmat, hvec, identity_coefficient


def svp_isingcoeffs_ham(Gram, sitek):
    """
Turns an n-dimensional SVP problem defined by a Grammian matrix and
a coefficient-range parameter k into an Ising model specification on
n*(ceil(log2(k))+1) spins, where eigenvalues correspond to squared-norms
of vectors in the lattice defined by the SVP problem.

Parameters
----------
Gram  : Grammian matrix as array of shape (n, n)
sitek : a parameter specifying the range of coefficients
   that will multiply each basis vector. The available
   coefficients will be in [-k,-k+1,...0,...,k-2,k-1].

Returns
-------
Jmat                 : array of shape (m, m), where m = n*(ceil(log2(k))+1),
                  containing the coupling (ZZ) coefficients of the Ising
                  Hamiltonian
hvec                 : array of shape (m,) containg the field (Z) coefficients
                  of the Ising Hamiltonian
identity_coefficient : a float representing the scalar that multiplies the
                  identity term that is added to the Ising Hamiltonian
    """
    nqudits = Gram.shape[0]
    qubits_per_qudit = 2 * sitek
    nqubits = nqudits * qubits_per_qudit
    Jmat = np.zeros((nqubits, nqubits))
    hvec = np.zeros(nqubits)
    identity_coefficient = 0

    for m in range(nqudits):
        for l in range(nqudits):
            for j in range(qubits_per_qudit):
                mj_qubit = (m * qubits_per_qudit) + j
                for k in range(qubits_per_qudit):
                    lk_qubit = (l * qubits_per_qudit) + k
                    coeff = (1 / 4) * Gram[m, l]
                    if mj_qubit == lk_qubit:
                        identity_coefficient += coeff
                    else:
                        Jmat[mj_qubit, lk_qubit] += coeff


    return Jmat, hvec, identity_coefficient


def ising_hamiltonian(Jmat, hvec, identity_coefficient, as_diag_vec=False):
    """
Generates the n-spin Ising Hamiltonian matrix specified by the input coupling (ZZ)
coefficients, field (Z) coefficients and an identity coefficient. The basis
is such that if the jth index corresponds to the spin configuration
(s_0,s_1,s_2,...,s_(n-2),(s_(n-1)) then the binary representation of j is
s_(n-1)s_(n-2)...s_2s_1s_0.

Parameters
----------
Jmat                 : array of shape (m, m), where m = n*(ceil(log2(k))+1),
                  containing the coupling (ZZ) coefficients of the Ising
                  Hamiltonian
hvec                 : array of shape (m,) containg the field (Z) coefficients
                  of the Ising Hamiltonian
identity_coefficient : a float representing the scalar that multiplies the
                  identity term that is added to the Ising Hamiltonian
as_diag_vec          : (defaults to False) if True, the output will be an
                  array of shape (2**m,) instead of a diagonal sparse matrix

Returns
-------
H : The Ising Hamiltonian matrix of shape (2**m, 2**m), stored as a diagonal
 sparse matrix.
    """
    nqubits = Jmat.shape[0]
    N = 2 ** nqubits
    Hdiag = np.zeros(N, dtype=np.float64)
    ket = np.array(list(range(N)))
    for i in range(nqubits):
        coeff, bra = Z(ket, i)
        Hdiag[bra] += coeff * hvec[i]
        for j in range(nqubits):
            coeff1, bra = Z(ket, i)
            coeff2, bra = Z(bra, j)
            Hdiag[bra] += coeff1 * coeff2 * Jmat[i, j]
    Hdiag += identity_coefficient
    if as_diag_vec:
        return Hdiag
    else:
        H = spsp.diags(Hdiag, 0)
        return H


"""
bin_mapping refers to the choice of whether to use log(k) or k qubits to represent a qudit
"""


def isingify(Gram, sitek, bin_mapping=True):
    if bin_mapping:
        return ising_hamiltonian(*svp_isingcoeffs_bin(Gram, sitek, bin_mapping))
    else:
        return ising_hamiltonian(*svp_isingcoeffs_ham(Gram, sitek, bin_mapping))
