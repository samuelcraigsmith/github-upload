"""This module extends some functionality from the qcext paulitools."""

def is_whole(f, eps=1e-5):
    return abs(f - round(f)) < abs(eps)

def support(pauli_bsf): 
    if len(pauli_bsf)%2 != 0: 
        raise ValueError("{} is not a valid Pauli operator".format(pauli_bsf))  
    n_qubits = int(len(pauli_bsf)/2) 
    pauli_bsf_x = pauli_bsf[:n_qubits] 
    pauli_bsf_z = pauli_bsf[n_qubits:] 
    support = pauli_bsf_x | pauli_bsf_z 
    return support 