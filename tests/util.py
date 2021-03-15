"""Provides some testing utilities."""

from itertools import product

import numpy as np


def _single_qubit_errors(code, time_steps, support=None, error_type="history"):
    if support is None:
        support = range(code.n_k_d[0])

    error_histories = []
    qubits = []
    pauli_types = []
    times = []
    for t, qubit in product(range(time_steps), support):
        x_error = np.zeros(2*code.n_k_d[0], dtype=int)
        x_error[qubit] = 1
        z_error = np.zeros(2*code.n_k_d[0], dtype=int)
        z_error[qubit+code.n_k_d[0]] = 1
        y_error = x_error ^ z_error
        for pauli_type, error in zip("XZY", [x_error, z_error, y_error]):
            error_history = [np.zeros(2*code.n_k_d[0], dtype=int)]*time_steps
            error_history[t] = error
            error_histories.append(error_history)
            qubits.append(qubit)
            pauli_types.append(pauli_type)
            times.append(t)

    try:
        qubits = [code.ordered_qubits[qubit] for qubit in qubits]
    except AttributeError:
        pass

    if time_steps == 1 and error_type == "instantaneous":
        errors = [error_history[0] for error_history in error_histories]
        return qubits, pauli_types, errors

    return times, qubits, pauli_types, error_histories


def single_qubit_error_histories(code, time_steps, support=None):
    """
    Return all labelled single-qubit errors over a some support in a code.

    The qubits list will be a list of qubit sites of the code has an
    ordered_qubits attribute. Otherwise, it will be a list of integers
    corresponding to qubit indices in the symplectic space.

    Single qubit errors are also distributed through time. The error history is
    a list indexed by time. A single time-slice contains a single-qubit error
    and all other time-slices contain the identity.

    :param code: code to return single qubit errors over.
    :type code: StabilizerCode
    :param time_steps: number of time steps over which to generate errors.
    :type: int
    :param support: qubits over which to return single qubit errors.
        (default all)
    :type support: list
    :return times: times of the corresponding errors in error_histories
    :rtype: int
    :return qubits: qubit sites corresponding to the errors in error_histories.
    :rtype: list
    :return pauli_type: Pauli types (XZY) of the corresponding errors in
        error_histories.
    :rtype: list
    :return error_histories: single qubit errors over the support region and
        distributed through time.
    :rtype: list
    """
    times, qubits, pauli_types, error_histories = _single_qubit_errors(code, time_steps, support, "history")
    return times, qubits, pauli_types, error_histories


def single_qubit_errors(code, support=None):
    """
    Return all labelled single-qubit errors over a some support in a code.

    The qubits list will be a list of qubit sites of the code has an
    ordered_qubits attribute. Otherwise, it will be a list of integers
    corresponding to qubit indices in the symplectic space.

    :param code: code to return single qubit errors over.
    :type code: StabilizerCode
    :param support: qubits over which to return single qubit errors.
        (default all)
    :type support: list
    :return qubits: qubit sites corresponding to the errors in errors.
    :rtype: list
    :return pauli_type: Pauli types (XZY) of the corresponding errors in
        errors.
    :rtype: list
    :return errors: single qubit errors over the support region.
    :rtype: list
    """
    qubits, pauli_types, errors = _single_qubit_errors(code, 1, support, "instantaneous")
    return qubits, pauli_types, errors