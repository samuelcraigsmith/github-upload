"""Test the Ising decoder class."""

import unittest
from parameterized import parameterized
import numpy as np
from itertools import product

import tests.util
from qecsim import paulitools as pt
from qcext.models.codes import Color666CodeNoisy
from qcext.models.decoders import IsingDecoder


class TestIsingDecoderSingleQubitErrors(unittest.TestCase):
    """Test the Ising decoder with single-qubit errors."""

    d = 13
    code = Color666CodeNoisy(d)
    decoder = IsingDecoder()
    time_steps = 1

    args_list = []
    times, qubits, pauli_types, error_histories = tests.util.single_qubit_error_histories(
        code, time_steps, code.noisy_qubits
    )
    bulk_qubits = [(8, 4), (12, 4), (10, 6), (12, 6), (14, 4), (14, 8),
                   (14, 10)]
    #  list comprehension inapplicable due to scope issues?
    for t, qubit, pauli_type, error_history in zip(times, qubits, pauli_types,
                                                   error_histories):
        if qubit in bulk_qubits:
            args_list.append((t, qubit, pauli_type, error_history))

    @parameterized.expand(args_list)
    def test_bulk_single_qubit_errors_single_time_step(self, t, qubit, pauli_type,
                                                       error_history):
        """Test single qubit errors in the bulk are corrected locally."""
        syndrome = []
        if self.time_steps > 1:
            raise Exception("Number of time steps should be time_steps=1 for this test.")
        for t in range(self.time_steps):
            syndrome.append(pt.bsp(error_history[t], self.code.stabilizers.T))

        total_error = np.bitwise_xor.reduce(error_history)
        correction = self.decoder.decode_ft(self.code, 1, syndrome)
        self.assertFalse(
            any(pt.bsp(total_error ^ correction, self.code.stabilizers.T)),
            ("Decoder does not correct a Pauli {} error on site {} to the"
             " codespace.").format(pauli_type, qubit)
        )
        self.assertFalse(
            any(pt.bsp(total_error ^ correction, self.code.logicals.T)),
            ("Decoder does not correct a Pauli {} error on site {} up to"
             " stabilizers.").format(pauli_type, qubit)
        )


class TestIsingDecoder(unittest.TestCase):
    """Test the Ising decoder with extended timelike errors."""

    def test_bulk_extended_error_multiple_time_steps(self):
        """Tests an temporally extended error gets corrected over time."""
        d = 13
        code = Color666CodeNoisy(d)
        decoder = IsingDecoder()
        bulk_qubits = [(8, 4), (12, 4), (10, 6), (12, 6), (14, 4), (14, 8),
                       (14, 10)]
        time_steps = len(bulk_qubits)

        error_history = [
            code.new_pauli().site("X", bulk_qubit).to_bsf()
            for t, bulk_qubit in product(range(time_steps), bulk_qubits)
        ]

        syndrome = []
        for t in range(time_steps):
            syndrome.append(pt.bsp(error_history[t], code.stabilizers.T))

        total_error = np.bitwise_xor.reduce(error_history)
        correction = decoder.decode_ft(code, time_steps, syndrome)
        self.assertFalse(
            any(pt.bsp(total_error ^ correction, code.stabilizers.T))
        )
        self.assertFalse(
            any(pt.bsp(total_error ^ correction, code.logicals.T))
        )


if __name__ == "__main__":
    unittest.main()
