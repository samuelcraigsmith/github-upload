"""Test the hybrid decoder."""

import unittest
from parameterized import parameterized
from itertools import product

import numpy as np

from qecsim import paulitools as pt
from qcext.models.codes import Color666CodeNoisy
from qcext.models.decoders import HybridDecoder


class TestHybridDecoder(unittest.TestCase):
    r"""Test the hybrid decoder.

    Will mainly use a distance five color code. For reference:
    (0,0)
      |
      |
    (1,0)   [1,1]
         \
           \
    [2,0]   (2,1)---(2,2)
              |          \
              |            \
    (3,0)---(3,1)   [3,2]   (3,3)
      |          \            |
      |            \          |
    (4,0)   [4,1]   (4,2)---(4,3)   [4,4]
         \            |          \
           \          |            \
    [5,0]   (5,1)---(5,2)   [5,3]   (5,4)---(5,5)
              |          \            |          \
              |            \          |            \
    (6,0)---(6,1)   [6,2]   (6,3)---(6,4)   [6,5]   (6,6)
    """

    #  TEST SINGLE QUBIT ERRORS
    d = 5
    time_steps = 3
    n_cyc = 1
    code = Color666CodeNoisy(d)
    decoder = HybridDecoder()
    time_steps = 3
    arguments = []
    for t in range(time_steps):
        for site, pauli_type in product(product(range(2*d), repeat=2), "XYZ"):
            if code.is_in_bounds(site) and code.is_site(site):
                error = [code.new_pauli().to_bsf()]*3
                error[t] = code.new_pauli().site(pauli_type, site).to_bsf()
                arguments.append((t, site, pauli_type, error))

    @parameterized.expand(
        arguments
    )
    def test_all_single_qubit_errors(self, t, site, pauli_type, error):
        """Test decoder succeeds for all single-qubit errors.

        All single-qubit errors contained in a space-time region defined by a
        code distance d=5 and a time-like dimension t=3.
        """
        syndrome = [pt.bsp(error[t], self.code.stabilizers.T)
                    for t in range(self.time_steps)]
        recovery = self.decoder.decode_ft(self.code, self.time_steps, syndrome)
        total_error = np.bitwise_xor.reduce(error)
        # RETURNS TO CODESPACE
        self.assertTrue(
            np.all(pt.bsp(recovery ^ total_error,
                   self.code.stabilizers.T) == 0),
            ("Decoder does not correct a Pauli {} error on site {} at time"
             " step {} to the codespace.").format(pauli_type, site, t)
        )
        # NO LOGICAL ERROR
        self.assertTrue(
            np.all(pt.bsp(recovery ^ total_error,
                   self.code.logicals.T) == 0),
            ("Decoder does not correct a Pauli {} error on site {} at time"
             " step {} to stabilizers.").format(pauli_type, site, t)
        )

    #  TEST MULTIPLE-QUBIT ERRORS THROUGH TIME



if __name__ == "__main__":
    unittest.main()
