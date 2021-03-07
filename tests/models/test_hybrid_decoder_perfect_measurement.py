"""Test the hybrid decoder."""

import unittest
from parameterized import parameterized
from itertools import product
import logging

import numpy as np

from qecsim import paulitools as pt
from qcext.models.codes import Color666CodeNoisy
from qcext.models.decoders import HybridDecoder
from tests.util import single_qubit_error_histories


logger = logging.getLogger(__name__)
loggerHybridDecoder = logging.getLogger(
    "qcext.models.decoders._hybrid_decoder_perfect_measurement"
)
logging.basicConfig(level=logging.DEBUG)


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
    d = 9
    time_steps = 3
    n_cyc = 1
    code = Color666CodeNoisy(d)
    decoder = HybridDecoder()
    time_steps = 3
    times, sites, pauli_types, error_histories = single_qubit_error_histories(
        code, time_steps, support=code.noisy_qubits)
    arguments = zip(times, sites, pauli_types, error_histories)

    @parameterized.expand(
        arguments
    )
    def test_all_single_qubit_errors_noisy_qubits(self, t, site, pauli_type,
                                                  error_history):
        """Test decoder succeeds for all single-qubit errors.

        All single-qubit errors contained in a space-time region defined by a
        code distance d=5 and a time-like dimension t=3.
        """
        logger.setLevel(logging.WARNING)
        loggerHybridDecoder.setLevel(logging.WARNING)

        logger.debug("Error at {} \n".format(t) + self.code.ascii_art(pauli=self.code.new_pauli(bsf=error_history[t])))

        syndrome = [pt.bsp(error_history[t], self.code.stabilizers.T)
                    for t in range(self.time_steps)]

        logger.debug("Flattened syndrome: \n" + self.code.ascii_art(syndrome=np.bitwise_xor.reduce(syndrome)))

        recovery = self.decoder.decode_ft(self.code, self.time_steps, syndrome)
        total_error = np.bitwise_xor.reduce(error_history)
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
