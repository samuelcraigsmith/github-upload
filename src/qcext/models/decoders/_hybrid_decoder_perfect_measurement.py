"""Implement the HybridDecoder."""

import numpy as np

from qcext.modelext import DecoderFT
from qcext.models.decoders import ColorMatchingDecoder, IsingDecoder
from qecsim import paulitools as pt


class HybridDecoder(DecoderFT):
    """Combines the local Ising decoder with the colour matching decoder.

    Not actually fault-tolerant, requires perfect measurement. The fault
    tolerant meta class was a useful one to use because it naturally
    breaks decoding up into chunks that may involve multiple rounds of
    measurement and error correction, like this decoder.
    """

    def __init__(self):
        """Construct a hybrid decoder."""
        self._global_decoder = ColorMatchingDecoder()
        self._local_decoder = None

    @property
    def global_decoder(self):
        """Return the global decoder."""
        return self._global_decoder

    @property
    def local_decoder(self):
        """Return the local decoder."""
        return self._local_decoder

    def decode_ft(self, code, time_steps, syndrome, *kwargs):
        """See :meth:`qcext.modelext.DecoderFT.decode_ft`."""
        local_corrections = []
        n_stabs = np.shape(code.stabilizers)[0]
        syndrome.append(np.zeros(n_stabs, dtype=int))
        for t in range(time_steps):
            # TODO: local_correction = local_decode(syndrome[t])
            local_correction = code.new_pauli().to_bsf()
            local_corrections.append(local_correction)
            syndrome[t+1] = (syndrome[t+1] ^ syndrome[t] ^ pt.bsp(
                local_correction, code.stabilizers.T)
            )

        global_correction = self.global_decoder.decode(code,
                                                       syndrome[time_steps])

        return global_correction ^ np.bitwise_xor.reduce(local_corrections)

    @property
    def label(self):
        """See :meth:`qcext.modelext.DecoderFT.label`."""
        return "hybrid decoder perfect measurement"
