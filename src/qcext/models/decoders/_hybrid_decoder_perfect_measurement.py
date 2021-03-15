"""Implement the HybridDecoderPerfectMeasurement."""
import logging

import numpy as np

from qcext.modelext import DecoderFT
from qcext.models.decoders import ColorMatchingDecoder, IsingDecoder
from qecsim import paulitools as pt


logger = logging.getLogger(__name__)


class HybridDecoderPerfectMeasurement(DecoderFT):
    """Combines the local Ising decoder with the colour matching decoder.

    Not actually fault-tolerant, requires perfect measurement. The fault
    tolerant meta class was a useful one to use because it naturally
    breaks decoding up into chunks that may involve multiple rounds of
    measurement and error correction, like this decoder.
    """

    def __init__(self, *local_correction):
        """Construct a hybrid decoder."""
        self._global_decoder = ColorMatchingDecoder()
        self._label = "hybrid decoder perfect measurement"
        if local_correction[0] == "no_local":
            self._local_decoder = None
            self._label += " (no local)"
            logger.info("Hybrid decoder initiated without local correction.")
        else:
            self._local_decoder = IsingDecoder()
            self._label += " (local)"

    @property
    def label(self):
        return self._label

    @property
    def global_decoder(self):
        """Return the global decoder."""
        return self._global_decoder

    @property
    def local_decoder(self):
        """Return the local decoder."""
        return self._local_decoder

    def decode_ft(self, code, time_steps, syndrome, **kwargs):
        """See :meth:`qcext.modelext.DecoderFT.decode_ft`."""
        if self.local_decoder is not None:
            local_correction = self.local_decoder.decode_ft(code, time_steps,
                                                            syndrome)
        else:
            local_correction = code.new_pauli().to_bsf()
        leftover_syndrome = (np.bitwise_xor.reduce(syndrome)
                             ^ pt.bsp(local_correction, code.stabilizers.T))
        global_correction = self.global_decoder.decode(code, leftover_syndrome)

        logger.debug("Local correction \n" + code.ascii_art(pauli=code.new_pauli(bsf=local_correction)))
        logger.debug("Leftover syndrome \n" + code.ascii_art(syndrome=leftover_syndrome))
        logger.debug("Global correction \n" + code.ascii_art(pauli=code.new_pauli(bsf=global_correction)))
        return global_correction ^ local_correction
