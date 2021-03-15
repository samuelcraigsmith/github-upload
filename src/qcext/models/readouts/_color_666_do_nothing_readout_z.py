"""Readout logical Z assuming a code state."""

import numpy as np
import logging

from qcext.models.readouts import Color666ReadoutZ


logger = logging.getLogger(__name__)


class Color666DoNothingReadoutZ(Color666ReadoutZ):
    """Readout that has no measurement errors and returns the identity.

    Corresponds to reading out the logical Z operator from a code-state
    assuming perfect measurement.
    """

    def generate_error(self, code, measurement_error_probability,
                       rng=np.random.default_rng()):
        """Return no error."""
        if measurement_error_probability > 0:
            logger.warning(
                "Using do-nothing readout with non-zero measurement"
                + " error probability. Do-nothing readout does not generate"
                + " measurement errors."
            )

        return np.zeros(code.n_k_d[0], dtype=int)

    def decode(self, code, syndrome, **kwargs):
        """Return the identity."""
        if np.any(syndrome):
            logger.warning("Using do-nothing readout outside the code space.")
        return code.new_pauli().to_bsf()

    @property
    def label(self):
        """See ..."""
        return "Color666 do-nothing readout Z"
