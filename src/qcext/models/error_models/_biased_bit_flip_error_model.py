"""Implements a restricted bit flip error model."""
import functools
import logging
from itertools import islice
import math

import numpy as np
from qecsim import paulitools as pt
from qecsim.model import ErrorModel, cli_description


logger = logging.getLogger(__name__)


@cli_description("Bit-flip noise over noisy qubits.")
class BiasedBitFlipErrorModel(ErrorModel):
    """Bit flip error model with a bias towards noisy qubits.

    The noise model introduces bit flips with a bias towards acting on noisy
    qubits. Specifically, if p_high and p_low are the probabilities that a
    bit-flip will be placed on a high-rate qubit and a low-rate qubit
    respectively, then the noise model can be characterised by two parameters:

    p = (n_high.p_high + n_low.p_low)/n
    bias = (p_high/p_low - 1)

    where n_high and n_low are the number of high-rate qubits and low-rate
    qubits respectively. Note that bias=0 corresponds to the homogenous noise
    case, and bias=inf corresponds to the case where low-rate qubits do not
    experience noise. The expected error weight is given by np independently of
    the bias.
    """

    def __init__(self, bias, *args):
        """Initialise a restricted bit flip error model."""
        if args:
            noisy_qubits = args[0]
            try:
                list.sort(noisy_qubits)
                if not noisy_qubits[0] >= 0:
                    raise ValueError('{} valid noisy qubits cannot have negative indices'.format(type(self).__name__))
            except IndexError:
                pass  # empty qubit list is allowed.
            except TypeError as ex:
                raise TypeError('{} invalid parameter type'.format(type(self).__name__)) from ex
            self._noisy_qubits = noisy_qubits
        else:
            self._noisy_qubits = None

        self._bias = bias

    @property
    def bias(self):
        """Return the noise bias."""
        return self._bias

    @property
    def noisy_qubits(self):
        """Qubits over which noise is applied."""
        return self._noisy_qubits

    @property
    def label(self):
        """See ..."""
        return 'Biased bit flip bias={}'.format(self.bias)

    def generate(self, code, probability, rng=None):
        """Generate an error on the code."""
        if self.noisy_qubits is not None:
            noisy_qubits = self.noisy_qubits
        else:
            noisy_qubits = code.noisy_qubits

        n = code.n_k_d[0]
        if noisy_qubits[-1] >= n:
            raise ValueError('{} was instantiated to act on codes with at least {} qubits.'.format(
                type(self).__name__, noisy_qubits[-1]+1))
        rng = np.random.default_rng() if rng is None else rng
        n_high = len(noisy_qubits)

        if self.bias < math.inf:
            p_low = n*probability / (self.bias*n_high + n)
            p_high = min((self.bias + 1)*p_low, 1)
        else:
            p_low = 0
            p_high = min(n*probability / n_high, 1)

        error_pauli_high_rate_qubits = ''.join(rng.choice(
            ('I', 'X', 'Y', 'Z'),
            size=n_high,
            p=self.probability_distribution(p_high)
        ))

        error_pauli_low_rate_qubits = ''.join(rng.choice(
            ('I', 'X', 'Y', 'Z'),
            size=n-n_high,
            p=self.probability_distribution(p_low)
        ))

        error_pauli_low_rate_qubits = iter(error_pauli_low_rate_qubits)
        error_pauli = "".join(
            ["".join(islice(error_pauli_low_rate_qubits, i-j-1)) + e for i, j, e in zip(
                noisy_qubits,
                [-1]+noisy_qubits,
                list(error_pauli_high_rate_qubits),
            )]
        )
        error_pauli += "".join(error_pauli_low_rate_qubits)

        return pt.pauli_to_bsf(error_pauli)

    @functools.lru_cache()
    def probability_distribution(self, probability):
        """See :meth:`qecsim.model.ErrorModel.probability_distribution`."""
        p_x = probability
        p_y = p_z = 0
        p_i = 1 - sum((p_x, p_y, p_z))
        return p_i, p_x, p_y, p_z
