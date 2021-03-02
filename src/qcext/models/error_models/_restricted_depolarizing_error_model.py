import functools
import logging

import numpy as np
from qecsim import paulitools as pt
from qecsim.model import ErrorModel, cli_description

logger = logging.getLogger(__name__)

@cli_description('Depolarizing noise over noisy qubits')
class RestrictedDepolarizingErrorModel(ErrorModel):
    """Depolarizing error model restricted to noisy qubits."""
    def __init__(self, *args):
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

    @property
    def noisy_qubits(self):
        return self._noisy_qubits

    @property
    def label(self):
        return 'Restricted depolarizing error model'

    def generate(self, code, probability, rng=None):
        if self.noisy_qubits is not None:
            noisy_qubits = self.noisy_qubits
        else:
            noisy_qubits = code.noisy_qubits

        n_qubits = code.n_k_d[0]
        if noisy_qubits[-1] >= n_qubits:
            raise ValueError('{} was instantiated to act on codes with at least {} qubits.'.format(
                type(self).__name__, noisy_qubits[-1]+1))
        rng = np.random.default_rng() if rng is None else rng
        n_qubits_noisy = len(noisy_qubits)
        error_pauli_restricted = ''.join(rng.choice(
            ('I', 'X', 'Y', 'Z'),
            size=n_qubits_noisy,
            p=self.probability_distribution(probability)
        ))
        error_pauli = ''.join(['I'*(i-j-1) + e for i,j,e in zip(
            noisy_qubits+[n_qubits], 
            [-1]+noisy_qubits, 
            list(error_pauli_restricted)+[''])
        ])
        return pt.pauli_to_bsf(error_pauli)

    @functools.lru_cache()
    def probability_distribution(self, probability):
        p_x = p_y = p_z = probability / 3
        p_i = 1 - sum((p_x, p_y, p_z))
        return p_i, p_x, p_y, p_z