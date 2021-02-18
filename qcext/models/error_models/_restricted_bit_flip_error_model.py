import functools
import logging 

import numpy as np 
from qecsim import paulitools as pt
from qecsim.model import ErrorModel, cli_description

logger = logging.getLogger(__name__) 

@cli_description("Bit-flip noise over noisy qubits.") 
class RestrictedBitFlipErrorModel(ErrorModel): 
    """Depolarizing error model restricted to noisy qubits."""  
    def __init__(self, noisy_qubits): 
        try: 
            list.sort(noisy_qubits) 
            if not noisy_qubits[0] >= 0: 
                raise ValueError('{} valid noisy qubits cannot have negative indices'.format(type(self).__name__)) 
        except IndexError: 
            pass # empty qubit list is allowed.
        except TypeError as ex: 
            raise TypeError('{} invalid parameter type'.format(type(self).__name__)) from ex 
        self._noisy_qubits = noisy_qubits 

    @property
    def noisy_qubits(self):
        return self._noisy_qubits

    @property 
    def label(self): 
        return 'Restricted depolarizing error model' 

    def generate(self, code, probability, rng=None): 
        n_qubits = code.n_k_d[0]
        if self.noisy_qubits[-1] >= n_qubits: 
            raise ValueError('{} was instantiated to act on codes with at least {} qubits.'.format(
                type(self).__name__, self.noisy_qubits[-1]+1))  
        rng = np.random.default_rng() if rng is None else rng
        n_qubits_noisy = len(self.noisy_qubits)
        error_pauli_restricted = ''.join(rng.choice(
            ('I', 'X', 'Y', 'Z'),
            size=n_qubits_noisy,
            p=self.probability_distribution(probability)
        ))
        error_pauli = ''.join(['I'*(i-j-1) + e for i,j,e in zip(
            self.noisy_qubits+[n_qubits], 
            [-1]+self.noisy_qubits, 
            list(error_pauli_restricted)+[''])
        ]) 
        return pt.pauli_to_bsf(error_pauli)

    @functools.lru_cache()
    def probability_distribution(self, probability):
        """See :meth:`qecsim.model.ErrorModel.probability_distribution`"""
        p_x = probability
        p_y = p_z = 0
        p_i = 1 - sum((p_x, p_y, p_z))
        return p_i, p_x, p_y, p_z