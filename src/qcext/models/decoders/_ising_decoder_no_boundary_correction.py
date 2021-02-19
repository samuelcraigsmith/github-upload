import functools 
import logging 

import numpy as np 

from qcext.models.decoders import IsingDecoderPiece, IsingDecoder

logger = logging.getLogger(__name__) 

class IsingDecoderPieceNoBoundaryCorrection(IsingDecoderPiece): 
    def pauli_to_local_correction(decode): 
        functools.wraps(decode) 
        def wrapper(self, code, partial_syndrome, qubit, **kwargs): 
            pauli = decode(self, code, partial_syndrome, qubit, **kwargs)   
            region_support = self._get_region_support(code, qubit)
            qubit_index = code.ordered_qubits.index(qubit)  
            correction = np.array([pauli.to_bsf()[i] for i in region_support] 
                + [pauli.to_bsf()[i+code.n_k_d[0]] for i in region_support]) 
            return correction 
        return wrapper 

    @pauli_to_local_correction
    def decode(self, code, partial_syndrome, qubit, **kwargs): 
        # create a correction at the level of Color666Pauli
        if len(partial_syndrome) != len(self._get_stabiliser_subset(code, qubit)): 
            raise ValueError("{} was not given a partial syndrome when decoding".format(type(self).__name__))  
        rng = kwargs["rng"] 
        n_stabilisers_each = int(len(partial_syndrome)/2) 
        correction = code.new_pauli() 

        if sum(partial_syndrome[:n_stabilisers_each]) > n_stabilisers_each/2: 
            correction.site('Z', qubit)  # add Z flip on qubit to correction. 

        if sum(partial_syndrome[n_stabilisers_each:]) > n_stabilisers_each/2: 
            correction.site('X', qubit) 

        return correction 

class IsingDecoderNoBoundaryCorrection(IsingDecoder): 
    @property
    @functools.lru_cache(maxsize=None)
    def partial_decoder_piece(self):
        return IsingDecoderPieceNoBoundaryCorrection() 