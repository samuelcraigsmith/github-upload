import functools
import logging

import numpy as np

from qcext.modelext import PartialDecoder, PartialDecoderPiece

logger = logging.getLogger(__name__)


class IsingDecoderPiece(PartialDecoderPiece):
    """
    Implement a piece of an Ising decoder.

    See :class:`IsingDecoder`.
    """

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
        """See :meth:`qcext.modelext.PartialDecoderPiece.decode`."""
        if len(partial_syndrome) != len(self._get_stabiliser_subset(code, qubit)):
            raise ValueError("{} was not given a partial syndrome when decoding".format(type(self).__name__))

        try:
            rng = kwargs["rng"]
        except KeyError:
            rng = np.random.default_rng()

        n_stabilisers_each = int(len(partial_syndrome)/2)
        correction = code.new_pauli()
        if sum(partial_syndrome[:n_stabilisers_each]) > n_stabilisers_each/2:
            correction.site('Z', qubit)  # add Z flip on qubit to correction.
        elif sum(partial_syndrome[:n_stabilisers_each]) == n_stabilisers_each/2:
            if rng.random() < 0.5:  # flip a coin to decide whether to add Z flip to corection.
                correction.site('Z', qubit)
        if sum(partial_syndrome[n_stabilisers_each:]) > n_stabilisers_each/2:
            correction.site('X', qubit)
        elif sum(partial_syndrome[n_stabilisers_each:]) == n_stabilisers_each/2:
            if rng.random() < 0.5:
                correction.site('X', qubit)
        return correction

    def _get_stabiliser_subset(self, code, qubit):
        plaquettes = code.ising_star(qubit)
        plaquette_indices = [i for i, p in enumerate(code.ordered_plaquettes) if p in plaquettes]
        num_stabs = np.shape(code.stabilizers)[0]
        x_stabiliser_indices = plaquette_indices
        z_stabiliser_indices = [int(i + num_stabs/2) for i in x_stabiliser_indices]
        return x_stabiliser_indices + z_stabiliser_indices


class IsingDecoder(PartialDecoder):
    """
    Implement an Ising decoder.

    An Ising decoder piece sits at every noisy qubit in a Color666Code. The
    partial syndrome data is the syndrome data restricted to all plaquettes
    that contain the noisy qubit. The decoder will return a Pauli X (Z)
    operator on the qubit if the majority vote of the Pauli Z (X) type
    stabilizers is -1. In the case of noisy qubits living near the boundary, a
    fair coin is flipped to break ties.
    """

    def __init__(self):
        pass

    @property
    @functools.lru_cache(maxsize=None)
    def partial_decoder_piece(self):
        return IsingDecoderPiece()

    @functools.lru_cache(maxsize=None)
    def get_regions(self, code):
        return [code.ordered_qubits[i] for i in code.noisy_qubits]

    @property
    def label(self):
        return "Ising decoder for 6,6,6 colour code."
