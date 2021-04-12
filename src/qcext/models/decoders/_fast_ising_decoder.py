"""Fast ising decoder."""
import functools
import copy
import numpy as np

from qecsim import paulitools as pt
from qcext.modelext import DecoderFT


class FastIsingDecoder(DecoderFT):
    """Fast Ising decoder."""

    @property
    def label(self):
        return "fast Ising decoder"



    def decode_ft(self, code, time_steps, syndrome, **kwargs):
        syndrome = copy.copy(syndrome)
        try:
            rng = kwargs["rng"]
        except KeyError:
            rng = np.random.default_rng()

        neighbourhoods = self._compute_neighbourhoods(code, time_steps,
                                                 syndrome, **kwargs)
        sizes = np.sum(neighbourhoods, axis=1)
        noisy_qubits_support = np.zeros(2 * code.n_k_d[0], dtype=int)
        for i in code.noisy_qubits:
            noisy_qubits_support[i] = 1
            noisy_qubits_support[i + code.n_k_d[0]] = 1

        total_correction = []
        for t in range(time_steps):
            coin_flips = rng.choice([-1, 1], 2 * code.n_k_d[0])
            correction = (syndrome[t] @ neighbourhoods.T
                          > (sizes / 2 + 0.1 * coin_flips))
            correction = np.bitwise_and(correction, noisy_qubits_support)
            try:
                syndrome[t+1] = (syndrome[t+1] ^ syndrome[t]
                                 ^ pt.bsp(correction, code.stabilizers.T))
            except IndexError:
                pass
            total_correction.append(correction)
        total_correction = np.bitwise_xor.reduce(total_correction)  # flatten
        return total_correction



    def _compute_neighbourhoods(self, code, time_steps, syndrome, **kwargs):
        """Return a matrix whose rows give the neighbourhood of a qubit.

        The ith row of the upper half matrix is a binary vectors of length n with non-zero entries at
        indices that correspond to the indices of X-type stabilisers contained
        in the ising star of the ith qubit. The lower half has rows corresponding to indices of Z-type stabilisers.

        Notes
        -----

        * Should I include non-noisy qubits so the indices are right? Ultimately what I want is a correction (pauli X operator). I need
        either site (i, j) or indices to construct that operator. I do have
        a fast map from noisy qubit indices to qubit indices, so I don't need
        to include non-noisy qubits. Although it will involve constructing an array at every time step. (enlarge corrections_x to include it into the code). Yes, include non-noisy qubits as zero vectors.
        """

        s = np.shape(code.stabilizers)[0]
        n = code.n_k_d[0]

        neighbourhoods = np.zeros((2 * n, s),
                                  dtype=int)
        plaquette_site_to_index = {site: i for i, site in
                                   enumerate(code.ordered_plaquettes)}

        for q_index, q_site in enumerate(code.ordered_qubits):
            if q_index in code.noisy_qubits:
                p_sites = code.ising_star(q_site)
                p_indices = [plaquette_site_to_index[p_site] for p_site in p_sites]
                for p_index in p_indices:
                    neighbourhoods[q_index + n, p_index] = 1
                    neighbourhoods[q_index,
                                   int(p_index + s / 2)] = 1
        return neighbourhoods
