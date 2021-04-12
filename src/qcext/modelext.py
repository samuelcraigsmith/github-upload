"""Provide abstract interfaces for models."""

from abc import ABCMeta, abstractmethod
import functools
import copy

import numpy as np
from qecsim.error import QecsimError
from qecsim import paulitools as pt

from qcext.ptext import support


class DecoderFT(metaclass=ABCMeta):
    @abstractmethod
    def decode_ft(self, code, time_steps, syndrome, **kwargs):
        """"""

    @abstractmethod
    def label(self):
        """"""


class PartialDecoderPiece(metaclass=ABCMeta):

    # def verify_partial_syndrome(decode):
    #     @functools.wraps(decode)
    #     def wrapper(*args):
    #         code, partial_syndrome, region = args[1:]
    #         stabiliser_subset = self._get_stabiliser_subset(code, region)
    #         if len(partial_syndrome) > len(stabiliser_subset):
    #             raise ValueError(f"{type(self).__name__} is trying to cheat with too much syndrome data.")
    #         return decode(*args)
    #     return wrapper

    # @verify_partial_syndrome
    @abstractmethod
    def decode(self, code, partial_syndrome, region):
        """
        Return a local correction to the syndrome.

        Region is some object that identifies a subset of the stabilisers by
        the get_partial_syndrome method. For example, on an Ising model, a
        vertex might be used as a region and get_partial_syndrome might return
        the stabilisers living on edges incident to the vertex.

        :param code: stabilizer code on which to perform decoding.
        :type code: Color666CodeNoisy
        :param partial_syndrome: syndrome information around qubit.
        :type partial_syndrome: np.ndarray
        :param qubit: qubit (i, j) around which to decode the syndrome.
        :type qubit: tuple
        :return: a local correction to the syndrome.
        :rtype: np.ndarray
        """

    @abstractmethod
    @functools.lru_cache()
    def _get_stabiliser_subset(self, code, region):
        """Return the indices of the stabilisers that are intended to be specified by region."""

    @functools.lru_cache(maxsize=None)
    def _get_region_support(self, code, region):  #!!!
        """Return the indices of the qubits on which the region has support"""
        stabiliser_subset = self._get_stabiliser_subset(code, region)
        stabilisers = [code.stabilizers[i, :] for i in stabiliser_subset]
        region_support = np.bitwise_or.reduce([support(stabiliser) for stabiliser in stabilisers])
        return np.nonzero(region_support)[0]


class PartialDecoder(DecoderFT):
    def __init__(self, partial_decoder_piece):
        self._partial_decoder_piece = partial_decoder_piece

    @property
    def partial_decoder_piece(self):
        return self._partial_decoder_piece

    @abstractmethod
    @functools.lru_cache()
    def get_regions(self, code):
        pass

    def decode_ft(self, code, time_steps, syndrome, **kwargs):
        """Decode_ft."""
        syndrome = copy.copy(syndrome)
        total_correction = []
        for t in range(time_steps):
            regions = self.get_regions(code)
            partial_corrections_included = []
            for region in regions:
                partial_syndrome = self.get_partial_syndrome(code, syndrome[t], region)
                partial_correction = self.partial_decoder_piece.decode(code, partial_syndrome, region, **kwargs)
                partial_correction_included = self.include_partial_correction(code, partial_correction, region)
                partial_corrections_included.append(partial_correction_included)
            correction = np.bitwise_xor.reduce(partial_corrections_included)
            total_correction.append(correction)
            try:
                syndrome[t+1] = (syndrome[t+1] ^ syndrome[t]
                                 ^ pt.bsp(correction, code.stabilizers.T))
            except IndexError:
                pass
        total_correction = np.bitwise_xor.reduce(total_correction)  # flatten
        return total_correction

    def get_partial_syndrome(self, code, syndrome, region):  #!!!
        """Return the syndrome data for stabilisers contained in region."""
        stabiliser_subset = self.partial_decoder_piece._get_stabiliser_subset(code, region)
        return [syndrome[i] for i in stabiliser_subset]

    def include_partial_correction(self, code, correction, region):  #!!!
        """Include the partial correction into a Pauli operator over the code."""
        region_support = self.partial_decoder_piece._get_region_support(code, region)
        global_correction = np.zeros(2*code.n_k_d[0], dtype=int)
        for q, cx in zip(region_support, correction[:len(region_support)]):
            global_correction[q] = cx
        for q, cz in zip(region_support, correction[len(region_support):]):
            global_correction[q + code.n_k_d[0]] = cz
        return global_correction


class Readout(metaclass=ABCMeta):
    @functools.lru_cache(maxsize=None)
    def validate(self, code):
        # ensure that the measurement basis consists only of single-qubit operators.
        if not np.all(np.apply_along_axis(pt.bsf_wt, axis=1, arr=self.measurement_basis(code)) == 1):
            raise QecsimError("{} has an invalid measurement basis.".format(type(self).__name__))
        # ensure that the conserved stabilisers commutes with the measurement basis.
        if not np.all(pt.bsp(self.conserved_stabilisers(code), self.measurement_basis(code).T) == 0):
            raise QecsimError("{} has conserved stabilisers that do not commute with the measurement basis.".format(type(self).__name__))
        # ensure that the logical representative commutes with the measurement basis.
        if not np.all(pt.bsp(self.conserved_logicals(code), self.measurement_basis(code).T) == 0):
            raise QecsimError("{} has logicals that do not commute with the measurement basis".format(type(self).__name__))
        # ensure that the logical representative commutes with the conserved stabilisers.
        if not np.all(pt.bsp(self.conserved_logicals(code), self.conserved_stabilisers(code).T) == 0):
            raise QecsimError("{} has logicals that do not commute with stabilisers.".format(type(self).__name__))
        return

    @abstractmethod
    def measurement_basis(self, code):
        """A numpy array of single qubit Paulis giving the measurement basis."""

    @abstractmethod
    def generate_error(self, code, measurement_error_rate, rng=None):
        """Generate a set of measurement errors.

        :param code: The quantum code over which readout is performed.
        :type code: Color666Code
        :param measurement_error_probability: measurement error probability in
            [0, 1]
        :type measurement_error_probability: float
        :param rng: random number generator.
        :type rng: numpy.random._generator.Generator or None
        :return: a binary array signalling the presence of measurement
            errors. A non-zero entry error[i]==1 indicates an error on
            measurement measurement_basis[i].
        :rtype: np.ndarray
        """

    @abstractmethod
    def conserved_stabilisers(self, code):
        """A list of generators for the set of stabilisers that commute with the measurement basis."""

    @abstractmethod
    def conserved_logicals(self, code):
        """A representation of the logical operators being readout that commutes with the measurement basis."""

    @abstractmethod # cannot pass measurement outcomes as they are random, only stabilisers contain usable info.
    def decode(self, code, syndrome, **kwargs):
        """Map syndrome data obtained from conserved stabilisers to a logical correction.

        Returns a binary vector over the logical operators to specify a logical correction.
        This method should be thought of as a global decoding step assuming perfect measurements
        over a limited set of stabilizers. The reasoning here is that single-qubit measurement
        errors in the readout stage look like single-qubit Pauli errors prior to measurement,
        and hence measurement itself can be thought of as perfect."""

    @property
    @abstractmethod
    def label(self):
        """
        """


class DecoderFT(metaclass=ABCMeta):
    """
    Defines (fault-tolerant) decoder properties and methods.
    This class cannot be instantiated directly, see :class:`qecsim.models.rotatedtoric.RotatedToricSMWPMDecoder` for an
    example implementation.
    """

    @abstractmethod
    def decode_ft(self, code, time_steps, syndrome, **kwargs):
        """
        Resolve recovery operation for given (fault-tolerant) syndrome.
        Assumptions:
        * The syndrome has shape (number of time steps, number of stabilizers).
        * In the absence of a measurement error, a syndrome element value of 0 or 1 indicates that the corresponding
          stabilizer commutes or does not commute with the error, respectively.
        * The presence of a measurement error inverts the value of the corresponding syndrome element.
        Notes:
        * The keyword parameters ``kwargs`` may be provided by the client with context values such as `error_model`,
          `error_probability`, `error`, `step_errors`, `measurement_error_probability` and `step_measurement_errors`,
          see :func:`qecsim.app.run_once_ftp`. Most implementations will ignore such parameters; however, if they are
          used, implementations should declare them explicitly and treat them as optional.
        * This method typically returns a recovery operation but it may, alternatively, return :class:`DecodeResult`
          to indicate success/failure more explicitly.
        :param code: Stabilizer code.
        :type code: StabilizerCode
        :param time_steps: Number of time steps.
        :type time_steps: int
        :param syndrome: Syndrome as binary array.
        :type syndrome: numpy.array (2d)
        :param kwargs: Optional context parameters passed by a client.
        :type kwargs: dict
        :return: Recovery operation as binary symplectic vector, or decode result indicating recovery success.
        :rtype: numpy.array (1d) or DecodeResult
        """

    @property
    @abstractmethod
    def label(self):
        """
        Label suitable for use in plots.

        :rtype: str
        """
