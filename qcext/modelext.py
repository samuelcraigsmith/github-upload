from abc import ABCMeta, abstractmethod
import functools 

import numpy as np 
from qecsim.error import QecsimError

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
        """Return a local correction over a region in the code given partial syndrome information. 

        Notes: 
            * region is some object that identifies a subset of the stabilisers by the get_partial_syndrome method. For 
            example, on an Ising model, a vertex might be used as a region and get_partial_syndrome might return the stabilisers
            living on edges incident to the vertex.
        """ 

    @abstractmethod
    @functools.lru_cache() 
    def _get_stabiliser_subset(self, code, region): 
        """Return the indices of the stabilisers that are intended to be specified by region.""" 

    def _get_region_support(self, code, region): 
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
        if time_steps != 1: 
            raise ValueError("{} is a single-shot decoder, must have time_step = 1".format(type(self).__name__)) 
        syndrome = syndrome[0] 
        regions = self.get_regions(code) 
        partial_corrections_included = [] 
        for region in regions: 
            partial_syndrome = self.get_partial_syndrome(code, syndrome, region) 
            partial_correction = self.partial_decoder_piece.decode(code, partial_syndrome, region, **kwargs)  
            partial_correction_included = self.include_partial_correction(code, partial_correction, region) 
            partial_corrections_included.append(partial_correction_included)  
        total_correction = np.bitwise_xor.reduce(partial_corrections_included)  
        return total_correction 

    def get_partial_syndrome(self, code, syndrome, region):
        """Return the syndrome data for stabilisers contained in region.""" 
        stabiliser_subset = self.partial_decoder_piece._get_stabiliser_subset(code, region)  
        return [syndrome[i] for i in stabiliser_subset]

    def include_partial_correction(self, code, correction, region): 
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
        """Generate a set of measurement errors.""" 
    
    @abstractmethod
    def conserved_stabilisers(self, code): 
        """A list of generators for the set of stabilisers that commute with the measurement basis.""" 

    @abstractmethod
    def conserved_logicals(self, code):
        """A representation of the logical operators being readout that commutes with the measurement basis.""" 

    @abstractmethod # cannot pass measurement outcomes as they are random, only stabilisers containe usable info. 
    def decode(self, code, syndrome, **kwargs):
        """Map syndrome data obtained from conserved stabilisers to a logical correction. 

        Returns a binary vector over the logical operators to specify a logical correction. """ 

    @property
    @abstractmethod
    def label(self):
        """
        """