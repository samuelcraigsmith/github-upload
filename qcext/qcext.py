import functools
from itertools import product, compress
import collections
import itertools
import json
import logging
import statistics
import time
from abc import ABCMeta, abstractmethod
from collections import deque 

import numpy as np

from qecsim import paulitools as pt
from qecsim.model import StabilizerCode, ErrorModel, Decoder, DecoderFTP, cli_description
from qecsim.models.color import Color666Code, Color666Pauli 
from qecsim.models.generic import DepolarizingErrorModel 
from qecsim.error import QecsimError 
from qecsim.app import _add_rate_statistics

import qcext 

logger = logging.getLogger(__name__)


######################################################### USEFUL FUNCTIONS ##################################################

def is_whole(f, eps=1e-5):
    return abs(f - round(f)) < abs(eps)

def support(pauli_bsf): 
    if len(pauli_bsf)%2 != 0: 
        raise ValueError("{} is not a valid Pauli operator".format(pauli_bsf))  
    n_qubits = int(len(pauli_bsf)/2) 
    pauli_bsf_x = pauli_bsf[:n_qubits] 
    pauli_bsf_z = pauli_bsf[n_qubits:] 
    support = pauli_bsf_x | pauli_bsf_z 
    return support 

########################################################## CODE #############################################################

@cli_description('Colour code with some noisier qubits') 
class Color666CodeNoisy(Color666Code): 
    """Implements a 6,6,6 colour code and keeps a list of noisy qubits.

    Intended for use with a restricted error model.

    Notes: 
    * Also included some lattice methods and properties (hyperboundary, ordered_qubits).

    TODO: all public methods should take and return arguments in the form of index sets. 
    Otherwise implementation details are needed to use the class."""  
    def __init__(self, size): 
        super().__init__(size) 
        if size%4 != 1: 
            raise ValueError('{} size must be of form 4*n+1 for integer n.'.format(type(self).__name__))

    @property 
    @functools.lru_cache() 
    def noisy_qubits(self): # working
        """Return list of indices of noisy qubits (ordering determined by basis of symplectic space).""" 
        protected_plaquettes = [(i, j) for i, j in product(range(self.bound+1), repeat=2) 
            if i%2==0 and j%2==0 and (j-1/2*i)%3==2 and self.is_in_bounds((i, j))] 
        quiet_qubits = [q for p in protected_plaquettes for q in self.hyperboundary(p)] 
        noisy_qubits_indices = [i for i, q in enumerate(self.ordered_qubits) if q not in quiet_qubits] 
        return noisy_qubits_indices

    @property 
    @functools.lru_cache() 
    def boundary_qubits(self): 
        size = self.n_k_d[2] 
        n_sites = int(1/2 * (3*size-1)) # number of sites along any boundary 
        boundary1 = [ (i, 0) for i in range(n_sites)] 
        boundary2 = [(n_sites-1, j) for j in range(n_sites)] 
        boundary3 = [(i, i) for i in range(n_sites)] 
        boundary_qubit_indices = [i for i, j in enumerate(self.ordered_qubits) if j in boundary1+boundary2+boundary3] 
        return boundary_qubit_indices 

    @property
    @functools.lru_cache() 
    def ordered_qubits(self): # working
        """Return a list over tuples qubits in the order specified by the basis of the symplectic space.""" 
        indices = [] 
        qubits = [] 
        for i,j in product(range(self.bound+1), repeat=2): 
            if self.is_site((i,j)) and self.is_in_bounds((i,j)): 
                x_ij = self.new_pauli().site('X', (i,j)) 
                x_ij_bsf = x_ij.to_bsf() 
                index_ij = np.where(x_ij_bsf==1)[0][0]
                qubits.append((i, j)) 
                indices.append(index_ij) 
        enumeration = sorted(list(zip(indices, qubits)), key=lambda pair: pair[0]) 
        ordered_qubits = [qubit for index, qubit in enumeration] 
        return ordered_qubits 

    @property
    @functools.lru_cache() 
    def ordered_plaquettes(self): # working 
        """Return a list over plaquettes in the order specified by the stabilisers.""" 
        indices = [] 
        plaquettes = [] 
        for i,j in product(range(self.bound+1), repeat=2): 
            if self.is_plaquette((i, j)) and self.is_in_bounds((i, j)): 
                stabilizer_ij = self.new_pauli().plaquette('X', (i, j)) 
                stabilizer_ij_bsf = stabilizer_ij.to_bsf() 
                index_ij = np.where(np.apply_along_axis(lambda s: np.all(s==stabilizer_ij_bsf), axis=1, arr=self.stabilizers))[0][0] 
                indices.append(index_ij) 
                plaquettes.append((i, j)) 
        enumeration = sorted(list(zip(indices, plaquettes)), key=lambda pair: pair[0]) 
        ordered_plaquettes = [plaquette for index, plaquette in enumeration] 
        return ordered_plaquettes 


    @functools.lru_cache(maxsize=None)
    def hyperboundary(self, plaquette):
        """Return all qubits contained in a plaquette."""
        if not self.is_plaquette(plaquette):
            raise IndexError('{} is not a plaquette index.'.format(plaquette))
        i, j = plaquette 
        hyperboundary = [site for site in [(i+1, j), (i+1, j+1), (i, j+1), (i-1, j), (i-1, j-1), (i, j-1)]
            if self.is_in_bounds(site)]  
        return hyperboundary 

    @functools.lru_cache(maxsize=None) 
    def ising_neighbourhood(self, site): 
        """Return an Ising neighbourhood of site.

        An Ising neighbourhood is defined with reference to links on the Ising model that arises in the restricted 
        noise setting."""
        i, j = site
        noisy_qubit_sites = [self.ordered_qubits[i] for i in self.noisy_qubits] 
        if not self.is_site(site): 
            raise IndexError("{} is not a site.".format(site)) 
        if not self.is_in_bounds(site): 
            raise IndexError("{} is not within bounds.".format(site)) 
        if not site in noisy_qubit_sites: 
            raise IndexError("{} is not the site of a noisy qubit.".format(site)) 
        possible_neighbours = [(i+2, j), (i+2, j+2), (i, j+2), (i-2, j), (i-2, j-2), (i, j-2)] 
        neighbourhood = [neighbour for neighbour in possible_neighbours if neighbour in noisy_qubit_sites] 
        return neighbourhood 

    @functools.lru_cache(maxsize=None)
    def ising_link(self, site1, site2): 
        """Return an Ising link (plaquette site) connecting site1 and site2.""" 
        if site2 not in self.ising_neighbourhood(site1): 
            raise IndexError("Sites {} and {} are not ising neighbours.".format(site1, site2)) 
        i1, j1 = site1 
        i2, j2 = site2 
        plaquette = ( int((i1+i2)/2), int((j1+j2)/2) )
        assert(self.is_plaquette(plaquette)) 
        return plaquette 

    @functools.lru_cache(maxsize=None)
    def ising_star(self, site): 
        return [self.ising_link(site, neighbour) for neighbour in self.ising_neighbourhood(site)] 

########################################################### NOISE #############################################################

@cli_description('Depolarizing noise over noisy qubits') 
class RestrictedDepolarizingErrorModel(ErrorModel): 
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
        p_x = p_y = p_z = probability / 3
        p_i = 1 - sum((p_x, p_y, p_z))
        return p_i, p_x, p_y, p_z

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

################################################### DECODER #################################################################

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


class IsingDecoderPiece(PartialDecoderPiece): 
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
        elif sum(partial_syndrome[:n_stabilisers_each]) == n_stabilisers_each/2: 
            if rng.random() < 0.5: # flip a coin to decide whether to add Z flip to correction.
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

class IsingDecoder(PartialDecoder): # appears to work. 
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


################################################## READOUT ##################################################################

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


class Color666ReadoutZ(Readout): 
    """Read out the Z-type logical operators on the 6,6,6 colour code."""
    @functools.lru_cache()
    def measurement_basis(self, code): 
        return np.concatenate([np.zeros((code.n_k_d[0], code.n_k_d[0]), dtype=int), np.eye(code.n_k_d[0], dtype=int)], axis=1)    

    @functools.lru_cache()
    def conserved_stabilisers(self, code):
        stabilisers = code.stabilizers
        stabilisers_z = stabilisers[ np.all(stabilisers[:, :code.n_k_d[0]]==0, axis=1) ]
        return stabilisers_z 

    @functools.lru_cache()
    def conserved_logicals(self, code):  
        logicals = code.logicals 
        logical_zs = logicals[ np.all(logicals[:, :code.n_k_d[0]]==0, axis=1) ]
        return logical_zs

    def decode(self, code, syndrome, **kwargs): 
        # implement some 2D colour-code decoder here... 
        pass

    @property
    def label(self):
        return "Color code logical Z readout." 
    

class Color666NoisyReadoutZ(Color666ReadoutZ): 
    """Read out the Z-type logical operators on the 6,6,6 colour code assuming quiet qubits are silent.""" 

    def generate_error(self, code, measurement_error_probability, rng=np.random.default_rng()): 
        qubit_readout_error = rng.choice(
            (0, 1),
            size=len(code.noisy_qubits),
            p=(1 - measurement_error_probability, measurement_error_probability)
        )
        included_qubit_readout_error = np.zeros(code.n_k_d[0], dtype=int) 
        for i, e in zip(code.noisy_qubits, qubit_readout_error): 
            included_qubit_readout_error[i] = e 
        return included_qubit_readout_error    

    def decode(self, code, syndrome, **kwargs): # working 
        """Return a correction based on a majority vote scheme. 

        Notes: 
            * An error acting over a region on the lattice gives rise to syndrome defects over the boundary of that region. An 
            error acting over the complement of that region gives rise to the same syndrome defects. Hence, the decoder must choose
            between two possible errors. 
            * This decoder runs a BFS over the lattice to partition the qubits into the two regions defining the two possible errors. It then returns 
            the correction supported on the smaller region. 
        """
        black = [] 
        red = [] 
        get_list = {"black":black, "red":red} 
        assignments = {} 
        assigned_unvisited = deque()
        # assign any site
        assignments[(0, 0)] = "black" 
        black.append((0, 0))    
        assigned_unvisited.append((0, 0))  
        
        # assignment rule
        get_assignment = lambda colour, syndrome_bit: { # key[0] is assignment of a neighbour, key[1] is value of the link.
            ("black", 0): "black", 
            ("black", 1): "red", 
            ("red", 0): "red", 
            ("red", 1): "black",
            }[(colour, syndrome_bit)] 

        while assigned_unvisited:  
            site = assigned_unvisited.popleft() 
            for neighbour in code.ising_neighbourhood(site): 
                if neighbour not in assignments: 
                    ising_link = code.ising_link(site, neighbour) 
                    link_syndrome = syndrome[code.ordered_plaquettes.index(ising_link)] # z-type syndrome 
                    colour = get_assignment(assignments[site], link_syndrome) 
                    assignments[neighbour] = colour 
#                    print("assigned {} to {}".format(neighbour, colour)) 
                    get_list[colour].append(neighbour) 
                    assigned_unvisited.append(neighbour) 

        error_support = min(black, red, key=lambda list_: len(list_)) # list of sites represented as tuples (i, j) 
        error_support = np.array([1 if site in error_support else 0 for _, site in enumerate(code.ordered_qubits)]) # binary arrary 
        logical_support = support(self.conserved_logicals(code)[0]) 
        return np.array(np.count_nonzero(error_support & logical_support)%2) # binary array giving corrections to logical operator measurement outcomes. 

    @property 
    def label(): 
        return "Colour code logical Z readout assuming 100% bias."  

##################################################### CORE ##################################################################

def _run_once_ft(code, time_steps, num_cycles, error_model, decoder, readout, error_probability, 
    measurement_error_probability, rng, results="simple", initial_error=None):
    """Implements run_once and run_once_ftp functions."""       

    # generate step_error, step_syndrome and step_measurement_error for each time step
    if initial_error is not None: 
        residual_error = initial_error 
    else: 
        residual_error = np.zeros(2*code.n_k_d[0], dtype=int) 

    if results == "full_history": 
        residual_error_history = np.zeros((num_cycles+1, 2*code.n_k_d[0]), dtype=int)
        residual_error_history[0,:] = residual_error

    for cycle in range(num_cycles):  

        step_errors, step_syndromes, step_measurement_errors = [], [], []

        # print("Residual error:") # REMOVE
        # print(code.ascii_art(pauli=code.new_pauli(bsf=residual_error))) # REMOVE
        for _ in range(time_steps):
            # step_error: random e rror based on error probability
            step_error = error_model.generate(code, error_probability, rng)
            step_errors.append(step_error)
            # step_syndrome: stabilizers that do not commute with the error
            step_syndrome = pt.bsp(step_error, code.stabilizers.T)
            step_syndromes.append(step_syndrome)
            residual_syndrome = pt.bsp(residual_error, code.stabilizers.T) 
            # step_measurement_error: random syndrome bit flips based on measurement_error_probability
            if measurement_error_probability:
                step_measurement_error = rng.choice(
                    (0, 1),
                    size=step_syndrome.shape,
                    p=(1 - measurement_error_probability, measurement_error_probability)
                ) 
            else:
                step_measurement_error = np.zeros(step_syndrome.shape, dtype=int)
            step_measurement_errors.append(step_measurement_error)

        if logger.isEnabledFor(logging.DEBUG): # for performance, perhaps?  
            try: 
                for i, step_error in enumerate(step_errors): 
                    logger.debug('run: step_error[{}]:\n{}'.format(i, code.ascii_art(pauli=code.new_pauli(bsf=step_error)))) 
                for i, step_syndrome in enumerate(step_syndromes): 
                    logger.debug("run: step_syndrome[{}]:\n{}".format(i, code.ascii_art(syndrome=step_syndrome)))  
                for i, step_measurement_error in enumerate(step_measurement_errors): 
                    logger.debug("run: step_measurement_error[{}]:\n{}".format(i, code.ascii_art(syndrome=step_measurement_error))) 
            except AttributeError: 
                logger.debug('run: step_errors={}'.format(step_errors))
                logger.debug('run: step_syndromes={}'.format(step_syndromes))
                logger.debug('run: step_measurement_errors={}'.format(step_measurement_errors))

        # error: sum of errors at each time step
        error = np.bitwise_xor.reduce([residual_error] + step_errors)

        if logger.isEnabledFor(logging.DEBUG):
            try: 
                logger.debug("run: total error:\n{}".format(code.ascii_art(pauli=code.new_pauli(bsf=error)))) 
            except AttributeError: 
                logger.debug('run: error={}'.format(error))

        # syndrome: apply measurement_error at times t-1 and t to syndrome at time t 
        step_measurement_errors.append(np.zeros(step_syndrome.shape, dtype=int)) # ensure smooth t=0 bc 
        syndrome = [] 
        for t in range(time_steps):
            syndrome.append(step_measurement_errors[t - 1] ^ step_syndromes[t] ^ step_measurement_errors[t])
        syndrome[0] = np.bitwise_xor.reduce([syndrome[0], residual_syndrome]) 
        # convert syndrome to 2d numpy array
        syndrome = np.array(syndrome)

        if logger.isEnabledFor(logging.DEBUG):
            try: 
                for i, syndrome_slice in enumerate(syndrome): 
                    logger.debug("run: syndrome[{}]: \n{}".format(i, code.ascii_art(syndrome=syndrome[i]))) 
            except AttributeError: 
                logger.debug('run: syndrome={}'.format(syndrome))

        # decoding: boolean or best match recovery operation based on decoder
        ctx = {'error_model': error_model, 'error_probability': error_probability, 'error': error,
               'step_errors': step_errors, 'measurement_error_probability': measurement_error_probability,
               'step_measurement_errors': step_measurement_errors, 'rng': rng} 
        
        recovery = decoder.decode_ft(code, time_steps, syndrome, **ctx) 
        residual_error = recovery ^ error

        if results == "full_history": 
            residual_error_history[cycle+1,:] = residual_error 

        if logger.isEnabledFor(logging.DEBUG):
            try: 
                logger.debug("run: recovery:\n{}".format(code.ascii_art(pauli=code.new_pauli(bsf=recovery)))) 
                logger.debug("run: residual_error:\n{}".format(code.ascii_art(pauli=code.new_pauli(bsf=residual_error)))) 
            except AttributeError: 
                logger.debug('run: recovery={}'.format(recovery))
                logger.debug("run: residual_error={}".format(residual_error)) 


    # success checks
    # compute readout error from measurement in the computational basis. 
    qubit_readout_error = readout.generate_error(code, measurement_error_probability, rng) 
    conserved_stabilisers_support = np.apply_along_axis(support, 1, readout.conserved_stabilisers(code)) 
    stabiliser_readout_error = np.dot(qubit_readout_error, conserved_stabilisers_support.T)%2
    readout_syndrome = pt.bsp(residual_error, readout.conserved_stabilisers(code).T) ^ stabiliser_readout_error 
    correction = readout.decode(code, readout_syndrome)

    included_readout_syndrome = np.zeros(np.shape(code.stabilizers)[0]) # for logging.  
    for s, stabiliser in zip(readout_syndrome, readout.conserved_stabilisers(code)): 
        index = np.where(np.all(code.stabilizers==stabiliser, axis=1))[0][0] 
        included_readout_syndrome[index] = s 
    if logger.isEnabledFor(logging.DEBUG): 
        try: 
            logger.debug("run: qubit_readout_error ('X' marks qubit measurement errors, not necessarily indicating a Pauli X\n{}".format(
                code.ascii_art(pauli=code.new_pauli(bsf=np.concatenate((qubit_readout_error, np.zeros(code.n_k_d[0], dtype=int))))))) 
            logger.debug("run: readout_syndrome (included into full code):\n{}".format(code.ascii_art(syndrome=included_readout_syndrome))) 
        except AttributeError: 
            logger.debug("run: qubit_readout_error={}".format(qubit_readout_error)) 
            logger.debug("run: included_readout_syndrome={}".format(included_readout_syndrome)) 
    # if logger.isEnabledFor(logging.DEBUG): 
    #     logger.debug(f"run: residual_error={residual_error}") 
    #     logger.debug(f"run: qubit_readout_error={qubit_readout_error}")
    #     logger.debug(f"run: stabiliser_readout_error={stabiliser_readout_error}") 
    #     logger.debug(f"run: readout_syndrome={readout_syndrome}") 
    #     logger.debug(f"run: correction={correction}") 

    # sanity checks 
    # commutes_with_stabilizers = np.all(pt.bsp(recovered, readout.conserved_stabilisers(code).T) == 0)
    # if not commutes_with_stabilizers:
    #     log_data = {
    #         # models
    #         'code': repr(code),
    #         'error_model': repr(error_model),
    #         'decoder': repr(decoder),
    #         # variables
    #         'error': pt.pack(error),
    #         'recovery': pt.pack(recovery),
    #         # step variables
    #         'step_errors': [pt.pack(v) for v in step_errors],
    #         'step_measurement_errors': [pt.pack(v) for v in step_measurement_errors],
    #     }
    #     logger.warning('RECOVERY DOES NOT RETURN TO (+1)-EIGENSPACE OF CONSERVED STABILIZERS: {}'.format(json.dumps(log_data, sort_keys=True)))
    #     logger.warning(str(code.ascii_art(pauli=code.new_pauli(bsf=residual_error)))) 
    #     logger.warning(str(code.ascii_art(pauli=code.new_pauli(bsf=correction))))     
    #     meas_err_as_qubit_err = np.concatenate([qubit_readout_error, [0]*code.n_k_d[0]])  
    #     logger.warning(str(code.ascii_art(pauli=code.new_pauli(bsf=meas_err_as_qubit_err)))) 
    #     total = meas_err_as_qubit_err ^ residual_error ^ correction 
    #     logger.warning(str(code.ascii_art(pauli=code.new_pauli(bsf=total)))) 


    commutes_with_logicals = pt.bsp(residual_error, readout.conserved_logicals(code).T) 
    conserved_logical_support = np.apply_along_axis(support, 1, readout.conserved_logicals(code)) 
    measurement_introduces_error = np.dot(qubit_readout_error, conserved_logical_support.T)%2 
    success = np.all((commutes_with_logicals + measurement_introduces_error + correction)%2 == 0) 
    # if logger.isEnabledFor(logging.DEBUG):
    #     logger.debug('run: commutes_with_stabilizers={}'.format(commutes_with_stabilizers))
    #     logger.debug('run: commutes_with_logicals={}'.format(commutes_with_logicals))

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('run: success={}'.format(success))

    data = {
#        'error_weight': pt.bsf_wt(np.array(step_errors)),
        'success': success,
    }
    if results == "full_history":
        data["history"] = residual_error_history 

    return data

def run_ft(code, time_steps, num_cycles, error_model, decoder, readout, error_probability, measurement_error_probability, max_runs=None, max_failures=None, random_seed=None, results="simple", initial_error=None): 

    if max_runs is None and max_failures is None: 
        max_runs = 1 

    wall_time_start = time.perf_counter() 

    runs_data = {
        'code': code.label,
        'n_k_d': code.n_k_d,
        'time_steps': time_steps,
        'num_cycles': num_cycles, 
        'error_model': error_model.label,
        'decoder': decoder.label,
        'error_probability': error_probability,
        'measurement_error_probability': measurement_error_probability,
        'n_run': 0,
        'n_success': 0,
        'n_fail': 0,
        'error_weight_total': 0,
        'error_weight_pvar': 0.0,
        'logical_failure_rate': 0.0,
        'physical_error_rate': 0.0,
        'wall_time': 0.0,
        'seed': 0, 
        'version':qcext.__version__,
    }
    if results == "full_history": 
        runs_data["history"] = np.zeros((num_cycles+1, 2*code.n_k_d[0]), dtype=int)
    if initial_error is not None: 
        runs_data["initial_error"] = initial_error.tolist() 
    else: 
        runs_data["initial_error"] = None

    seed_sequence = np.random.SeedSequence(random_seed)
    runs_data['seed'] = seed_sequence.entropy 
    rng = np.random.default_rng(seed_sequence) 

    while ((max_runs is None or runs_data['n_run'] < max_runs)
       and (max_failures is None or runs_data['n_fail'] < max_failures)):
        # run simulation
        data = _run_once_ft(code, 1, num_cycles, error_model, decoder, readout, error_probability, measurement_error_probability,
                         rng, results=results, initial_error=initial_error)
        # increment run counts
        runs_data['n_run'] += 1
        if data['success']:
            runs_data['n_success'] += 1
        else:
            runs_data['n_fail'] += 1
        if results == "full_history": 
            runs_data["history"] += data["history"] 

    runs_data['wall_time'] = time.perf_counter() - wall_time_start 
    if results == "full_history": 
        runs_data["history"] = runs_data["history"].tolist() # for serialization with JSON 

    _add_rate_statistics(runs_data) 

    return runs_data 





def main(): 
    logger.setLevel(logging.DEBUG)  
    code = Color666CodeNoisy(9) 
    noise = RestrictedDepolarizingErrorModel(code.noisy_qubits) 
    ising_decoder = IsingDecoder() 
    z_readout = Color666NoisyReadoutZ() 
    p = 0.1 
    m = 0.1
    rng = np.random.default_rng() 
    result = _run_once_ft(code, 1, 5, noise, ising_decoder, z_readout, p, m, rng) 
    print(result) 



if __name__ == "__main__": 
    main() 