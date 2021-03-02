from collections import deque 
import functools
import logging 

import numpy as np 
from qecsim import paulitools as pt
from qecsim.model import ErrorModel, cli_description

from qcext.models.readouts import Color666ReadoutZ 
from qcext.ptext import support 

logger = logging.getLogger(__name__) 

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
                    # print("assigned {} to {}".format(neighbour, colour)) 
                    get_list[colour].append(neighbour) 
                    assigned_unvisited.append(neighbour) 

        error_support = min(black, red, key=lambda list_: len(list_)) # list of sites represented as tuples (i, j) 
        error_support = np.array([1 if site in error_support else 0 for _, site in enumerate(code.ordered_qubits)]) # binary arrary 
        logical_support = support(self.conserved_logicals(code)[0]) 
        return np.array(np.count_nonzero(error_support & logical_support)%2) # binary array giving corrections to logical operator measurement outcomes. 

    @property
    def label(self):
        return "Colour code logical Z readout assuming 100% bias."

