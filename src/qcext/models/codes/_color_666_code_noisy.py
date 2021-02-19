import functools 
from itertools import product
import logging 

import numpy as np 

from qecsim import paulitools as pt
from qecsim.model import cli_description
from qecsim.models.color import Color666Code, Color666Pauli 

logger = logging.getLogger(__name__)  


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

    @functools.lru_cache(maxsize=None) 
    def color(self, plaquette):
        if not self.is_plaquette(plaquette): 
            raise IndexError("{} is not a plaquette".format(plaquette)) 
        i, j = plaquette 
        color = ["red", "green", "blue"][i%3] # red edge is vertical, green edges is horizontal, blue edge is the diagonal. 
        return color  