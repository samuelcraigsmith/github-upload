import functools 
import logging

import numpy as np 

from qcext.modelext import Readout 

logger = logging.getLogger(__name__) 

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