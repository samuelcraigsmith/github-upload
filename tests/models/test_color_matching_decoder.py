import unittest
import logging
from itertools import combinations, product
from collections import Counter
from parameterized import parameterized

from qcext.models.codes import Color666CodeNoisy
from qcext.models.decoders import ColorMatchingDecoder
from qecsim import paulitools as pt

logger = logging.getLogger(__name__) 
logging.basicConfig(level=logging.WARNING)

class TestColorMatchingDecoder(unittest.TestCase): 
    code = Color666CodeNoisy(9) 
    decoder = ColorMatchingDecoder() 

    def test_unfold_defects_bulk(self): 
        defects = [(5, 3), (6, 2), (7, 4)] 
        unfolded_defects = self.decoder.unfold_defects(self.code, defects) 
        self.assertEqual(Counter(unfolded_defects), Counter([(7, 4), (5, 3), (3, 5), (2, 6)]))  

    def test_unfold_defects_blue_boundary(self):  
        defects = [(3, 2), (4, 4)] 
        unfolded_defects = self.decoder.unfold_defects(self.code, defects) 
        self.assertEqual(Counter(unfolded_defects), Counter([(2, 3), (4, 4)])) 

    def test_distance_gives_equilateral_triangle(self): 
        corners = [(0, 0), (12, 0), (12, 12)] 
        side_lengths = [self.decoder.distance(self.code, *side) for side in combinations(corners, 2)] 
        for side_length1, side_length2 in combinations(side_lengths, 2): 
            self.assertAlmostEqual(side_length1, side_length2) 

    def test_distance_gives_correct_colors(self): 
        sites = [(4, 1), (10, 1), (8, 9), (2, 9)]
        colors = [self.decoder.distance(self.code, site)[1] for site in sites] 
        self.assertEqual(colors, ["red", "green", "green", "red"]) 

    def test_distance_gives_valid_distances(self): 
        sites = [(1, 1), (4, 1)] 
        distances = [self.decoder.distance(self.code, site)[0] for site in sites] 
        self.assertGreater(distances[1], distances[0])

    def test_get_any_correction(self): 
        error = self.code.new_pauli().site("X", (2, 1), (3, 1), (11, 7)).site("Z", (9, 1), (6, 3)).to_bsf() 
        syndrome = pt.bsp(error, self.code.stabilizers.T) 
        x_defects, z_defects = self.code.syndrome_to_plaquette_indices(syndrome)  
        correction = self.decoder.get_any_correction(self.code, z_defects, "X") ^ self.decoder.get_any_correction(self.code, x_defects, "Z") 

        logger.debug("Error:\n{}".format(self.code.ascii_art(syndrome=syndrome, pauli=self.code.new_pauli(error)))) 
        logger.debug("Correction:\n{}".format(self.code.ascii_art(pauli=self.code.new_pauli(correction)))) 

        self.assertFalse(any(pt.bsp(error^correction, self.code.stabilizers.T)))

    single_qubit_errors = [] 
    for site, pauli_type in product(product(range(13), repeat=2), "XYZ"): 
        if code.is_in_bounds(site) and code.is_site(site): 
            single_qubit_errors.append(("site_{}_type_{}".format(str(site), pauli_type), code.new_pauli().site(pauli_type, site).to_bsf())) 
    @parameterized.expand(single_qubit_errors) 
    def test_decode_all_single_qubit_errors(self, _, error): 
        syndrome = pt.bsp(error, self.code.stabilizers.T) 
        correction = self.decoder.decode(self.code, syndrome) 
        self.assertFalse(any(pt.bsp(error^correction, self.code.stabilizers.T)))
        self.assertFalse(any(pt.bsp(error^correction, self.code.logicals.T))

if __name__ == "__main__": 
    unittest.main() 