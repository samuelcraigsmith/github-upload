
from itertools import combinations
import math
import logging

import numpy as np

from qecsim.model import Decoder
from qecsim import graphtools as gt
from qecsim import paulitools as pt

logger = logging.getLogger(__name__)


class ColorMatchingDecoder(Decoder):
    """
    Implements a matching decoder on the color code.

    Matches by unfolding the color code and treating it like a toric code.
    """
    def decode(self, code, syndrome, **kwargs):
        x_defects, z_defects = code.syndrome_to_plaquette_indices(syndrome)
        total_correction = code.new_pauli().to_bsf()

        for defect_type in "XZ":
            error_type = {"X": "Z", "Z": "X"}[defect_type]
            defects = {"X": x_defects, "Z": z_defects}[defect_type]
            logical_error = {"X": code.logical_xs[0],
                             "Z": code.logical_zs[0]}[error_type]
            anticommuting_logical = {"X": code.logical_xs[0],
                                     "Z": code.logical_zs[0]}[defect_type]

            unfolded_defects = self.unfold_defects(code, defects)

            g = gt.SimpleGraph()
            boundary_defects = {"red": [], "green": []}
            for edge in combinations(unfolded_defects, 2):
                weight = self.distance(code, *edge)
                g.add_edge(*edge, weight)
            for unique_label, defect in enumerate(unfolded_defects):
                weight, color = self.distance(code, defect)
                boundary_defect = color + str(unique_label)
                boundary_defects[color].append(boundary_defect)
                g.add_edge(defect, boundary_defect, weight)
            for edge in combinations(boundary_defects["red"]
                                     + boundary_defects["green"], 2):
                g.add_edge(*edge, 0)

            matching = gt.mwpm(g)

            red_boundary_matchings = 0
            for match in matching:
                if (  # condition for defect matching to red boundary.
                        any(["red" in defect for defect in match]) and
                        any([isinstance(defect, tuple) for defect in match])
                ):
                    red_boundary_matchings = (red_boundary_matchings + 1) % 2

            correction = self.get_any_correction(code, defects, error_type)

            if (
                    red_boundary_matchings !=
                    pt.bsp(correction, anticommuting_logical)
            ):
                correction = correction ^ logical_error

            total_correction = total_correction ^ correction

        return total_correction

    def unfold_defects(self, code, defects, **kwargs):
        """
        Unfold a set of defects (plaquette sites).

        Does not depend on the Pauli type of the defects.
        """
        unfolded_defects = []
        for defect in defects:
            if code.color(defect) == "green":
                unfolded_defects.append(defect)
            elif code.color(defect) == "red":
                i, j = defect
                reflected_defect = (j, i)
                unfolded_defects.append(reflected_defect)
            elif code.color(defect) == "blue":
                i, j = defect
                reflected_defect = (j, i)
                unfolded_defects.append(defect)
                unfolded_defects.append(reflected_defect)
        return unfolded_defects

    def distance(self, code, site1, *args):
        """
        Compute isotropic Euclidean distance between points.

        After shearing the code to make it isotropic.
        Compute Euclidean distance between a point and the nearest
        valid boundary after shearing the code to make it isotropic.
        Distance to edges are slightly off (it is not quite the perpendicular
        distance to the edge)
        When matching to an edge, return (distance, boundary_colour)

        TODO: test the distance to the boundary part of this method.
        """
        shear = np.array([[-1/2, -math.sqrt(3)/2], [1, 0]])
        if args:
            site2 = args[0]
            displacement = np.array(site1)-np.array(site2)
            isotropic_distance = np.linalg.norm(displacement@shear)
            return isotropic_distance
        else:  # distance to boundary
            i, j = site1
            distance_to_closest_boundary = min(i, code.bound-i)
            if distance_to_closest_boundary == i:
                boundary_colour = "red"
            else:
                boundary_colour = "green"
            displacement = np.array((distance_to_closest_boundary, 0))
            isotropic_distance = np.linalg.norm(displacement@shear)
            return isotropic_distance, boundary_colour

    def get_any_correction(self, code, defects, error_type):
        correction = code.new_pauli().to_bsf()
        for defect in defects:
            i, j = defect
            if code.color(defect) == "red":
                sites = [(i, k) for k in range(j) if k not in range(-1, j, 3)]
            elif code.color(defect) == "blue":
                sites = [(k, j) for k in range(j, i)
                         if k not in range(j-1, i, 3)]
            elif code.color(defect) == "green":
                sites = [(k, j) for k in range(i, code.bound+1)
                         if k not in range(i, code.bound+1, 3)]
            destabilizer = code.new_pauli().site(error_type, *sites).to_bsf()
            correction = correction ^ destabilizer
        return correction

    @property
    def label(self):
        return "Color Matching Decoder"
