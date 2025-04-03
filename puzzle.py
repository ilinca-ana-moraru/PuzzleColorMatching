import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import List

from fragment import *
from side import *
from sides_comparison import *
from utils import *
from puzzle import *

def create_sides_comparisons(fragments: List[Fragment]):
    sides_comparisons = []
    for fr_idx1 in range(len(fragments) - 1):
        for side_idx1 in range(len(fragments[fr_idx1].sides)):
            side1 = fragments[fr_idx1].sides[side_idx1]
            
            if all(len(side1.value) >= len(fragments[fr_idx1].sides[side_idx].value) for side_idx in range(len(fragments[fr_idx1].sides))):
                for fr_idx2 in range(fr_idx1 + 1, len(fragments)):
                    for side_idx2 in range(len(fragments[fr_idx2].sides)):
                        side2 = fragments[fr_idx2].sides[side_idx2]
                        
                        if len(side1.value) == len(side2.value):
                            sides_comparisons.append(SidesComparison(fragments, side1, side2))
                            # print(f"fragment {fr_idx1} side {side_idx1} VS fragment {fr_idx2} side {side_idx2}")

    
    return sides_comparisons  


def sort_sides_comparisons(sides_comparisons: List[SidesComparison]):
        return sorted(sides_comparisons, key=lambda x: x.score)


def draw_red_border(fragment:Fragment, side: Side):
    fragment_value = fragment.value.copy()

    fragment_value[side.side_indexes_of_fragment[:,0],side.side_indexes_of_fragment[:,1]] = [255, 0, 0, 255]
    return fragment_value






def merge_fragments_two_by_two(fragments: List[Fragment], sides_comparisons: List[SidesComparison]):
    banned_fragments_idx = []
    new_fragments = []
    new_fr_idx = 0
    for comp in sides_comparisons:
        if comp.side1.fragment_idx not in banned_fragments_idx and comp.side2.fragment_idx not in banned_fragments_idx:
            new_fragment_value = comp.value

            # print(f"new fragment shapes: {new_fragment_value.shape}")
            new_fragment = Fragment(new_fragment_value, new_fr_idx)
            new_fragments.append(new_fragment)
            banned_fragments_idx.append(comp.side1.fragment_idx)
            banned_fragments_idx.append(comp.side2.fragment_idx)
            new_fr_idx += 1
    return new_fragments


