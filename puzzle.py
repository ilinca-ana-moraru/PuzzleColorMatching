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
from tqdm import tqdm
from global_values import *

def create_sides_comparisons(fragments: List[Fragment]):
    sides_comparisons = []
    for fr_idx1 in tqdm(range(len(fragments) - 1)):
        for side_idx1 in range(len(fragments[fr_idx1].sides)):
            side1 = fragments[fr_idx1].sides[side_idx1]
            
            if all(len(side1.value) >= len(fragments[fr_idx1].sides[side_idx].value) for side_idx in range(len(fragments[fr_idx1].sides))):
                for fr_idx2 in range(fr_idx1 + 1, len(fragments)):
                    for side_idx2 in range(len(fragments[fr_idx2].sides)):
                        side2 = fragments[fr_idx2].sides[side_idx2]
                        if len(side1.value) == len(side2.value):
                            if  ROTATING_PIECES or ((side1.side_idx == 2 and side2.side_idx == 0) or (side1.side_idx == 1 and side2.side_idx == 3) \
                            or (side1.side_idx == 0 and side2.side_idx == 2) or (side1.side_idx == 3 and side2.side_idx == 1)):
                                sides_comparisons.append(SidesComparison(fragments, side1, side2))
                                # print(f"fragment {fr_idx1} side {side_idx1} VS fragment {fr_idx2} side {side_idx2}")

    
    return sides_comparisons  


def sort_sides_comparisons(sides_comparisons: List[SidesComparison]):
        return sorted(sides_comparisons, key=lambda x: x.score)





# def draw_red_border(fragment:Fragment, side: Side):
#     fragment_value = fragment.value.copy()

#     fragment_value[side.side_indexes_of_fragment[:,0],side.side_indexes_of_fragment[:,1]] = [255, 0, 0, 255]
#     return fragment_value





# def rotate_fragment(fragments, side, side_type):
#     image = fragments[side.fragment_idx].value
   
#     h, w = image.shape[:2]

#     p1 = side.side_indexes_of_fragment[0]
#     p2 = side.side_indexes_of_fragment[-1]

#     x, y = p2[0] - p1[0], p2[1] - p1[1]
#     th1 = np.degrees(np.arctan2(y, x))

#     if side_type == 1:
#         p1 = [0, w-1]
#         p2 = [h-1, w-1]
#     else:
#         p1 = [h-1, 0]
#         p2 = [0, 0]
#     x, y = p2[0] - p1[0], p2[1] - p1[1]
#     th2 = np.degrees(np.arctan2(y, x))

#     rotation_angle = th2 - th1
  

#     if abs(rotation_angle) < 5:
#         return image
    
#     elif rotation_angle > 80 and rotation_angle < 100 or rotation_angle < -260 and rotation_angle > -280:
#         image = cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE)

#     elif abs(rotation_angle) > 170 and abs(rotation_angle) < 190 or rotation_angle < -170 and rotation_angle > -190:
#         image = cv.rotate(image, cv.ROTATE_180)
    
#     elif rotation_angle < -80 and rotation_angle > -100  or rotation_angle > 260 and rotation_angle < 280:
#         image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
#     else:
#         print(f"invalid rotation angle {rotation_angle}")
#         return 0
  
#     return image



# def two_fragments_merger(fragments, side1, side2):
#     rotated_fragment1 = rotate_fragment(fragments, side1, 1)
#     rotated_fragment2 = rotate_fragment(fragments, side2, 2)


#     new_fragment = np.hstack((rotated_fragment1, rotated_fragment2))
#     return new_fragment


# def merge_fragments_two_by_two(fragments: List[Fragment], sides_comparisons: List[SidesComparison]):
#     banned_fragments_idx = []
#     new_fragments = []
#     new_fr_idx = 0
#     for comp in sides_comparisons:
#         if comp.side1.fragment_idx not in banned_fragments_idx and comp.side2.fragment_idx not in banned_fragments_idx:
#             new_fragment_value = two_fragments_merger(fragments, comp.side1, comp.side2)

#             new_fragment = Fragment(new_fragment_value, new_fr_idx)
#             new_fragments.append(new_fragment)
#             banned_fragments_idx.append(comp.side1.fragment_idx)
#             banned_fragments_idx.append(comp.side2.fragment_idx)
#             new_fr_idx += 1
#     return new_fragments








