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
    for fr_idx1 in range(len(fragments)-1):
        for side_idx1 in range(len(fragments[fr_idx1].sides)):
            side1 = fragments[fr_idx1].sides[side_idx1]
            for fr_idx2 in range(fr_idx1+1, len(fragments)):
                for side_idx2 in range(len(fragments[fr_idx2].sides)):
                    side2 = fragments[fr_idx2].sides[side_idx2]
                    if len(side1.value) == len(side2.value):
                        sides_comparisons.append(SidesComparison(side1, side2))
    return sides_comparisons


def sort_sides_comparisons(sides_comparisons: List[SidesComparison]):
        return sorted(sides_comparisons, key=lambda x: x.score)


def draw_red_border(fragment:Fragment, side: Side):
    fragment_value = fragment.value.copy()

    fragment_value[side.side_indexes_of_fragment[:,0],side.side_indexes_of_fragment[:,1]] = [255, 0, 0, 255]
    return fragment_value



def rotate_fragment(fragments: List[Fragment],side, side_type):
    image = fragments[side.fragment_idx].value
    h, w = image.shape[:2]

    p1 = side.side_indexes_of_fragment[0]
    p2 = side.side_indexes_of_fragment[1]

    x, y = p2[0] - p1[0], p2[1] - p1[1]
    th1 = np.degrees(np.arctan2(y, x))

    if side_type == 1:
        p1 = [0, h-1]
        p2 = [h-1, h-1]
    else:
        p1 = [h-1, 0]
        p2 = [0, 0]
    x, y = p2[0] - p1[0], p2[1] - p1[1]
    th2 = np.degrees(np.arctan2(y, x))

    rotation_angle = th2 - th1
    centroid = [fragments[side.fragment_idx].cx, fragments[side.fragment_idx].cy]

    # if abs(rotation_angle) < 1: 
    #     return image

    # if abs(rotation_angle - 180) < 1:
    #     return cv.flip(image, 1)  

    rotation_matrix = cv.getRotationMatrix2D(centroid, rotation_angle, 1)
    rotated_image = cv.warpAffine(image, rotation_matrix, (w, h))
    
    return rotated_image

def two_fragments_merger(fragments: List[Fragment], comp: SidesComparison):
    rotated_fragment1 = rotate_fragment(fragments, comp.side1, 1)
    rotated_fragment2 = rotate_fragment(fragments, comp.side2, 2)

    h1, w1 = rotated_fragment1.shape[:2]
    h2, w2 = rotated_fragment2.shape[:2]

    target_h = min(h1, h2)
    target_w = min(w1, w2)

    rotated_fragment1 = cv.resize(rotated_fragment1, (target_w, target_h), interpolation=cv.INTER_LINEAR)
    rotated_fragment2 = cv.resize(rotated_fragment2, (target_w, target_h), interpolation=cv.INTER_LINEAR)

    return np.hstack((rotated_fragment1, rotated_fragment2))


def show_similar_colored_edges(fragments: List[Fragment], sorted_sides_comparisons: List[SidesComparison]):


    for comp in sorted_sides_comparisons:
        print(f"score: {comp.score}")
        # print(f"distances: {comp.points_distances}")
        fragment1 = draw_red_border(fragments[comp.side1.fragment_idx], comp.side1)  
        fragment2 = draw_red_border(fragments[comp.side2.fragment_idx], comp.side2)  

        plt.figure(figsize=(12, 6)) 

        plt.subplot(1, 2, 1)  
        plt.imshow(fragment1) 
        plt.title(f"Fragment {comp.side1.fragment_idx}  {comp.side1.side_idx}")

        plt.subplot(1, 2, 2)  
        plt.imshow(fragment2)
        plt.title(f"Fragment {comp.side2.fragment_idx}  {comp.side2.side_idx}")

        plt.show()

        new_fragment = two_fragments_merger(fragments, comp)
        plt.imshow(new_fragment)
        plt.show()

        ########print just the sides
        side1_printable_values = comp.reversed_side1_value[np.newaxis, :, :]
        side2_printable_values = comp.side2.value[np.newaxis, :, :]
        plt.figure(figsize=(6, 1))
        
        plt.subplot(2, 1, 1)
        plt.imshow(side1_printable_values)
        plt.title(f" {comp.side1.side_idx} of Fragment {comp.side1.fragment_idx}")
        plt.axis("off")
        
        plt.subplot(2, 1, 2)
        plt.imshow(side2_printable_values)
        plt.title(f" {comp.side2.side_idx} of Fragment {comp.side2.fragment_idx}")
        plt.axis("off")
        
        plt.show()