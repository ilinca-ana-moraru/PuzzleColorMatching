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
                            sides_comparisons.append(SidesComparison(side1, side2))
                            # print(f"fragment {fr_idx1} side {side_idx1} VS fragment {fr_idx2} side {side_idx2}")

    
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

    # cx, cy = find_centroid(image)
    # centroid = [cx, cy]

    p1 = side.side_indexes_of_fragment[0]
    p2 = side.side_indexes_of_fragment[-1]

    x, y = p2[0] - p1[0], p2[1] - p1[1]
    th1 = np.degrees(np.arctan2(y, x))

    if side_type == 1:
        p1 = [0, w-1]
        p2 = [h-1, w-1]
    else:
        p1 = [h-1, 0]
        p2 = [0, 0]
    x, y = p2[0] - p1[0], p2[1] - p1[1]
    th2 = np.degrees(np.arctan2(y, x))

    rotation_angle = th2 - th1
    # if abs(rotation_angle) < 1: 
    #     return image

    # if abs(rotation_angle - 180) < 1:
    #     return cv.flip(image, 1)  

    if abs(rotation_angle) < 5:
        return image
    
    elif rotation_angle > 80 and rotation_angle < 100 or rotation_angle < -260 and rotation_angle > -280:
        image = cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE)

    elif abs(rotation_angle) > 170 and abs(rotation_angle) < 190 or rotation_angle < -170 and rotation_angle > -190:
        image = cv.rotate(image, cv.ROTATE_180)
    
    elif rotation_angle < -80 and rotation_angle > -100  or rotation_angle > 260 and rotation_angle < 280:
        image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
    else:
        print(f"invalid rotation angle {rotation_angle}")
        return 0
    # plt.imshow(image)
    # plt.show()
    return image

    # rotation_matrix = cv.getRotationMatrix2D(centroid, rotation_angle, 1)
    # cos_angle = abs(rotation_matrix[0, 0])
    # sin_angle = abs(rotation_matrix[0, 1])

    # new_w = int(h * sin_angle + w * cos_angle)
    # new_h = int(h * cos_angle + w * sin_angle)

    # rotation_matrix[0, 2] += (new_w - w) / 2
    # rotation_matrix[1, 2] += (new_h - h) / 2

    # rotated_image = cv.warpAffine(image, rotation_matrix, (new_w, new_h))
    # rotated_image = fix_border(rotated_image)

    # return rotated_image

def two_fragments_merger(fragments: List[Fragment], comp: SidesComparison):
    rotated_fragment1 = rotate_fragment(fragments, comp.side1, 1)
    rotated_fragment2 = rotate_fragment(fragments, comp.side2, 2)

    # h1, w1 = rotated_fragment1.shape[:2]
    # h2, w2 = rotated_fragment2.shape[:2]

    # target_h = min(h1, h2)
    # target_w = min(w1, w2)

    # rotated_fragment1 = cv.resize(rotated_fragment1, (target_w, target_h), interpolation=cv.INTER_LINEAR)
    # rotated_fragment2 = cv.resize(rotated_fragment2, (target_w, target_h), interpolation=cv.INTER_LINEAR)
    new_fragment = np.hstack((rotated_fragment1, rotated_fragment2))
    return new_fragment

def merge_fragments_two_by_two(fragments: List[Fragment], sides_comparisons: List[SidesComparison]):
    banned_fragments_idx = []
    new_fragments = []
    new_fr_idx = 0
    for comp in sides_comparisons:
        if comp.side1.fragment_idx not in banned_fragments_idx and comp.side2.fragment_idx not in banned_fragments_idx:
            new_fragment_value = two_fragments_merger(fragments, comp)

            # print(f"new fragment shapes: {new_fragment_value.shape}")
            new_fragment = Fragment(new_fragment_value, new_fr_idx)
            new_fragments.append(new_fragment)
            banned_fragments_idx.append(comp.side1.fragment_idx)
            banned_fragments_idx.append(comp.side2.fragment_idx)
            new_fr_idx += 1
    return new_fragments

def show_similar_colored_edges(fragments: List[Fragment], sorted_sides_comparisons: List[SidesComparison]):


    for comp in sorted_sides_comparisons:
        print(f"score: {comp.score}")
        print(f"is correct: {comp.is_valid_match}")
        print(f"grad presence: {comp.grad_presence} grad match: {comp.grad_match} grad score: {comp.grad_score}")


        #############################show the 2 images side by side
        # fragment1 = draw_red_border(fragments[comp.side1.fragment_idx], comp.side1)  
        # fragment2 = draw_red_border(fragments[comp.side2.fragment_idx], comp.side2)  
        # plt.figure(figsize=(12, 6)) 

        # plt.subplot(1, 2, 1)  
        # plt.imshow(fragment1) 
        # plt.title(f"Fragment {comp.side1.fragment_idx}  {comp.side1.side_idx}")

        # plt.subplot(1, 2, 2)  
        # plt.imshow(fragment2)
        # plt.title(f"Fragment {comp.side2.fragment_idx}  {comp.side2.side_idx}")

        # plt.show()

        new_fragment = two_fragments_merger(fragments, comp)
        plt.imshow(new_fragment)
        plt.show()

        ########print just the sides colors
        # side1_printable_values = comp.reversed_side1_value[np.newaxis, :, :]
        # side2_printable_values = comp.side2.value[np.newaxis, :, :]
        # plt.figure(figsize=(6, 1))
        
        # plt.subplot(2, 1, 1)
        # plt.imshow(side1_printable_values)
        # plt.title(f" {comp.side1.side_idx} of Fragment {comp.side1.fragment_idx}")
        # plt.axis("off")
        
        # plt.subplot(2, 1, 2)
        # plt.imshow(side2_printable_values)
        # plt.title(f" {comp.side2.side_idx} of Fragment {comp.side2.fragment_idx}")
        # plt.axis("off")
        
        # plt.show()

        #########print the sides grads
        # side1_printable_grad = np.uint8(comp.reversed_side1_grad[np.newaxis, :]*255)
        # side2_printable_grad = np.uint8(comp.side2.grad[np.newaxis, :]*255)

        # side1_grad_img = apply_Grad(fragments[comp.side1.fragment_idx].value)
        # plt.imshow(side1_grad_img, cmap="gray")
        # plt.show()

        # side2_grad_img = apply_Grad(fragments[comp.side2.fragment_idx].value)
        # plt.imshow(side2_grad_img, cmap="gray")
        # plt.show()

        # plt.figure(figsize=(6, 1))
        
        # plt.subplot(2, 1, 1)
        # plt.imshow(side1_printable_grad, cmap="gray")
        # plt.title(f" {comp.side1.side_idx} of Fragment {comp.side1.fragment_idx}")
        # plt.axis("off")
        
        # plt.subplot(2, 1, 2)
        # plt.imshow(side2_printable_grad, cmap="gray")
        # plt.title(f" {comp.side2.side_idx} of Fragment {comp.side2.fragment_idx}")
        # plt.axis("off")
        
        # plt.show()
