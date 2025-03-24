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
            
            # Check if side1.value has the largest length compared to other sides in the fragment
            if all(len(side1.value) >= len(fragments[fr_idx1].sides[side_idx].value) for side_idx in range(len(fragments[fr_idx1].sides))):
                for fr_idx2 in range(fr_idx1 + 1, len(fragments)):
                    for side_idx2 in range(len(fragments[fr_idx2].sides)):
                        side2 = fragments[fr_idx2].sides[side_idx2]
                        
                        # Compare side1 and side2 if their lengths match
                        if len(side1.value) == len(side2.value):
                            sides_comparisons.append(SidesComparison(side1, side2))
    
    return sides_comparisons  


def sort_sides_comparisons(sides_comparisons: List[SidesComparison]):
        return sorted(sides_comparisons, key=lambda x: x.score)


def draw_red_border(fragment:Fragment, side: Side):
    fragment_value = fragment.value.copy()

    fragment_value[side.side_indexes_of_fragment[:,0],side.side_indexes_of_fragment[:,1]] = [255, 0, 0, 255]
    return fragment_value



def remove_transparent_borders(image):


    alpha_channel = image[:, :, 3]  
    h, w = image.shape[:2]
    
    row_transparency_count = np.sum(alpha_channel == 0, axis=1) / w
    col_transparency_count = np.sum(alpha_channel == 0, axis=0) / h


    rows_to_remove = np.where(row_transparency_count == 1)[0]
    cols_to_remove = np.where(col_transparency_count == 1)[0]
    print(f"removing rows: {rows_to_remove}")
    print(f"rows count: {row_transparency_count}")
    print(f"removing columns: {cols_to_remove}")
    print(f"columns count: {col_transparency_count}")

    if rows_to_remove.size > 0:
        image = np.delete(image, rows_to_remove, axis=0) 

    if cols_to_remove.size > 0:
        image = np.delete(image, cols_to_remove, axis=1)  

    return image

def rotate_fragment(fragments: List[Fragment],side, side_type):
    image = fragments[side.fragment_idx].value
    needs_bordering_h = 0
    needs_bordering_w = 0
    h, w = image.shape[:2]
    original_h, original_w = image.shape[:2]

    cx, cy = find_centroid(image)
    centroid = [cx, cy]
    # print(f"initial centroid: {cx} {cy}")

    # if h%2 == 0:
        
    #     bordered_image = np.zeros((h + 1, w, 4), dtype=np.uint8)
    #     bordered_image[0:h, 0:w] = image 
    #     image = bordered_image
    #     h, w = image.shape[:2]
    #     needs_bordering_h = 1

    #     cx +=1
    #     print("bordered height")
    #     print(f"centroid: {cx} {cy}")

    # if w%2 == 0:
        
    #     bordered_image = np.zeros((h, w + 1, 4), dtype=np.uint8)
    #     bordered_image[0:h, 0:w] = image 
    #     image = bordered_image
    #     h, w = image.shape[:2]
    #     needs_bordering_w = 1
    #     cy +=1
    #     print("bordered width")
    #     print(f"centroid: {cx} {cy}")
    # print(f"updated centroid: {cx} {cy}")

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

    rotation_matrix = cv.getRotationMatrix2D(centroid, rotation_angle, 1)
    cos_angle = abs(rotation_matrix[0, 0])
    sin_angle = abs(rotation_matrix[0, 1])

    new_w = int(h * sin_angle + w * cos_angle)
    new_h = int(h * cos_angle + w * sin_angle)

    rotation_matrix[0, 2] += (new_w - w) / 2
    rotation_matrix[1, 2] += (new_h - h) / 2

    rotated_image = cv.warpAffine(image, rotation_matrix, (new_w, new_h))
    rotated_image = fix_border(rotated_image)
    # rotated_image = cv.warpAffine(image, rotation_matrix, (new_w, new_h),borderMode=cv.BORDER_TRANSPARENT)

    # print(f"rotated fragment bordered shape: {rotated_image.shape}")
    # rotated_image = remove_transparent_borders(rotated_image)
    # print(f"rotated fragment final shape: {rotated_image.shape}")
    return rotated_image

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
        # print(f"distances: {comp.points_distances}")


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