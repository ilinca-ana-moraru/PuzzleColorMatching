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
from global_values import *

####show the 2 images side by side

def display_the_fragments_matching_sides(fragments: List[Fragment], comp: SidesComparison):
    # fragment1 = draw_red_border(fragments[comp.side1.fragment_idx], comp.side1)  
    # fragment2 = draw_red_border(fragments[comp.side2.fragment_idx], comp.side2)  

    fragment1 = rotate_fragment(fragments, comp.side1, 1)
    fragment2 = rotate_fragment(fragments, comp.side2, 2)
    plt.figure(figsize=(4, 3)) 

    plt.subplot(1, 2, 1)  
    plt.imshow(fragment1) 
    plt.title(f"Fragment {comp.side1.fragment_idx}  {comp.side1.side_idx}")

    plt.subplot(1, 2, 2)  
    plt.imshow(fragment2)
    plt.title(f"Fragment {comp.side2.fragment_idx}  {comp.side2.side_idx}")

    plt.show()


def display_new_piece(fragments: List[Fragment], comp: SidesComparison):
    new_fragment = two_fragments_merger(fragments, comp.side1, comp.side2)
    plt.figure(figsize=(4, 4)) 
    plt.imshow(new_fragment)
    plt.show()


###display the sides colors
def display_sides_colors(comp):
    side1_printable_values = comp.reversed_side1_value[np.newaxis, :]
    side2_printable_values = comp.side2.value[np.newaxis, :]
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

#display the sides grads
def display_two_fragments_grads(fragments: List[Fragment], comp: SidesComparison):


    rotated_fragment1 = rotate_fragment(fragments, comp.side1, 1)
    rotated_fragment2 = rotate_fragment(fragments, comp.side2, 2)

    side1_grad_img = grad_func(rotated_fragment1)
    side1_grad_img = np.clip(side1_grad_img * 255, 0, 255).astype(np.uint8)
    side2_grad_img = grad_func(rotated_fragment2)
    side2_grad_img = np.clip(side2_grad_img * 255, 0, 255).astype(np.uint8)

    plt.figure(figsize=(4, 2)) 
    plt.subplot(1, 2, 1)
    plt.imshow(side1_grad_img, cmap="gray", vmin=0, vmax=255)
    
    plt.subplot(1, 2, 2)
    plt.imshow(side2_grad_img, cmap="gray", vmin=0, vmax=255)
    
    plt.show()


def display_sides_grads(comp: SidesComparison):
    side1_printable_grad = comp.reversed_side1_grad[np.newaxis, :]
    side1_printable_grad = np.clip(side1_printable_grad * 255, 0, 255).astype(np.uint8)
    side2_printable_grad = comp.side2.grad[np.newaxis, :]
    side2_printable_grad = np.clip(side2_printable_grad * 255, 0, 255).astype(np.uint8)
    plt.figure(figsize=(6, 1))
    
    plt.subplot(2, 1, 1)
    plt.imshow(side1_printable_grad, cmap="gray", vmin=0, vmax=255)
    plt.title(f" {comp.side1.side_idx} of Fragment {comp.side1.fragment_idx}")
    plt.axis("off")
    
    plt.subplot(2, 1, 2)
    plt.imshow(side2_printable_grad, cmap="gray", vmin=0, vmax=255)
    plt.title(f" {comp.side2.side_idx} of Fragment {comp.side2.fragment_idx}")
    plt.axis("off")
    
    plt.show()



def display_fragments_characteristics(fragments: List[Fragment], sorted_sides_comparisons: List[SidesComparison]):

    for comp in sorted_sides_comparisons:
        # if comp.is_valid_match == False:
        # print(f"score: {comp.score}")
        # print(f"is correct: {comp.is_valid_match}")
        # print(f"color score: {comp.color_score}")
        # print(f"grad presence: {comp.grad_presence}  joint grad match: {comp.merged_grad_score}")
        # plt.imshow(comp.merged_image)
        # plt.show()

        # print(f"DLR: {comp.DLR} DRL: {comp.DRL}")
        display_the_fragments_matching_sides(fragments, comp)

        display_new_piece(fragments, comp)

        display_sides_colors(comp)

        display_two_fragments_grads(fragments, comp)

        display_sides_grads(comp)

        print("-----------------------------------------------------------------------------------")