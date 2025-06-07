import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import List
np.set_printoptions(threshold=np.inf)  
from utils import *
from typing import List

ROTATING_PIECES = True

GRAD_SCORING = False
NN_SCORING = True
MODEL = None
DEVICE = None

COL_NR = 32
ROW_NR = 32

IMAGE_TH = 0.2
GROUP_TH = 0.08
PRUDENT_X_BUDDY_TH_IMAGE = 0.70


TILE_W = None
TILE_H = None

GRAD_GBR = True
GRAD_GRAY = False

DIFF_GBR = True
DIFF_GRAY = False

SYMMETRIC_COMPARISONS = []


def grad_func(image):
    if GRAD_GBR == True:
        return apply_Grad_GBR(image)
    if GRAD_GRAY == True:
        return apply_Grad_Gray(image)
    
##bun pt pana la 14 x 14
def clasic_score_one_image_passing(side_comparison, th):
    if side_comparison.score < th:
        return True
    return False

def prudent_x_buddy_one_image_passing(side_comparison, th):
    if side_comparison.prudent_score * side_comparison.buddy_score < th:
        return True
    return False

## mai indicat pt astea mai mari
def safe_clasic_score_one_image_passing(side_comparison, th):
    if side_comparison.score < th:
        return True
    return False

def mean_clasic_score(mean, th):
    if mean < th:
        return True
    return False

def no_mean_score(mean, th):
    return True

def no_image_condition(sides_comparison, th):
    return True


####th iterative alg: IMAGE_TH = 0.2 GROUP_TH = 0.08
#### th siamese global_values.IMAGE_TH = 0.01 global_values.GROUP_TH = 0.0025
