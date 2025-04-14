import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import List
np.set_printoptions(threshold=np.inf)  
from utils import *


COL_NR = 16
ROW_NR = 16


GRAD_GBR = True
GRAD_GRAY = False

DIFF_GBR = True
DIFF_GRAY = False



def grad_func(image):
    if GRAD_GBR == True:
        return apply_Grad_GBR(image)
    if GRAD_GRAY == True:
        return apply_Grad_Gray(image)

