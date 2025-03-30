import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import List
np.set_printoptions(threshold=np.inf)  



COL_NR = 8
ROW_NR = 8


GRAD_GBR = True
GRAD_GRAY = False



def grad_func(image):
    if GRAD_GBR == True:
        return apply_Grad_GBR(image)
    if GRAD_GRAY == True:
        return apply_Grad_Gray(image)
    


def grad_func(image):
    if GRAD_GBR == True:
        return apply_Grad_GBR(image)
    if GRAD_GRAY == True:
        return apply_Grad_Gray(image)

def apply_Grad_GBR(image):

    image = image[:, :, :3] 
    image_g_blur = cv.GaussianBlur(image, (5,5), 0) 
    image_float = np.float32(image_g_blur)

    dx = cv.Sobel(image_float, cv.CV_64F, 1, 0, ksize=5)
    dx = np.sqrt(dx[:,:,0]**2 + dx[:,:,1]**2 + dx[:,:,2]**2)
    dy = cv.Sobel(image_float, cv.CV_64F, 0, 1, ksize=5)
    dy = np.sqrt(dy[:,:,0]**2 + dy[:,:,1]**2 + dy[:,:,2]**2)

    grad = np.sqrt(dx**2 + dy**2)/10000

    return grad

def apply_Grad_Gray(image):

    image = image[:, :, :3] 
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_g_blur = cv.GaussianBlur(image, (5,5), 0) 
    image_float = np.float32(image_g_blur)

    dx = cv.Sobel(image_float, cv.CV_64F, 1, 0, ksize=5)
    dy = cv.Sobel(image_float, cv.CV_64F, 0, 1, ksize=5)

    grad = np.sqrt(dx**2 + dy**2)/5120
    # grad = normalized_sigmoid(grad)
    # print(f"min {np.min(grad)} max:{np.max(grad)}")

    return grad