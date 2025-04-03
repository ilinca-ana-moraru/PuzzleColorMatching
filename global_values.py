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

DIFF_GBR = True
DIFF_GRAY = False

def sigmoid(x):
    x = 2 * abs(x) - 1
    return 1 / (1 + np.exp(-x))


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
    # grad[grad<0.1] = 0
    return grad

def apply_Grad_Gray(image):

    image = image[:, :, :3] 
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_g_blur = cv.GaussianBlur(image, (5,5), 0) 
    image_float = np.float32(image_g_blur)

    dx = cv.Sobel(image_float, cv.CV_64F, 1, 0, ksize=5)
    dy = cv.Sobel(image_float, cv.CV_64F, 0, 1, ksize=5)

    grad = np.sqrt(dx**2 + dy**2)/5120
    grad[grad<0.1] = 0

    return grad


def apply_Grad_On_Joint_Piece(image):
    middle = image.shape[2]//2
    image = image[:, :, :3] 

    sobel_filter = np.array([[-1, -2],
                            [0, 0],
                            [1, 2]])
    V = np.zeros((image.shape[0]-2,3))
    for p in range(image.shape[0]-2):
        channel_response = np.zeros(3)
        for c in range(3):
            patch = image[p:p+3, middle-1:middle+1, c]
            V[p,c] = np.sum(sobel_filter * patch)/255
        
    response = np.sqrt(np.sum(np.mean(V**2, axis = 0)))
    print(response)


