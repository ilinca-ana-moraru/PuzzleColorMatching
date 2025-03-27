import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import List
np.set_printoptions(threshold=np.inf)  

from global_values import *



COL_NR = 8
ROW_NR = 8

def apply_Grad(image):
    # print("initial img:")
    # plt.imshow(image)
    # plt.show()
    image = image[:, :, :3] 
    image_g_blur = cv.GaussianBlur(image, (5,5), 0) 
    # plt.imshow(image_g_blur)
    # plt.show()
    image_float = np.float32(image_g_blur)

    dx = cv.Sobel(image_float, cv.CV_64F, 1, 0, ksize=5)
    dx = np.sqrt(dx[:,:,0]**2 + dx[:,:,1]**2 + dx[:,:,2]**2)
    dy = cv.Sobel(image_float, cv.CV_64F, 0, 1, ksize=5)
    dy = np.sqrt(dy[:,:,0]**2 + dy[:,:,1]**2 + dy[:,:,2]**2)

    grad = np.sqrt(dx**2 + dy**2)/10000
    # print(f"min pixel: {np.min(grad)} max pixel: {np.max(grad)}")

    # plt.imshow(grad, cmap = "gray")
    # plt.show()
    return grad



def apply_Grad(image):
    # print("initial img:")
    # plt.imshow(image)
    # plt.show()
    image = image[:, :, :3] 
    image_g_blur = cv.GaussianBlur(image, (5,5), 0) 
    # plt.imshow(image_g_blur)
    # plt.show()
    image_float = np.float32(image_g_blur)

    dx = cv.Sobel(image_float, cv.CV_64F, 1, 0, ksize=5)
    dx = np.sqrt(dx[:,:,0]**2 + dx[:,:,1]**2 + dx[:,:,2]**2)
    dy = cv.Sobel(image_float, cv.CV_64F, 0, 1, ksize=5)
    dy = np.sqrt(dy[:,:,0]**2 + dy[:,:,1]**2 + dy[:,:,2]**2)

    grad = np.sqrt(dx**2 + dy**2)/10000
    # grad = np.uint8(grad / 10000)

    
    # edges = cv.Canny(grad, 200, 400)
    
    # print(f"min pixel: {np.min(grad)} max pixel: {np.max(grad)}")

    # plt.imshow(grad, cmap = "gray")
    # plt.show()
    return grad