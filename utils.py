import os
import cv2 as cv
import matplotlib.pyplot as plt
from fragment import *
import numpy as np

def sigmoid(x):
    x = 2 * abs(x) - 1
    return 1 / (1 + np.exp(-x))
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


def two_fragments_edge_merger(side1, side2):

    revered_edge1 = side1.value[::-1]
    edge1 = revered_edge1[:, np.newaxis, :]
    edge2 = side2.value[:, np.newaxis, :]

    return np.hstack((edge1, edge2))

def apply_Grad_On_Joint_Piece(image):
    image = image[:, :, :3] 
    
    sobel_filter = np.array([[-1, -2], [0, 0], [1, 2]]) 
    h, w, _ = image.shape
    image = image.astype(np.float32)

    patches = np.stack([image[i:i+3, :, :] for i in range(h - 2)], axis=0) 
    filtered = np.tensordot(patches, sobel_filter, axes=([1], [0])) 
    
    mean_sq = np.mean(filtered**2, axis=(0, 1)) 
    response = np.sqrt(np.sum(mean_sq))
    
    return response


    
def extract_edge_columns(image, start_coord, end_coord, thickness):
    row1, col1 = start_coord
    row2, col2 = end_coord
    h, w = image.shape[:2]

    if row1 == row2:
        row = row1
        cols = np.arange(min(col1, col2), max(col1, col2) + 1)
        if row == 0:
            rows = np.arange(0, min(row + thickness, h))
        else:
            rows = np.arange(max(row - thickness + 1, 0), row + 1)

        strip = image[rows[:, None], cols, :]  
        strip = np.transpose(strip, (1, 0, 2)) 

    elif col1 == col2:
        col = col1
        rows = np.arange(min(row1, row2), max(row1, row2) + 1)

        if col == 0:
            cols = np.arange(0, min(col + thickness, w))
        else:
            cols = np.arange(max(col - thickness + 1, 0), col + 1)

        strip = image[rows[:, None], cols, :] 

    return strip[:,:,:3] 



def mahalanobis_merger(comp, fragments):

    xi = extract_edge_columns(fragments[comp.side1.fragment_idx].value, comp.side1.side_indexes_of_fragment[-1], comp.side1.side_indexes_of_fragment[0], 2)
    xj = extract_edge_columns(fragments[comp.side2.fragment_idx].value, comp.side2.side_indexes_of_fragment[0], comp.side2.side_indexes_of_fragment[-1], 2)

    GiL = xi[:,1,:] - xi[:,0,:]
    GijLR = xj[:,0,:] - xi[:,1,:]
    
    muiL = np.mean(GiL, axis = 0)
    S = np.cov(GiL.T)
    if np.linalg.matrix_rank(S) < S.shape[0]:
        S += np.eye(S.shape[0]) * 1e-5
    S_inv = np.linalg.inv(S)
    
    # print(f"S_inv: {S_inv}")

    DLR_arr = np.zeros(GijLR.shape[0])
    for p in range(GijLR.shape[0]):
        DLR_arr[p] = (GijLR[p] - muiL) @ S_inv @ (GijLR[p] - muiL).T

    # print(f"arr {DLR_arr}")
    DLR = np.sqrt(np.sum(DLR_arr))


    GjL = xj[:, 1, :] - xj[:, 0, :]
    GjiRL = xi[:, 0, :] - xj[:, 1, :]

    mujL = np.mean(GjL, axis=0)
    S = np.cov(GjL.T)
    if np.linalg.matrix_rank(S) < S.shape[0]:
        S += np.eye(S.shape[0]) * 1e-5
    S_inv = np.linalg.inv(S)

    DRL_arr = np.zeros(GjiRL.shape[0])
    for p in range(GjiRL.shape[0]):
        diff = GjiRL[p] - mujL
        DRL_arr[p] = diff @ S_inv @ diff.T

    DRL = np.sqrt(np.sum(DRL_arr))

    # print(f"DLR {DLR} DRL {DRL}")

    return DLR, DRL

def find_centroid(image):
    binary_mask =  (image[:, :, 3])
    moments = cv.moments(binary_mask)  
    if moments["m00"] != 0:
        cx = int(moments["m10"] // moments["m00"])  
        cy = int(moments["m01"] // moments["m00"]) 
    else:
        cx = 0
        cy = 0
    return cx, cy



def fix_border(image):

    width, height = image.shape[:2]

    for x in range(1, width - 1):
        image[x, 0] = image[x, 1]  
    
    for x in range(1, width - 1):
        image[x, height - 1] = image[x, height - 2]  

    for y in range(1, height - 1):
        image[0, y] = image[1, y] 
    
    for y in range(1, height - 1):
        image[width - 1, y] = image[width - 2, y] 

    return image




