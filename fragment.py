import cv2 as cv
import numpy as np 
from side import *
from global_values import *
import random
from evaluate_sol import *

def rotate_image(image, rotation):
    
    if rotation == 1:
        image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
    elif rotation == 2:
        image = cv.rotate(image, cv.ROTATE_180)
    elif rotation == 3:
        image = cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE)

    return image


class Fragment:
    def __init__(self,value, fragment_idx):
        self.value = value
        self.fragment_idx = fragment_idx
        self.rotation = 0
        # to do: side detection
        # for now, hardcoded
#--------------------------------------------------------------------
        h, w = self.value.shape[:2]
        self.corners = [[0,0],[0,w-1], [h-1,w-1], [h-1,0]]
        self.corners = np.array(self.corners, dtype=int)
#---------------------------------------------------------------------
        binary_mask =  (self.value[:, :, 3])
        # contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        # contour = max(contours, key=cv.contourArea)  
        # contour = contour.reshape(-1, 2)
        # self.contour = contour
        top_edge = np.array([[0, x] for x in range(w-1)])  
        right_edge = np.array([[y, w - 1] for y in range(h-1)]) 
        bottom_edge = np.array([[h - 1, x] for x in range(w - 1, 0, -1)])  
        left_edge = np.array([[y, 0] for y in range(h - 1, 0, -1)]) 
        self.contour = np.concatenate([top_edge, right_edge, bottom_edge, left_edge])
        self.contour = self.contour.astype(int)


        moments = cv.moments(binary_mask)  
        if moments["m00"] != 0:
            self.cx = int(moments["m10"] / moments["m00"])  
            self.cy = int(moments["m01"] / moments["m00"]) 
        else:
            self.cx = 0
            self.cy = 0
        
        self.grad = grad_func(self.value)
        self.create_sides()

    def create_sides(self):
        self.sides = []
        first_corner_idx_in_contour = np.where((self.contour == self.corners[0]).all(axis=1))[0][0]
        for first_corner_idx in range(0,len(self.corners)):
            second_corner_idx = (first_corner_idx+1)%len(self.corners)
            second_corner_idx_in_contour = np.where((self.contour == self.corners[second_corner_idx]).all(axis=1))[0][0]

            if first_corner_idx_in_contour < second_corner_idx_in_contour:
                side_indexes = self.contour[first_corner_idx_in_contour:(second_corner_idx_in_contour+1)%len(self.contour)]
            else:
                side_indexes1 = self.contour[first_corner_idx_in_contour:len(self.contour)]
                side_indexes2 = self.contour[0:second_corner_idx_in_contour+1]
                side_indexes = np.concatenate([side_indexes1, side_indexes2[::-1]]) 
            # side_indexes = np.unique(side_indexes, axis = 0)
            rgb_values = self.value[:,:,:3]
            side_value = np.squeeze(rgb_values[side_indexes[:,0],side_indexes[:,1]]) 
            side_grad = self.grad[side_indexes[:,0],side_indexes[:,1]]
            first_corner_idx_in_contour = second_corner_idx_in_contour
            self.sides.append(Side(side_value, side_grad, side_indexes, first_corner_idx, self.fragment_idx, self.rotation))
    

def divide_image(image_path, output_folder, n, m):
    random.seed(17)

    os.makedirs(output_folder, exist_ok=True)
    rgb_image = cv.imread(image_path, cv.IMREAD_COLOR)  
    rgb_image = rgb_image[..., ::-1]
    rgba_image = cv.cvtColor(rgb_image, cv.COLOR_BGR2BGRA)
    h, w = rgba_image.shape[:2] 

    tile_h, tile_w = h // n, w // m  
    gt_grid = [[0 for _ in range(n)] for _ in range(m)]

    fragments = []
    rotations = np.zeros((n*m))
    for i in range(n):
        for j in range(m):
            x, y = j * tile_w, i * tile_h  
            cropped_fragment = rgba_image[y:y + tile_h, x:x + tile_w]  
            fr_idx = i * m + j

            if ROTATING_PIECES:
                rotation = int(random.randint(0, 3)) 
                rotations[fr_idx] = rotation
            else:
                rotation = 0
            gt_grid[i][j] = fr_idx
            rotated_fragment = rotate_image(cropped_fragment, rotation)
            fragment_path = os.path.join(output_folder, f"fragment_{fr_idx}.jpg")
            cv.imwrite(fragment_path, rotated_fragment[..., [2, 1, 0, 3]])
            fr = Fragment(rotated_fragment, i*m + j)
            fragments.append(fr)
    
    write_sol_comp(gt_grid, rotations)

    print("Rotations by fragment:", ', '.join([f"{idx}){rotation}" for idx, rotation in enumerate(rotations)]))
    return fragments, tile_h, tile_w


