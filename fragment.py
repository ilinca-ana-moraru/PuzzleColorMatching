import cv2 as cv
import numpy as np 

class Fragment:
    def __init__(self,value, fragment_idx):
        self.value = value
        self.fragment_idx = fragment_idx

        # to do: side detection
        # for now, hardcoded
#--------------------------------------------------------------------
        h, w = self.value.shape[:2]
        self.corners = [[0,0],[0,h-1], [w-1,h-1], [w-1,0]]
#---------------------------------------------------------------------
        binary_mask =  (self.value[:, :, 3])
        contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv.contourArea)  
        contour = contour.reshape(-1, 2)
        self.contour = contour