from side import * 
import numpy as np
from global_values import *
from scipy.special import erf

class SidesComparison:
    def __init__(self, fragments, side1 : Side, side2: Side):
        self.side1 = side1
        self.side2 = side2
        self.merged_image = two_fragments_merger(fragments, self.side1, self.side2)

        
        self.reversed_side1_value = side1.value[::-1]
        self.color_points_distances = abs(self.reversed_side1_value - side2.value)
        color_score = self.color_points_distances/ 255

        if DIFF_GBR == True:
            grayscale_weights = 3 * np.array([0.2989, 0.5870, 0.1140])
            color_score *= grayscale_weights
            color_score =np.linalg.norm(color_score, axis = 1)

        if DIFF_GRAY == True:
            color_score =np.linalg.norm(color_score, axis = 1)


        self.color_score = erf(4 * color_score - 2)/2 + 0.5 ## input[0,1] -> [-2, 2] output[-1,1] -> [0,1]
        self.color_score = np.sum(color_score)/len(self.side1.value)



        self.reversed_side1_grad = side1.grad[::-1]
        grad_match = (self.reversed_side1_grad - self.side2.grad)
        grad_match = erf(4 * grad_match - 2)/2 + 0.5  ## input[0,1] -> [-2, 2] output[-1,1] -> [0,1]
        grad_match[grad_match < 0.3] = 0.0
        self.grad_match = np.sum(grad_match)/len(self.side1.value)

        self.grad_presence = np.sum(erf(4 * self.reversed_side1_grad - 2)/2 + 0.5 + erf(4 * side2.grad - 2)/2 + 0.5)
        self.grad_score = self.grad_match/ (self.grad_presence + 0.000001)
        # self.score = np.sqrt((self.grad_score)**2 + (self.color_score)**2)
        self.score = 1/(self.grad_presence + 0.000001) * np.sqrt(( self.grad_match)**2  + self.color_score**2)

        for i in color_score:
            if i > 0.3:
                self.score *= 3
        for i in grad_match:
            if i > 0.3:
                self.score *= 3

        # if self.color_score < 5 and self.grad_match < 3 and self.grad_presence > 1:
        #     self.score = 0


        self.is_valid_match = False
        if self.side1.fragment_idx == self.side2.fragment_idx - 1 and self.side1.side_idx == 1 and self.side2.side_idx == 3:
            self.is_valid_match = True


        elif self.side1.fragment_idx - 1 == self.side2.fragment_idx and self.side1.side_idx == 3 and self.side2.side_idx == 1:
            self.is_valid_match = True

        elif self.side1.fragment_idx == self.side2.fragment_idx - ROW_NR and self.side1.side_idx == 2 and self.side2.side_idx == 0:
            self.is_valid_match = True


        if self.side1.fragment_idx - ROW_NR == self.side2.fragment_idx and self.side1.side_idx == 0 and self.side2.side_idx == 1:
            self.is_valid_match = True


def rotate_fragment(fragments, side, side_type):
    image = fragments[side.fragment_idx].value
   
    h, w = image.shape[:2]

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
  

    if abs(rotation_angle) < 5:
        return image
    
    elif rotation_angle > 80 and rotation_angle < 100 or rotation_angle < -260 and rotation_angle > -280:
        image = cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE)

    elif abs(rotation_angle) > 170 and abs(rotation_angle) < 190 or rotation_angle < -170 and rotation_angle > -190:
        image = cv.rotate(image, cv.ROTATE_180)
    
    elif rotation_angle < -80 and rotation_angle > -100  or rotation_angle > 260 and rotation_angle < 280:
        image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
    else:
        print(f"invalid rotation angle {rotation_angle}")
        return 0
  
    return image



def two_fragments_merger(fragments, side1, side2):
    rotated_fragment1 = rotate_fragment(fragments, side1, 1)
    rotated_fragment2 = rotate_fragment(fragments, side2, 2)


    new_fragment = np.hstack((rotated_fragment1, rotated_fragment2))
    return new_fragment

