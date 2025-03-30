from side import * 
import numpy as np
from global_values import *

class SidesComparison:
    def __init__(self,side1 : Side, side2: Side):
        self.side1 = side1
        self.side2 = side2
        self.reversed_side1_value = side1.value[::-1]
        color_points_distances = self.reversed_side1_value - side2.value
        self.color_score = np.sqrt(np.sum(color_points_distances**2))/len(self.side1.value)
        

        self.reversed_side1_grad = side1.grad[::-1]
        self.grad_match = np.sqrt(np.sum((self.reversed_side1_grad - side2.grad)**2))
        
        self.grad_presence = np.sum(self.reversed_side1_grad + side2.grad)
        self.grad_score = 50 * self.grad_match/ (self.grad_presence + 0.000001)
        self.score =  np.sqrt(self.grad_score**2  + self.color_score**2)
    
        self.is_valid_match = False

        if self.side1.fragment_idx == self.side2.fragment_idx - 1 and self.side1.side_idx == 1 and self.side2.side_idx == 3:
            self.is_valid_match = True


        elif self.side1.fragment_idx - 1 == self.side2.fragment_idx and self.side1.side_idx == 3 and self.side2.side_idx == 1:
            self.is_valid_match = True

        elif self.side1.fragment_idx == self.side2.fragment_idx - ROW_NR and self.side1.side_idx == 2 and self.side2.side_idx == 0:
            self.is_valid_match = True


        if self.side1.fragment_idx - ROW_NR == self.side2.fragment_idx and self.side1.side_idx == 0 and self.side2.side_idx == 1:
            self.is_valid_match = True

