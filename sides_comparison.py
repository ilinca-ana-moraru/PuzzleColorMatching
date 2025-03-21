from side import * 
import numpy as np

class SidesComparison:
    def __init__(self,side1 : Side, side2: Side):
        self.side1 = side1
        self.side2 = side2
        self.reversed_side1_value = side1.value[::-1]
        # self.points_distances = np.linalg.norm(self.reversed_side1_value - side2.value, axis = 1)
        # self.score = sum(self.points_distances)/len(self.side1.value)
        self.points_distances = self.reversed_side1_value - side2.value
        self.score = np.sqrt(np.sum(self.points_distances**2))/len(self.side1.value)


