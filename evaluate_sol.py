import numpy as np
import os
from rotation import *

def write_sol_comp(gt_grid, rotations):
    gt_grid = np.array(gt_grid)

    n, m = gt_grid.shape
    comparisons = []

    for i in range(n):
        for j in range(m):
            fr1_idx = gt_grid[i][j]
            if j + 1 < m:
                side1_idx = find_side_idx_of_orientation((-1)*rotations[fr1_idx], 1)
                fr2_idx = gt_grid[i][j + 1]
                side2_idx = find_side_idx_of_orientation((-1)*rotations[fr2_idx],3)
                comparisons.append((fr1_idx, fr2_idx, side1_idx, side2_idx))

            if i + 1 < n:
                side1_idx = find_side_idx_of_orientation((-1)*rotations[fr1_idx], 2)
                fr2_idx = gt_grid[i + 1][j]
                side2_idx = find_side_idx_of_orientation((-1)*rotations[fr2_idx], 0)
                comparisons.append((fr1_idx, fr2_idx, side1_idx, side2_idx))

    output_path = os.path.join("solution", "valid_comparisons.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for fr1, fr2, s1, s2 in comparisons:
            f.write(f"{int(fr1)},{int(fr2)},{int(s1)},{int(s2)}\n")




def read_valid_comparisons(filepath):
    comparisons = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue 
            parts = list(map(int, line.split(",")))
            if len(parts) == 4:
                comparisons.append(tuple(parts))
    return comparisons


