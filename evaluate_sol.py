import numpy as np
import os
from rotation import *








def write_sol_comp(gt_grid, rotations,solution_path):
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

    os.makedirs(os.path.dirname(solution_path), exist_ok=True)
    with open(solution_path, "w") as f:
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



def solution_distace(solution_grid, gt_grid):
    best_score = None

    positions = []
    for i in range(len(solution_grid)):
        for j in range(len(solution_grid[0])):
            fr_idx = solution_grid[i][j]
            if fr_idx is not None:
                positions.append((i, j))


    min_i = min(pos[0] for pos in positions)
    min_j = min(pos[1] for pos in positions)
    current_positions = {}
    for i in range(len(solution_grid)):
        for j in range(len(solution_grid[0])):
            fr_idx = solution_grid[i][j]
            norm_i = i - min_i
            norm_j = j - min_j
            current_positions[fr_idx] = (norm_i, norm_j)

    for rotation in range(4): 
        gt_grid_rotated = np.rot90(np.array(gt_grid), k=rotation).tolist()

        total_distance = 0.0

        for i_gt in range(len(gt_grid_rotated)):
            for j_gt in range(len(gt_grid_rotated[0])):
                fr_idx = gt_grid_rotated[i_gt][j_gt]
                if fr_idx in current_positions:
                    i_curr, j_curr = current_positions[fr_idx]
                    distance = np.sqrt((i_curr - i_gt) ** 2 + (j_curr - j_gt) ** 2)
                    total_distance += distance


        if  best_score is None or total_distance < best_score:
            best_score = total_distance

    return best_score


