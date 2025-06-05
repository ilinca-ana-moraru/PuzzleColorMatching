import global_values 
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from typing import List
from sides_comparison import SidesComparison
from rotation import *
import pandas as pd
import cv2 as cv
import copy

def get_comparison(frag1, frag2, side1, side2):
    comp = global_values.SYMMETRIC_COMPARISONS[frag1][frag2][side1][side2]
    
    # if comp is None:
    #     comp = global_values.SYMMETRIC_COMPARISONS[frag2][frag1][side2][side1]  
    # if comp is None:
        # print(f"Comparison missing for f1={frag1}, f2={frag2}, s1={side1}, s2={side2}")

    return comp





class Group:

    def __init__(self, fragment_idx):
        self.used_fragments = []
        self.fragment_positions = {}
        self.col_nr = 3
        self.row_nr = 3
        self.grid = [[None for _ in range(self.col_nr)] for _ in range(self.row_nr)]
        self.neighbours_grid = [[0 for _ in range(self.col_nr)] for _ in range(self.row_nr)]

        self.grid[1][1] = fragment_idx
        self.used_fragments.append(fragment_idx)
        self.fragment_positions[fragment_idx] = [1,1]
        self.update_neighbours_grid_after_new_merge(1,1)

    def __str__(self):
        return(f"fragment indexes used: {self.used_fragments} at positions: {self.fragment_positions}")
    


    def update_neighbours_grid_after_new_merge(self, i, j):
        if i == 0 or j == 0 or i == self.row_nr-1 or j == self.col_nr -1:
            # print("no empty edge in merging")
            return
        self.neighbours_grid[i][j] = 0
        if self.grid[i-1][j] == None:
            self.neighbours_grid[i-1][j] += 1
        if self.grid[i+1][j] == None:
            self.neighbours_grid[i+1][j] += 1
        if self.grid[i][j-1] == None:
            self.neighbours_grid[i][j-1] += 1
        if self.grid[i][j+1] == None:
            self.neighbours_grid[i][j+1] += 1
            

    def show_group(self, fragments, extra_rotation):

        if self.row_nr <= 2 or self.col_nr <= 2:
            height = 100
            width = 100
            canvas_img = np.ones((height, width, 3), dtype=np.uint8) * 255

            fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
            ax.imshow(canvas_img)
            ax.axis("off")

            canvas = FigureCanvas(fig)
            canvas.draw()
            img_array = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            img_array = img_array.reshape((int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), 3))

            plt.close(fig)

            return img_array

        cropped_rows = self.row_nr - 2
        cropped_cols = self.col_nr - 2
        height = cropped_rows * global_values.TILE_H
        width = cropped_cols * global_values.TILE_W

        canvas_img = np.ones((height, width, 3), dtype=np.uint8) * 255

        for i in range(1, self.row_nr - 1): 
            for j in range(1, self.col_nr - 1):
                cell = self.grid[i][j]
                if cell is not None:
                    fragment = fragments[cell]
                    img = fragment.value[:, :, :3]
                    h, w = img.shape[:2]

                    if (h, w) != (global_values.TILE_H, global_values.TILE_W):
                        img_resized = resize(img, (global_values.TILE_H, global_values.TILE_W), preserve_range=True, anti_aliasing=True).astype(np.uint8)
                    else:
                        img_resized = img.astype(np.uint8)

                    img_rotated = rotate_image(img_resized, (fragment.rotation + extra_rotation) % 4)

                    top = (i - 1) * global_values.TILE_H
                    left = (j - 1) * global_values.TILE_W

                    canvas_img[top:top + global_values.TILE_H, left:left + global_values.TILE_W] = img_rotated

        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        ax.imshow(canvas_img)
        ax.axis("off")

    
        grid_thickness = 2
        grid_color = 'black'

        for i in range(0, cropped_rows + 1):
            y = i * global_values.TILE_H
            ax.plot([0, width], [y, y], color=grid_color, linewidth=grid_thickness)

        for j in range(0, cropped_cols + 1):
            x = j * global_values.TILE_W
            ax.plot([x, x], [0, height], color=grid_color, linewidth=grid_thickness)
        # neighbours
        # for i in range(1, self.row_nr - 1):
        #     for j in range(1, self.col_nr - 1):
        #         count = self.neighbours_grid[i][j]
        #         if count > 0:
        #             x = (j - 1) * global_values.TILE_W + global_values.TILE_W // 2
        #             y = (i - 1) * global_values.TILE_H + global_values.TILE_H // 2
        #             ax.text(x, y, str(count), color='red', ha='center', va='center', fontsize=18, weight='bold')

        canvas = FigureCanvas(fig)
        canvas.draw()
        img_array = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape((int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), 3))

        plt.close(fig)

        return img_array




def simulate_merge_positions(fragments, comp: SidesComparison, anchor_group: Group, pasted_group: Group):
    anchor_copy = copy.deepcopy(anchor_group)
    pasted_copy = copy.deepcopy(pasted_group)

    anchor_side = comp.side1
    pasted_side = comp.side2

    #### moved pasted group in .fragment_positions to line up with 

    offset_row, offset_col, pasted_group_additional_rotation = find_pasted_group_moving_distance_and_rotation(fragments, comp)

    pasted_copy = rotate_fragments_positions(pasted_copy, pasted_group_additional_rotation)

    anchor_row, anchor_col = anchor_copy.fragment_positions[anchor_side.fragment_idx]
    pasted_row, pasted_col = pasted_copy.fragment_positions[pasted_side.fragment_idx]
    row_offset = anchor_row + offset_row - pasted_row
    col_offset = anchor_col + offset_col - pasted_col

    for fr_idx in pasted_copy.used_fragments:
        row, col = pasted_copy.fragment_positions[fr_idx]
        pasted_copy.fragment_positions[fr_idx] = [row + row_offset, col + col_offset]

    #### moving anchor_copy.fragment_positions and pasted_copy.fragment_positions to create a correct grid
    all_rows = [row for row, col in anchor_copy.fragment_positions.values()] + \
               [row for row, col in pasted_copy.fragment_positions.values()]
    all_cols = [col for row, col in anchor_copy.fragment_positions.values()] + \
               [col for row, col in pasted_copy.fragment_positions.values()]

    min_row = min(all_rows)
    min_col = min(all_cols)
    max_row = max(all_rows)
    max_col = max(all_cols)

    anchor_shift_r = 1 - min_row
    anchor_shift_c = 1 - min_col

    for fr_idx in anchor_copy.fragment_positions:
        r, c = anchor_copy.fragment_positions[fr_idx]
        anchor_copy.fragment_positions[fr_idx] = [r + anchor_shift_r, c + anchor_shift_c]

    for fr_idx in pasted_copy.fragment_positions:
        r, c = pasted_copy.fragment_positions[fr_idx]
        pasted_copy.fragment_positions[fr_idx] = [r + anchor_shift_r, c + anchor_shift_c]

    ### finding the size of new grid
    all_rows = [row for row, col in anchor_copy.fragment_positions.values()] + \
               [row for row, col in pasted_copy.fragment_positions.values()]
    all_cols = [col for row, col in anchor_copy.fragment_positions.values()] + \
               [col for row, col in pasted_copy.fragment_positions.values()]

    new_row_nr = max(all_rows) + 2
    new_col_nr = max(all_cols) + 2

    anchor_copy.row_nr = new_row_nr
    anchor_copy.col_nr = new_col_nr
    pasted_copy.row_nr = new_row_nr
    pasted_copy.col_nr = new_col_nr

    ## populating grids final form
    anchor_copy.grid = [[None for _ in range(anchor_copy.col_nr)] for _ in range(anchor_copy.row_nr)]
    anchor_copy.neighbours_grid = [[0 for _ in range(anchor_copy.col_nr)] for _ in range(anchor_copy.row_nr)]

    pasted_copy.grid = [[None for _ in range(pasted_copy.col_nr)] for _ in range(pasted_copy.row_nr)]
    pasted_copy.neighbours_grid = [[0 for _ in range(pasted_copy.col_nr)] for _ in range(pasted_copy.row_nr)]

    for fr_idx in anchor_copy.fragment_positions:
        row, col = anchor_copy.fragment_positions[fr_idx]
        anchor_copy.grid[row][col] = fr_idx

    for fr_idx in pasted_copy.fragment_positions:
        row, col = pasted_copy.fragment_positions[fr_idx]
        pasted_copy.grid[row][col] = fr_idx


    # anchor_img = anchor_copy.show_group(fragments,0)
    # pasted_img = pasted_copy.show_group(fragments, pasted_group_additional_rotation)
    # plt.imshow(anchor_img)
    # plt.show()
    # plt.imshow(pasted_img)
    # plt.show()


    return anchor_copy, pasted_copy, pasted_group_additional_rotation



def check_groups_shapes_for_merging(shifted_anchor_group: Group, shifted_pasted_group: Group):

    for fr_idx in shifted_pasted_group.used_fragments:
        row, col = shifted_pasted_group.fragment_positions[fr_idx]

        if shifted_anchor_group.grid[row][col] is not None:
            # print("impossible merging: incompatible group shapes")
            return False
    return True
    
    

def does_merge_fit_within_bounds(shifted_anchor_group: Group):

    if shifted_anchor_group.row_nr - 2 > global_values.ROW_NR:
        # print(f"Merge would exceed puzzle size")
        return False
    if shifted_anchor_group.col_nr - 2  > global_values.COL_NR:  
        # print(f"Merge would exceed puzzle size")
        return False
    return True



def check_all_group_matchings_scores(one_image_condition, mean_condition, fragments, pasted_group_additional_rotation, shifted_anchor_group: Group, shifted_pasted_group: Group, one_match_th, group_th):
    total_score = 0.0
    total_matchings = 0

    directions = [(-1, 0, 0, 2), (1, 0, 2, 0), (0, -1, 3, 1), (0, 1, 1, 3)]

    for pasted_fr_idx in shifted_pasted_group.used_fragments:
        row, col = shifted_pasted_group.fragment_positions[pasted_fr_idx]

        for neighbour_row_offset, neighbour_col_offset, s1, s2 in directions:
            neighbor_row = row + neighbour_row_offset
            neighbor_col = col + neighbour_col_offset
            anchor_fr_idx = shifted_anchor_group.grid[neighbor_row][neighbor_col]
            if anchor_fr_idx is not None:
                pasted_fragment_rotation = (pasted_group_additional_rotation + fragments[pasted_fr_idx].rotation) % 4
                side1 = find_side_idx_of_orientation(pasted_fragment_rotation, s1)
                side2 = find_side_idx_of_orientation(fragments[anchor_fr_idx].rotation, s2)
                neighbor_comp = get_comparison(pasted_fr_idx, anchor_fr_idx, side1, side2)
                if neighbor_comp:
                    # print(neighbor_comp)
                    # if one_image_condition(neighbor_comp, one_match_th) == False:
                    #     # print("a score too bad")
                    #     return False
                    total_score += neighbor_comp.score
                    total_matchings += 1

    if total_matchings == 0:
        # print("no matchings")

        return False

    average_score = total_score / total_matchings
    if not mean_condition(average_score, group_th):
        # print("total score bad")
        return False
    
    return True


def update_after_merge(groups: List[Group],fragments, fragment_idx_to_group_idx, pasted_group_idx):
    for fr_idx in range(len(fragments)):
        if fragment_idx_to_group_idx[fr_idx] > pasted_group_idx:
            fragment_idx_to_group_idx[fr_idx] -= 1
    
    del groups[pasted_group_idx]

    

def merge_groups(fragments, pasted_group_additional_rotation, shifted_anchor_group: Group, shifted_pasted_group: Group, fragment_idx_to_group_idx):

    ### the final merged group is in shifted_anchor_group

    for fr_idx, pos in shifted_pasted_group.fragment_positions.items():
        ### add pasted group fragments idx in fragment positions
        shifted_anchor_group.fragment_positions[fr_idx] = pos
        ### update the rotation of the individual rotation of each fragment relative to initial state
        fragments[fr_idx].rotation = (fragments[fr_idx].rotation + pasted_group_additional_rotation) % 4

    ### update 
    shifted_anchor_group.used_fragments.extend(shifted_pasted_group.used_fragments)

    for fr_idx in shifted_pasted_group.fragment_positions:
        r, c = shifted_pasted_group.fragment_positions[fr_idx]
        
        shifted_anchor_group.grid[r][c] = fr_idx
        fragment_idx_to_group_idx[fr_idx] = fragment_idx_to_group_idx[shifted_anchor_group.used_fragments[0]]

    for fr_idx in shifted_anchor_group.used_fragments:
        row, col = shifted_anchor_group.fragment_positions[fr_idx]
        shifted_anchor_group.update_neighbours_grid_after_new_merge(row, col)

    return shifted_anchor_group

def show_all_groups(groups, fragments, fr_idx_to_group_idx, dont_show_1_fr_group, max_cols=8):
    images = []
    group_indices = []

    for gr in groups:
        if dont_show_1_fr_group == 0 or len(gr.used_fragments) > 1:
            image = gr.show_group(fragments,0)
            images.append(image)
            gr_idx = fr_idx_to_group_idx[gr.used_fragments[0]]
            group_indices.append(gr_idx)

    n = len(images)
    if n == 0:
        return
    n_cols = min(n, max_cols)
    n_rows = (n + max_cols - 1) // max_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(images[i])
            ax.set_title(f"Group {group_indices[i]}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()



