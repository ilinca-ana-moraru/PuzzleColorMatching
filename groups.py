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
    if comp is None:
        comp = global_values.SYMMETRIC_COMPARISONS[frag2][frag1][side2][side1]  
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
            

    def show_group(self, fragments):

        height = self.row_nr * global_values.TILE_H
        width = self.col_nr * global_values.TILE_W
        canvas_img = np.zeros((height, width, 3), dtype=np.uint8)

        for i in range(self.row_nr):
            for j in range(self.col_nr):
                cell = self.grid[i][j]
                if cell is not None:
                    fragment = fragments[cell]
                    img = fragment.value[:, :, :3]
                    h, w = img.shape[:2]

                    if (h, w) != (global_values.TILE_H, global_values.TILE_W):
                        img_resized = resize(img, (global_values.TILE_H, global_values.TILE_W), preserve_range=True, anti_aliasing=True)
                    else:
                        img_resized = img

                    img_rotated = rotate_image(img_resized, fragment.rotation)                 
                    top = i * global_values.TILE_H
                    left = j * global_values.TILE_W
                    canvas_img[top:top+global_values.TILE_H, left:left+global_values.TILE_W] = img_rotated


        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        ax.imshow(canvas_img)
        ax.axis("off")

        for i in range(self.row_nr):
            for j in range(self.col_nr):
                count = self.neighbours_grid[i][j]
                if count > 0:
                    x = j * global_values.TILE_W + global_values.TILE_W // 2
                    y = i * global_values.TILE_H + global_values.TILE_H // 2
                    ax.text(x, y, str(count), color='red', ha='center', va='center', fontsize=18, weight='bold')

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
    # offset_row, offset_col, pasted_group_additional_rotation = find_pasted_group_moving_distance_and_rotation(anchor_side, pasted_side)

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

    ## polulating grids correctly
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

def find_side_idx_of_orientation(current_rotation, orientation):
    return (4 + orientation - current_rotation) % 4


def check_all_group_matchings_scores(fragments, pasted_group_additional_rotation, shifted_anchor_group: Group, shifted_pasted_group: Group):
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
                print(f"anchor side of orientation {s2} with fragment rotated  {fragments[anchor_fr_idx].rotation} times  has index {side2}")
                print(f"pasted side or orientation {s1} with fragment rotated {pasted_fragment_rotation} has index {side1}")
                neighbor_comp = get_comparison(pasted_fr_idx, anchor_fr_idx, side1, side2)
                if neighbor_comp:
                    # print(neighbor_comp)

                    if neighbor_comp.score > global_values.IMAGE_TH:
                        # print("a score too bad")
                        return False
                    total_score += neighbor_comp.score
                    total_matchings += 1

    if total_matchings == 0:
        # print("no matchings")

        return False

    average_score = total_score / total_matchings
    if average_score > global_values.GROUP_TH:
        # print("total score bad")
        return False
    
    return True

def calculate_all_group_matchings_scores(shifted_anchor_group: Group, shifted_pasted_group: Group):

    total_score = 0.0
    total_matchings = 0
    directions = [(-1, 0, 0, 2), (1, 0, 2, 0), (0, -1, 3, 1), (0, 1, 1, 3)]

    for pasted_fr_idx in shifted_pasted_group.used_fragments:
        row, col = shifted_pasted_group.fragment_positions[pasted_fr_idx]

        for neighbour_row_offset, neighbour_col_offset, side1, side2 in directions:
            neighbor_row = row + neighbour_row_offset
            neighbor_col = col + neighbour_col_offset
            if neighbor_row >= 0 and  neighbor_row <= shifted_anchor_group.row_nr and neighbor_col >= 0 and  neighbor_col <= shifted_anchor_group.col_nr:
                anchor_fr_idx = shifted_anchor_group.grid[neighbor_row][neighbor_col]
                if anchor_fr_idx:
                    neighbor_comp = get_comparison(pasted_fr_idx, anchor_fr_idx, side1, side2)
                    if neighbor_comp:
                        total_score += neighbor_comp.score
                        total_matchings += 1

    if total_matchings == 0:
        return False
    average_score = total_score / total_matchings
    return average_score

def update_after_merge(groups: List[Group],fragments, fragment_idx_to_group_idx, pasted_group_idx):
    for fr_idx in range(len(fragments)):
        if fragment_idx_to_group_idx[fr_idx] >= pasted_group_idx:
            fragment_idx_to_group_idx[fr_idx] -= 1
    
    del groups[pasted_group_idx]

    

def merge_groups(fragments, pasted_group_additional_rotation, shifted_anchor_group: Group, shifted_pasted_group: Group, fragment_idx_to_group_idx):

    for fr_idx, pos in shifted_pasted_group.fragment_positions.items():
        shifted_anchor_group.fragment_positions[fr_idx] = pos
        fragments[fr_idx].rotation = (fragments[fr_idx].rotation + pasted_group_additional_rotation) % 4

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
            image = gr.show_group(fragments)
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



def edges_of_groups(groups):
    data = []

    for group_idx, group in enumerate(groups):

        rows, cols = len(group.grid), len(group.grid[0])

        for i in range(rows):
            for j in range(cols):
                if group.grid[i][j] is None:
                    neighbour_count = group.neighbours_grid[i][j]
                    if neighbour_count > 0: 
                        data.append({
                            'group_idx': group_idx,
                            'nr_of_neighbours': neighbour_count,
                            'row': i,
                            'col': j
                        })

    df = pd.DataFrame(data)

    if not df.empty:
        df = df.sort_values(by='nr_of_neighbours', ascending=False).reset_index(drop=True)

    return df

def find_best_candidate_for_empty_spot(row, groups):
    anchor_group_idx = row['group_idx']
    empty_row, empty_col = row['row'], row['col']
    g = groups[anchor_group_idx]

    neighbours = [
        g.grid[empty_row - 1][empty_col] if empty_row > 0 else None,
        g.grid[empty_row][empty_col + 1] if empty_col + 1 < g.col_nr else None,
        g.grid[empty_row + 1][empty_col] if empty_row + 1 < g.row_nr else None,
        g.grid[empty_row][empty_col - 1] if empty_col > 0 else None
    ]

    best_score = float('inf')
    best_comp = None
    best_fragment_idx = None
    best_pasted_group_idx = None

    for pasted_group_idx, pasted_group in enumerate(groups):
        if pasted_group_idx == anchor_group_idx:
            continue

        for fr_idx in pasted_group.used_fragments:
            score = 0
            valid = False
            comps = []

            ### need to find normalized values of sides
            if neighbours[0] is not None:
                comp = get_comparison(neighbours[0], fr_idx, 2, 0)
                if comp: score += comp.score; comps.append(comp); valid = True
            if neighbours[2] is not None:
                comp = get_comparison(neighbours[2], fr_idx, 0, 2)
                if comp: score += comp.score; comps.append(comp); valid = True
            if neighbours[3] is not None:
                comp = get_comparison(neighbours[3], fr_idx, 1, 3)
                if comp: score += comp.score; comps.append(comp); valid = True
            if neighbours[1] is not None:
                comp = get_comparison(neighbours[1], fr_idx, 3, 1)
                if comp: score += comp.score; comps.append(comp); valid = True
            


            if valid:
                comp = comps[0]
                shifted_anchor_group, shifted_pasted_group = simulate_merge_positions(comp, groups[anchor_group_idx], groups[pasted_group_idx])

                if does_merge_fit_within_bounds(shifted_anchor_group):
                        if check_groups_shapes_for_merging(shifted_anchor_group, shifted_pasted_group):
                            score = calculate_all_group_matchings_scores(shifted_anchor_group, shifted_pasted_group)
                            if score:
                                if score < best_score:
                                    best_score = score
                                    best_comp = comp
                                    best_fragment_idx = fr_idx
                                    best_pasted_group_idx = pasted_group_idx

    if best_comp:
        return {
            'anchor_group_idx': anchor_group_idx,
            'empty_spot_neighbours': neighbours,
            'pasted_group_idx': best_pasted_group_idx,
            'fragment_idx': best_fragment_idx,
            'score': best_score,
            'comp': best_comp
        }
    return None


def solve_groups(groups, fragments, fragment_idx_to_group_idx):
    while len(groups) > 1:
        edges_of_groups_df = edges_of_groups(groups)
        

        if edges_of_groups_df.empty:
            print("No empty spots with neighbours left.")
            break

        max_neighbours = edges_of_groups_df['nr_of_neighbours'][0] + 1

        #### more neighbours first, then score
        merge_candidates = []
        while not merge_candidates and max_neighbours > 1:
            max_neighbours -= 1
            for _, row in edges_of_groups_df.iterrows():
                if row['nr_of_neighbours'] == max_neighbours:
                    candidate = find_best_candidate_for_empty_spot(row, groups)
                    if candidate:
                        merge_candidates.append(candidate)

        ### best score first
        # merge_candidates = []
        # for _, row in edges_of_groups_df.iterrows():
        #     candidate = find_best_candidate_for_empty_spot(row, groups)
        #     if candidate:
        #         merge_candidates.append(candidate)


        if not merge_candidates:
            print("No valid merge candidates found.")
            break

        merge_candidates.sort(key=lambda c: c['score'])
        # print([round(c['score'], 6) for c in merge_candidates])
        while merge_candidates:
            best = merge_candidates.pop(0)
            comp = best['comp']
            anchor_group_idx = best['anchor_group_idx']
            pasted_group_idx = best['pasted_group_idx']

            shifted_anchor_group, shifted_pasted_group = simulate_merge_positions(comp, groups[anchor_group_idx], groups[pasted_group_idx])

            if does_merge_fit_within_bounds(shifted_anchor_group):
                if check_groups_shapes_for_merging(shifted_anchor_group, shifted_pasted_group):
                    groups[anchor_group_idx] = merge_groups(shifted_anchor_group, shifted_pasted_group, fragment_idx_to_group_idx)
                    update_after_merge(groups, fragments, fragment_idx_to_group_idx, pasted_group_idx)
                    print(f"Merged group {anchor_group_idx} and {pasted_group_idx} with total score: {best['score']} using: {comp}")
                    break
        else:
            print("No suitable merge candidate found after filtering.")
            break

    return groups, fragments, fragment_idx_to_group_idx
