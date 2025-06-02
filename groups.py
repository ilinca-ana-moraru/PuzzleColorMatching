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
            

    def show_group(self, fragments, extra_rotation):

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

                    img_rotated = rotate_image(img_resized, (fragment.rotation+extra_rotation)%4)                 
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



def merge_where_obvious(one_match_condition, mean_condition, one_image_th, group_th, sorted_sides_comparisons, fragment_idx_to_group_idx, fragments, groups):

    for comp in sorted_sides_comparisons:
        # if comp.score < 0.02:
        if one_match_condition(comp, one_image_th):
            
            anchor_fragment_idx = comp.side1.fragment_idx
            pasted_fragment_idx = comp.side2.fragment_idx
            anchor_group_idx = fragment_idx_to_group_idx[anchor_fragment_idx]
            pasted_group_idx = fragment_idx_to_group_idx[pasted_fragment_idx]

            if anchor_group_idx != pasted_group_idx:

                shifted_anchor_group, shifted_pasted_group, pasted_group_additional_rotation = simulate_merge_positions(fragments, comp, groups[anchor_group_idx], groups[pasted_group_idx])

                if does_merge_fit_within_bounds(shifted_anchor_group):
                    if check_groups_shapes_for_merging(shifted_anchor_group, shifted_pasted_group):
                        # print(f"{comp}")

                        if check_all_group_matchings_scores(one_match_condition,mean_condition, fragments, pasted_group_additional_rotation, shifted_anchor_group, shifted_pasted_group, one_image_th, group_th):    
                            groups[anchor_group_idx] = merge_groups(fragments, pasted_group_additional_rotation, shifted_anchor_group, shifted_pasted_group, fragment_idx_to_group_idx)
                            update_after_merge(groups, fragments, fragment_idx_to_group_idx, pasted_group_idx)
                            print(comp)

    show_all_groups(groups, fragments, fragment_idx_to_group_idx, 0)
    return groups, fragments, fragment_idx_to_group_idx


from collections import defaultdict
def create_able_to_vote_sides_df(groups, fragments):

    df = pd.DataFrame(columns=['group_idx','fragment_idx','side_idx', 'row', 'col'])

    # up 0 , down 2, left 3, right 1
    directions = [
        (-1, 0, 0), 
        (1, 0, 2),  
        (0, -1, 3),
        (0, 1, 1), 
    ]

    for group_idx, group in enumerate(groups):
        rows, cols = len(group.grid), len(group.grid[0])

        for i in range(rows):
            for j in range(cols):
                fragment_idx = group.grid[i][j]
                if fragment_idx is not None:
                    for di, dj, side_orientation in directions:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            if group.grid[ni][nj] is None:
                                side_idx = (side_orientation + fragments[fragment_idx].rotation) % 4
                                df.loc[len(df)] = {
                                    'group_idx': group_idx,
                                    'fragment_idx': fragment_idx,
                                    'side_idx': side_idx,
                                    'row': i,
                                    'col': j,
                                }

    return df

def vote_and_solve(groups, fragments, fragment_idx_to_group_idx, one_match_condition, group_condition, one_match_th, group_th):
    while len(groups) > 1:
        voting_df = create_able_to_vote_sides_df(groups, fragments)
    

        # For each vote key, we store [count, sum_score]
        vote_stats = defaultdict(lambda: [0, 0.0])
        best_comp_for_vote_key = {}
        # Loop through voters
        for _, voter in voting_df.iterrows():
            voting_fragment_idx = voter['fragment_idx']
            voting_side_idx = voter['side_idx']
            best_comp = None
            best_score = None

            for _, candidate in voting_df.iterrows():
                candidate_fragment_idx = candidate['fragment_idx']
                if  fragment_idx_to_group_idx[voting_fragment_idx] != fragment_idx_to_group_idx[candidate_fragment_idx] and candidate_fragment_idx != voting_fragment_idx:
                    candidate_side_idx = candidate['side_idx']
                    comp = get_comparison(voting_fragment_idx, candidate_fragment_idx, voting_side_idx, candidate_side_idx)
                    if best_score is None or best_score > comp.score:
                        best_score = comp.score
                        best_comp = comp

            offset_row, offset_col, pasted_group_additional_rotation = find_pasted_group_moving_distance_and_rotation(fragments, best_comp)

            vote_group_idx = fragment_idx_to_group_idx[best_comp.side1.fragment_idx]
            candidate_group_idx = fragment_idx_to_group_idx[best_comp.side2.fragment_idx]

            vote_group = groups[vote_group_idx]
            candidate_group = groups[candidate_group_idx]
            candidate_group_copy = copy.deepcopy(candidate_group)
            candidate_group_copy = rotate_fragments_positions(candidate_group_copy, pasted_group_additional_rotation)

            vote_row, vote_col = vote_group.fragment_positions[best_comp.side1.fragment_idx]
            candidate_row, candidate_col = candidate_group_copy.fragment_positions[best_comp.side2.fragment_idx]
            row_offset = vote_row + offset_row - candidate_row
            col_offset = vote_col + offset_col - candidate_col

            vote_key = (vote_group_idx, candidate_group_idx, best_comp.side1.side_idx, best_comp.side2.side_idx, pasted_group_additional_rotation)

            vote_stats[vote_key][0] += 1
            vote_stats[vote_key][1] += best_score

            if vote_key not in best_comp_for_vote_key:
                best_comp_for_vote_key[vote_key] = best_comp
        
        vote_list = sorted(vote_stats.items(), key=lambda x: (-x[1][0], x[1][1] / x[1][0])) 
        print("\nVotes ordered by number of votes:\n")
        was_merged = False
        posibilities_remain = False
        for vote_key, (count, sum_score) in vote_list:
            vote_group_idx, candidate_group_idx, anchor_side_idx, candidate_side_idx, rotation = vote_key
            mean_score = sum_score / count
            comp = best_comp_for_vote_key[vote_key]

            shifted_anchor_group, shifted_pasted_group, pasted_group_additional_rotation = simulate_merge_positions(fragments, comp, groups[vote_group_idx], groups[candidate_group_idx])

            if does_merge_fit_within_bounds(shifted_anchor_group):
                if check_groups_shapes_for_merging(shifted_anchor_group, shifted_pasted_group):
                    if check_all_group_matchings_scores(one_match_condition, group_condition,fragments, pasted_group_additional_rotation, shifted_anchor_group, shifted_pasted_group, one_match_th, group_th):
                        print(f"GROUP {vote_group_idx} votes for GROUP {candidate_group_idx} with offset ({row_offset},{col_offset}), rotation {rotation} --> {count} votes, mean_score={mean_score:.6f}")
                        groups[vote_group_idx] = merge_groups(fragments, pasted_group_additional_rotation, shifted_anchor_group, shifted_pasted_group, fragment_idx_to_group_idx)
                        update_after_merge(groups, fragments, fragment_idx_to_group_idx, candidate_group_idx)
                        was_merged = True
                        break
                    
        if posibilities_remain == False:
            return groups, fragments, fragment_idx_to_group_idx
        if was_merged == False:
            one_match_th *= 1.1
            group_th *= 1.1
            print(f"---------------------------")
            print(f"one match th {one_match_th} group th {group_th}")
        if one_match_th == 5:
            print("valeu")
            return groups, fragments, fragment_idx_to_group_idx


    return groups, fragments, fragment_idx_to_group_idx

