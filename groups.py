import global_values 
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from typing import List
from sides_comparison import SidesComparison
import pandas as pd


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
            print("no empty edge in merging")
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

                    top = i * global_values.TILE_H
                    left = j * global_values.TILE_W
                    canvas_img[top:top+global_values.TILE_H, left:left+global_values.TILE_W] = img_resized

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




def find_pasted_group_moving_distance(anchor_side, pasted_side):  

    row_dist = 0
    col_dist = 0
    if anchor_side.side_idx == 1 and pasted_side.side_idx == 3:
        col_dist = 1
    elif anchor_side.side_idx == 3 and pasted_side.side_idx == 1:
        col_dist = -1
    elif anchor_side.side_idx == 2 and pasted_side.side_idx == 0:
        row_dist = 1
    elif anchor_side.side_idx == 0 and pasted_side.side_idx == 2:
        row_dist = -1
    else:
        print("de implementat rotiri")
        return None
    return row_dist, col_dist


def check_groups_shapes_for_merging(comp: SidesComparison, anchor_group: Group, pasted_group: Group):
    anchor_side = comp.side1
    pasted_side = comp.side2

    side_row_offset, side_col_offset = find_pasted_group_moving_distance(anchor_side, pasted_side)
    anchor_row, anchor_col = anchor_group.fragment_positions[anchor_side.fragment_idx]
    pasted_row, pasted_col = pasted_group.fragment_positions[pasted_side.fragment_idx]

    row_dist = anchor_row + side_row_offset - pasted_row
    col_dist = anchor_col + side_col_offset - pasted_col

    for fr_idx in pasted_group.used_fragments:
        current_row, current_col = pasted_group.fragment_positions[fr_idx]
        new_row = current_row + row_dist
        new_col = current_col + col_dist

        if 0 < new_row < anchor_group.row_nr and 0 < new_col < anchor_group.col_nr:
            if anchor_group.grid[new_row][new_col] is not None:
                # print("impossible merging: incompatible group shapes")
                return False
    return True
    

def does_merge_fit_within_bounds(comp: SidesComparison, anchor_group: Group, pasted_group: Group):

    anchor_side = comp.side1
    pasted_side = comp.side2

    side_row_offset, side_col_offset = find_pasted_group_moving_distance(anchor_side, pasted_side)
    anchor_row, anchor_col = anchor_group.fragment_positions[anchor_side.fragment_idx]
    pasted_row, pasted_col = pasted_group.fragment_positions[pasted_side.fragment_idx]
    row_offset = anchor_row + side_row_offset - pasted_row
    col_offset = anchor_col + side_col_offset - pasted_col

    all_positions = []

    for fr_idx in anchor_group.used_fragments:
        all_positions.append(anchor_group.fragment_positions[fr_idx])

    for fr_idx in pasted_group.used_fragments:
        pr, pc = pasted_group.fragment_positions[fr_idx]
        all_positions.append([pr + row_offset, pc + col_offset])

    all_rows = [r for r, _ in all_positions]
    all_cols = [c for _, c in all_positions]

    min_row = min(all_rows)
    max_row = max(all_rows)
    min_col = min(all_cols)
    max_col = max(all_cols)

    height = max_row - min_row
    width = max_col - min_col

    if height + 1 > global_values.ROW_NR or width + 1 > global_values.COL_NR:  
        # print(f"Merge would exceed puzzle size")
        return False
    return True


def check_all_group_matchings_scores(comp: SidesComparison, anchor_group: Group, pasted_group: Group):
    anchor_side = comp.side1
    pasted_side = comp.side2

    side_row_offset, side_col_offset = find_pasted_group_moving_distance(anchor_side, pasted_side)
    anchor_row, anchor_col = anchor_group.fragment_positions[anchor_side.fragment_idx]
    pasted_row, pasted_col = pasted_group.fragment_positions[pasted_side.fragment_idx]
    row_offset = anchor_row + side_row_offset - pasted_row
    col_offset = anchor_col + side_col_offset - pasted_col

    total_score = 0.0
    total_matchings = 0
    directions = [(-1, 0, 0, 2), (1, 0, 2, 0), (0, -1, 3, 1), (0, 1, 1, 3)]

    for fr_idx in pasted_group.used_fragments:
        pr, pc = pasted_group.fragment_positions[fr_idx]
        new_r = pr + row_offset
        new_c = pc + col_offset

        for dr, dc, side_1, side_2 in directions:
            nr = new_r + dr
            nc = new_c + dc
            for anchor_idx in anchor_group.used_fragments:
                ar, ac = anchor_group.fragment_positions[anchor_idx]
                if ar == nr and ac == nc:
                    comp = get_comparison(fr_idx, anchor_idx, side_1, side_2)
                    if comp:
                        if comp.score > 0.2:
                            return False
                        total_score += comp.score
                        total_matchings += 1

    total_score /= total_matchings
    if total_score > 0.08:
        return False
    print(total_score)
    return True


def calculate_all_group_matchings_scores(comp: SidesComparison, anchor_group: Group, pasted_group: Group):
    anchor_side = comp.side1
    pasted_side = comp.side2

    side_row_offset, side_col_offset = find_pasted_group_moving_distance(anchor_side, pasted_side)
    anchor_row, anchor_col = anchor_group.fragment_positions[anchor_side.fragment_idx]
    pasted_row, pasted_col = pasted_group.fragment_positions[pasted_side.fragment_idx]
    row_offset = anchor_row + side_row_offset - pasted_row
    col_offset = anchor_col + side_col_offset - pasted_col

    total_score = 0.0
    total_matchings = 0
    directions = [(-1, 0, 0, 2), (1, 0, 2, 0), (0, -1, 3, 1), (0, 1, 1, 3)]

    for fr_idx in pasted_group.used_fragments:
        pr, pc = pasted_group.fragment_positions[fr_idx]
        new_r = pr + row_offset
        new_c = pc + col_offset

        for dr, dc, side_1, side_2 in directions:
            nr = new_r + dr
            nc = new_c + dc
            for anchor_idx in anchor_group.used_fragments:
                ar, ac = anchor_group.fragment_positions[anchor_idx]
                if ar == nr and ac == nc:
                    comp = get_comparison(fr_idx, anchor_idx, side_1, side_2)
                    if comp:
                        total_score += comp.score
                        total_matchings += 1

    total_score /= total_matchings
    return total_score



def update_after_merge(groups: List[Group],fragments, fragment_idx_to_group_idx, pasted_group_idx):
    for fr_idx in range(len(fragments)):
        if fragment_idx_to_group_idx[fr_idx] >= pasted_group_idx:
            fragment_idx_to_group_idx[fr_idx] -= 1
    
    del groups[pasted_group_idx]

def merge_groups(comp: SidesComparison, fragments, fragment_idx_to_group_idx, anchor_group: Group, pasted_group: Group):
    anchor_side = comp.side1
    pasted_side = comp.side2

    side_row_offset, side_col_offset = find_pasted_group_moving_distance(anchor_side, pasted_side)
    anchor_row, anchor_col = anchor_group.fragment_positions[anchor_side.fragment_idx]
    pasted_row, pasted_col = pasted_group.fragment_positions[pasted_side.fragment_idx]
    row_dist_pasted_group = anchor_row + side_row_offset - pasted_row
    col_dist_pasted_group = anchor_col + side_col_offset - pasted_col

    for fr_idx in pasted_group.used_fragments:
        old_row, old_col = pasted_group.fragment_positions[fr_idx]
        anchor_group.fragment_positions[fr_idx] = [old_row + row_dist_pasted_group, old_col + col_dist_pasted_group]

    all_rows = [r for r, c in anchor_group.fragment_positions.values()]
    all_cols = [c for r, c in anchor_group.fragment_positions.values()]
    min_row = min(all_rows)
    min_col = min(all_cols)
    max_row = max(all_rows)
    max_col = max(all_cols)

    row_dist_group = 1 - min_row 
    col_dist_group = 1 - min_col  

    anchor_group.row_nr = max_row - min_row + 3 
    anchor_group.col_nr = max_col - min_col + 3

    for fr_idx in anchor_group.fragment_positions:
        row, col = anchor_group.fragment_positions[fr_idx]
        anchor_group.fragment_positions[fr_idx] = [row + row_dist_group, col + col_dist_group]

    new_grid = [[None for _ in range(anchor_group.col_nr)] for _ in range(anchor_group.row_nr)]
    new_neigh_grid = [[0 for _ in range(anchor_group.col_nr)] for _ in range(anchor_group.row_nr)]

    anchor_group.used_fragments.extend(pasted_group.used_fragments)
    for fr_idx in anchor_group.fragment_positions:
        row, col = anchor_group.fragment_positions[fr_idx]
        new_grid[row][col] = fr_idx
        fragment_idx_to_group_idx[fr_idx] = fragment_idx_to_group_idx[anchor_group.used_fragments[0]] 

    anchor_group.grid = new_grid
    anchor_group.neighbours_grid = new_neigh_grid

    for fr_idx in anchor_group.used_fragments:
        row, col = anchor_group.fragment_positions[fr_idx]
        anchor_group.update_neighbours_grid_after_new_merge(row, col)
    

def show_all_groups(groups, fragments, fr_idx_to_group_idx, max_cols=8):
    images = []
    group_indices = []

    for gr in groups:
        if len(gr.used_fragments) > 1:
            image = gr.show_group(fragments)
            images.append(image)
            gr_idx = fr_idx_to_group_idx[gr.used_fragments[0]]
            group_indices.append(gr_idx)

    n = len(images)
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
                score = calculate_all_group_matchings_scores(comp, groups[anchor_group_idx], groups[pasted_group_idx])
                if check_groups_shapes_for_merging(comp, groups[anchor_group_idx], groups[pasted_group_idx]) and \
                   does_merge_fit_within_bounds(comp, groups[anchor_group_idx], groups[pasted_group_idx]):
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

        merge_candidates = []
        while not merge_candidates and max_neighbours > 1:
            max_neighbours -= 1
            for _, row in edges_of_groups_df.iterrows():
                if row['nr_of_neighbours'] == max_neighbours:
                    candidate = find_best_candidate_for_empty_spot(row, groups)
                    if candidate:
                        merge_candidates.append(candidate)

        if not merge_candidates:
            print("No valid merge candidates found.")
            break

        merge_candidates.sort(key=lambda c: c['score'])

        while merge_candidates:
            best = merge_candidates.pop(0)
            comp = best['comp']
            anchor_idx = best['anchor_group_idx']
            pasted_idx = best['pasted_group_idx']

            if check_groups_shapes_for_merging(comp, groups[anchor_idx], groups[pasted_idx]) and \
               does_merge_fit_within_bounds(comp, groups[anchor_idx], groups[pasted_idx]):

                merge_groups(comp, fragments, fragment_idx_to_group_idx, groups[anchor_idx], groups[pasted_idx])
                update_after_merge(groups, fragments, fragment_idx_to_group_idx, pasted_idx)
                print(f"Merged group {anchor_idx} and {pasted_idx} using: {comp}")
                break
        else:
            print("No suitable merge candidate found after filtering.")
            break

    return groups, fragments, fragment_idx_to_group_idx
