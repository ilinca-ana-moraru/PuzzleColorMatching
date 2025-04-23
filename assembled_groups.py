import global_values 
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def check_if_two_fragments_are_vertical(sidesComp):
    if sidesComp.side1.side_idx == 1 and sidesComp.side2.side_idx == 3:
        return True
    elif sidesComp.side1.side_idx == 2 and sidesComp.side2.side_idx == 0:
        return True
    return False

class assembledGroup:

    def __init__(self, sidesComp):
        max_tiles = max(global_values.ROW_NR, global_values.COL_NR)
        rows, cols = max_tiles * 2, max_tiles * 2
        self.origin_row = rows // 2 - 1
        self.origin_col = cols // 2 - 1

        self.grid = [[None for _ in range(cols)] for _ in range(rows)]

        self.used_fragments = []
        self.fragment_positions = {}

        self.used_fragments.append(sidesComp.side1.fragment_idx)
        self.used_fragments.append(sidesComp.side2.fragment_idx)
        if sidesComp.side1.side_idx == 1 and sidesComp.side2.side_idx == 3:
            self.grid[self.origin_row][self.origin_col] = sidesComp.side1.fragment_idx
            self.fragment_positions[sidesComp.side1.fragment_idx] = [self.origin_row, self.origin_col]
            self.grid[self.origin_row][self.origin_col + 1] = sidesComp.side2.fragment_idx
            self.fragment_positions[sidesComp.side2.fragment_idx] = [self.origin_row, self.origin_col + 1]

        elif sidesComp.side1.side_idx == 2 and sidesComp.side2.side_idx == 0:
            self.grid[self.origin_row][self.origin_col] = sidesComp.side1.fragment_idx
            self.fragment_positions[sidesComp.side1.fragment_idx] = [self.origin_row, self.origin_col]
            self.grid[self.origin_row + 1][self.origin_col] = sidesComp.side2.fragment_idx
            self.fragment_positions[sidesComp.side2.fragment_idx] = [self.origin_row + 1, self.origin_col]


    def add_fragment(self, fragments, anchor_side, new_side):

        anchor_row, anchor_col = self.fragment_positions[anchor_side.fragment_idx]

        new_fr_row, new_fr_col = find_pasted_fragment_coords(anchor_side, new_side, anchor_row, anchor_col)

        if self.grid[new_fr_row][new_fr_col] == None:

            self.grid[new_fr_row][new_fr_col] = new_side.fragment_idx
            self.fragment_positions[new_side.fragment_idx] = [new_fr_row, new_fr_col]
            self.used_fragments.append(new_side.fragment_idx)
            return True
        return False
            
    def nr_of_neighbours_grid(self):
        neighbours_grid = np.zeros_like(self.grid)
        rows, cols = len(self.grid), len(self.grid[0])

        placed_coords = list(self.fragment_positions.values())
        if not placed_coords:
            print("No fragments to display.")
            return

        min_row = max(min(pos[0] for pos in placed_coords) - 1 , 0)
        max_row = min(max(pos[0] for pos in placed_coords) + 1, rows - 1) 
        min_col = max(min(pos[1] for pos in placed_coords) - 1, 0)
        max_col = min(max(pos[1] for pos in placed_coords) + 1, cols - 1)

        for i in range(min_row, max_row + 1):
            for j in range(min_col, max_col + 1):
                if self.grid[i][j] is None:
                    if i - 1 >= min_row and self.grid[i - 1][j] is not None:
                        neighbours_grid[i][j] +=1
                    if i + 1 <= max_row and self.grid[i + 1][j] is not None:
                        neighbours_grid[i][j] +=1
                    if j - 1 >= min_col and self.grid[i][j - 1] is not None:
                        neighbours_grid[i][j] +=1
                    if j + 1 <= max_col and self.grid[i][j + 1] is not None:
                        neighbours_grid[i][j] +=1
        self.neighbours_grid = neighbours_grid

    def show(self, fragments):
        fig, ax = plt.subplots()
        rows, cols = len(self.grid), len(self.grid[0])
        canvas = np.zeros((rows * global_values.TILE_W, cols * global_values.TILE_H, 3), dtype=np.uint8)

        for i, row in enumerate(self.grid):
            for j, cell in enumerate(row):
                if  cell is not None:
                    fragment = fragments[cell]
                    img = fragment.value[:, :, :3] 
                    h, w = img.shape[:2]
                    
                    if (h, w) != (global_values.TILE_H, global_values.TILE_W):
                        img_resized = resize(img, (global_values.TILE_H, global_values.TILE_W), preserve_range=True, anti_aliasing=True)
                    else:
                        img_resized = img

                    top = i * global_values.TILE_H
                    left = j * global_values.TILE_W
                    canvas[top:top+global_values.TILE_H, left:left+global_values.TILE_W] = img_resized

        return canvas


    def show_no_background(self, fragments, save):
        rows, cols = len(self.grid), len(self.grid[0])

        placed_coords = list(self.fragment_positions.values())
        if not placed_coords:
            print("No fragments to display.")
            return None

        min_row = max(min(pos[0] for pos in placed_coords) - 1, 0)
        max_row = min(max(pos[0] for pos in placed_coords) + 1, rows - 1)
        min_col = max(min(pos[1] for pos in placed_coords) - 1, 0)
        max_col = min(max(pos[1] for pos in placed_coords) + 1, cols - 1)

        height = (max_row - min_row + 1) * global_values.TILE_H
        width = (max_col - min_col + 1) * global_values.TILE_W
        canvas_img = np.zeros((height, width, 3), dtype=np.uint8)

        for i in range(min_row, max_row + 1):
            for j in range(min_col, max_col + 1):
                cell = self.grid[i][j]
                if cell is not None:
                    fragment = fragments[cell]
                    img = fragment.value[:, :, :3]
                    h, w = img.shape[:2]

                    if (h, w) != (global_values.TILE_H, global_values.TILE_W):
                        img_resized = resize(img, (global_values.TILE_H, global_values.TILE_W), preserve_range=True, anti_aliasing=True)
                    else:
                        img_resized = img

                    top = (i - min_row) * global_values.TILE_H
                    left = (j - min_col) * global_values.TILE_W
                    canvas_img[top:top+global_values.TILE_H, left:left+global_values.TILE_W] = img_resized

        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        ax.imshow(canvas_img)
        ax.axis("off")

        for i in range(min_row, max_row + 1):
            for j in range(min_col, max_col + 1):
                count = self.neighbours_grid[i][j]
                if count > 0:
                    x = (j - min_col) * global_values.TILE_W + global_values.TILE_W // 2
                    y = (i - min_row) * global_values.TILE_H + global_values.TILE_H // 2
                    ax.text(x, y, str(count), color='red', ha='center', va='center', fontsize=12, weight='bold')

        canvas = FigureCanvas(fig)
        canvas.draw()
        img_array = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape((int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), 3))

        plt.close(fig)

        if save == 1:
            plt.imsave("solved_puzzle.jpg", img_array)

        return img_array


def merge_groups(comp,fragments, fragment_idx_to_group_idx, groups, anchor_group_idx, pasted_group_idx):
    
    print("------------------------------trying to paste group----------------------------------------------------------")

    print(f"anchor group: {anchor_group_idx} pasted group: {pasted_group_idx}")

    original_first_anchor_row, original_first_anchor_col = groups[anchor_group_idx].fragment_positions[comp.side1.fragment_idx]
    original_first_pasted_row, original_first_pasted_col = groups[pasted_group_idx].fragment_positions[comp.side2.fragment_idx]
    print(f"original_first_anchor_row{original_first_anchor_row} original_first_anchor_col {original_first_anchor_col}")
    print(f"original_first_pasted_row{original_first_pasted_row} original_first_pasted_col {original_first_pasted_col}")

    first_pasted_row, first_pasted_col = find_pasted_fragment_coords(comp.side1, comp.side2, original_first_anchor_row, original_first_anchor_col)
    print(f"first_pasted_row{first_pasted_row} first_pasted_col {first_pasted_col}")

    moving_dist_row = first_pasted_row - original_first_pasted_row
    moving_dist_col = first_pasted_col - original_first_pasted_col
    print(f"moving_dist_row{moving_dist_row} moving_dist_col {moving_dist_col}")

    for fr_idx in range(len(fragment_idx_to_group_idx)):
        if fragment_idx_to_group_idx[fr_idx] == pasted_group_idx:
            old_row, old_col = groups[pasted_group_idx].fragment_positions[fr_idx]
            new_row = old_row + moving_dist_row
            new_col = old_col + moving_dist_col

            if groups[anchor_group_idx].grid[new_row][new_col] is not None:
                print(f"Cell already occupied at ({new_row}, {new_col})")
                return None, None

    for fr_idx in range(len(fragment_idx_to_group_idx)):
        if fragment_idx_to_group_idx[fr_idx] == pasted_group_idx:
            old_row, old_col = groups[pasted_group_idx].fragment_positions[fr_idx]
            new_row = old_row + moving_dist_row
            new_col = old_col + moving_dist_col
            groups[anchor_group_idx].grid[new_row][new_col] = fr_idx
            groups[anchor_group_idx].fragment_positions[fr_idx] = new_row, new_col
            groups[anchor_group_idx].used_fragments.append(fr_idx)
            
            fragment_idx_to_group_idx[fr_idx] = anchor_group_idx
            
    for fr_idx in range(len(fragment_idx_to_group_idx)):
        if fragment_idx_to_group_idx[fr_idx] is not None and fragment_idx_to_group_idx[fr_idx] > pasted_group_idx:
            fragment_idx_to_group_idx[fr_idx] -= 1
    groups.pop(pasted_group_idx)

    return (fragment_idx_to_group_idx, groups)



def find_pasted_fragment_coords(anchor_side, new_side, anchor_row, anchor_col):  
    new_fr_row = anchor_row
    new_fr_col = anchor_col
    if anchor_side.side_idx == 1 and new_side.side_idx == 3:
        new_fr_col += 1
    elif anchor_side.side_idx == 3 and new_side.side_idx == 1:
        new_fr_col -= 1
    elif anchor_side.side_idx == 2 and new_side.side_idx == 0:
        new_fr_row += 1

    elif anchor_side.side_idx == 0 and new_side.side_idx == 2:
        new_fr_row -= 1
    
    else:
        print("de implementat rotiri")
        return None
    return new_fr_row, new_fr_col

