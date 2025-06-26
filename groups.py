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
import os

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

    import numpy as np
    import cv2

    def show_group_to_scale(self, fragments,
                            extra_rotation: int = 0,
                            draw_grid: bool = True,
                            tile_scale: float = 1.0,
                            cell_border_px: int = 2):
        """Returnează imagine RGB cu piese rotite şi grilă perfect pixel-aligned."""
        # ---- dimensiunea unei piese ------------------------------
        th0, tw0 = global_values.TILE_H, global_values.TILE_W
        th, tw   = int(round(th0 * tile_scale)), int(round(tw0 * tile_scale))

        rows = max(rc[0] for rc in self.fragment_positions.values()) 
        cols = max(rc[1] for rc in self.fragment_positions.values()) 
        H, W  = rows * th, cols * tw

        canvas = np.full((H, W, 3), 255, dtype=np.uint8)

        # ---- copiem piesele --------------------------------------
        for fid, (r, c) in self.fragment_positions.items():
            r = r - 1
            c = c - 1
            img = fragments[fid].value[:, :, :3]
            if img.shape[:2] != (th0, tw0):
                img = cv2.resize(img, (tw0, th0), interpolation=cv2.INTER_AREA)
            img = rotate_image(img, (fragments[fid].rotation + extra_rotation) % 4)

            if tile_scale != 1.0:
                img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)

            r0, c0 = r * th, c * tw
            canvas[r0:r0+th, c0:c0+tw] = img

        # ---- desenăm grila ---------------------------------------
        if draw_grid:
            col = (0, 0, 0)
            for r in range(rows + 1):
                y = r * th
                cv2.line(canvas, (0, y), (W-1, y), col, cell_border_px)
            for c in range(cols + 1):
                x = c * tw
                cv2.line(canvas, (x, 0), (x, H-1), col, cell_border_px)

        return canvas



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
            ax.set_title(f"Grupul {group_indices[i]}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    return fig

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from skimage.transform import resize
from math import gcd

try:
    from rectpack import newPacker
except ImportError:
    def show_all_groups_to_scale(*_, **__):
        raise ImportError("rectpack nu este instalat  →  pip install rectpack")

import numpy as np
import cv2


def _as_np_image(tile):
    """Extrage np.ndarray RGB din obiectul tile (inclusiv Fragment.value)."""
    if isinstance(tile, np.ndarray):
        arr = tile
    elif hasattr(tile, "value"):
        arr = tile.value
    elif hasattr(tile, "image"):
        arr = tile.image
    elif hasattr(tile, "img"):
        arr = tile.img
    elif hasattr(tile, "data"):
        arr = tile.data
    elif isinstance(tile, (tuple, list)):
        arr = next(part for part in tile if isinstance(part, np.ndarray))
    else:
        raise TypeError(f"Nu ştiu să extrag imagine din {type(tile).__name__}")

    # doar RGB
    return arr[..., :3] if arr.ndim == 3 and arr.shape[2] > 3 else arr


def _render_group(group,
                  fragments,
                  *,
                  tile_scale: float = 0.8,
                  cell_border_px: int = 2,     # linii negre între tile-uri
                  bg_color=(255, 255, 255)):
    """
    Desenează grupul: fiecare piesă la aceeaşi scară, separată de vecini
    printr-un chenar negru de `cell_border_px`.  Fundalul gol (tiles lipsă)
    este alb, astfel încât colajul final nu mai arată „blocuri” negre.
    """
    # ---- dimensiune unei piese -------------------------------------
    first_tile = fragments[0] if not isinstance(fragments, dict) else next(iter(fragments.values()))
    tile_h_orig, tile_w_orig = _as_np_image(first_tile).shape[:2]
    tile_h = int(round(tile_h_orig * tile_scale))
    tile_w = int(round(tile_w_orig * tile_scale))

    # ---- grid m×n ---------------------------------------------------
    rows = max(rc[0] for rc in group.fragment_positions.values()) + 1
    cols = max(rc[1] for rc in group.fragment_positions.values()) + 1

    cell_h = tile_h + cell_border_px
    cell_w = tile_w + cell_border_px

    H = rows * cell_h + cell_border_px    # margine sus/jos
    W = cols * cell_w + cell_border_px    # margine stg/dr

    canvas = np.full((H, W, 3), bg_color, dtype=np.uint8)

    # ---- copiem fiecare tile ---------------------------------------
    for fid, (r, c) in group.fragment_positions.items():
        tile = _as_np_image(fragments[fid])
        if tile_scale != 1.0:
            tile = cv2.resize(tile, (tile_w, tile_h), interpolation=cv2.INTER_AREA)

        r0 = cell_border_px + r * cell_h
        c0 = cell_border_px + c * cell_w
        canvas[r0:r0 + tile_h, c0:c0 + tile_w, :] = tile

    # ---- desenăm liniile negre pe grid ------------------------------
    for r in range(rows + 1):
        y = r * cell_h
        cv2.line(canvas, (0, y), (W - 1, y), (0, 0, 0), cell_border_px)

    for c in range(cols + 1):
        x = c * cell_w
        cv2.line(canvas, (x, 0), (x, H - 1), (0, 0, 0), cell_border_px)

    return canvas




from math import gcd
from typing import Iterable, Mapping, Any, Tuple

import numpy as np
from rectpack import newPacker

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


import numpy as np
import cv2
from math import gcd
from typing import Iterable, Mapping, Any, Tuple

from rectpack import newPacker
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


# ----------------------------------------------------------------------
# utilitar: extrage imaginea (RGB) din “tile” indiferent de tip
# ----------------------------------------------------------------------
def _as_np_image(tile):
    """
    Returnează un np.ndarray (H×W×3) din argumentul «tile».
    • dacă «tile» e deja ndarray ➜ păstrează doar primele 3 canale;
    • dacă e obiect cu .value / .image / .img / .data ➜ extrage & taie la 3 canale;
    • dacă e o colecție (tuple/list) ➜ caută primul ndarray și taie la 3 canale.
    """
    def _trim(arr):
        return arr[..., :3] if arr.ndim == 3 and arr.shape[2] > 3 else arr

    if isinstance(tile, np.ndarray):
        return _trim(tile)

    for attr in ('value', 'image', 'img', 'data'):
        if hasattr(tile, attr):
            arr = getattr(tile, attr)
            if isinstance(arr, np.ndarray):
                return _trim(arr)

    if isinstance(tile, (tuple, list)):
        for part in tile:
            if isinstance(part, np.ndarray):
                return _trim(part)

    raise TypeError(f"Nu pot extrage matrice imagine din obiect de tip {type(tile).__name__}")


# ----------------------------------------------------------------------
# funcția principală
# ----------------------------------------------------------------------
def show_all_groups_to_scale(
        groups: Iterable[Any],
        fragments: Mapping[int, Any] | list,
        fr_idx_to_group_idx: Mapping[int, int],
        *,
        canvas_px: Tuple[int, int] | None = None,   # (lățime, înălțime) fixă
        dpi: int = 100,
        margin_px: int = 2,
        tile_scale: float = 0.8,
        outer_border_px: int = 2,
        show_titles: bool = False,
        allow_rotation: bool = True,
        grid_align: bool = False,
        puzzle_width_px: int | None = None,
        bin_width_px: int | None = None,
        debug: bool = True):                        # ← pentru print-uri rapide
    """
    Desenează toate grupurile la aceeaşi scară şi le plasează într-un colaj.
    Dacă `canvas_px` e dat, imaginea finală are exact acea mărime
    (ridică ValueError dacă nu încape). În caz contrar, lăţimea se
    calculează automat, iar înălţimea devine minimul necesar.
    """

    # --------------------------------------------------------------- #
    # 1) generează imaginile fiecărui grup                             #
    # --------------------------------------------------------------- #
    rects = []          # (rid, w, h, img, label, orig_w, orig_h)
    grid_px = None

    for gr in groups:
        # ↓↓ apel direct; obții o imagine deja rotită și cu grilă
        img = gr.show_group_to_scale(
                fragments,
                extra_rotation = 0,          # sau alt unghi dacă vrei
                draw_grid      = True,
                tile_scale     = tile_scale)

        h, w = img.shape[:2]
        label = fr_idx_to_group_idx[gr.used_fragments[0]]

        # opțional: dacă vrei un chenar extern identic (2 px) în jurul
        # fiecărui grup, decomentează următoarea linie:
        # import cv2; cv2.rectangle(img, (0, 0), (w-1, h-1), (0,0,0), 2)

        rects.append((len(rects), w, h, img, label, w, h))
        grid_px = h if grid_px is None else gcd(grid_px, h)


    # --------------------------------------------------------------- #
    # 2) container (bin) pentru rectpack                              #
    # --------------------------------------------------------------- #
    if canvas_px is not None:
        bin_width_px, bin_height_px = canvas_px
    else:
        max_w = max(w for _, w, *_ in rects)
        if puzzle_width_px is not None:
            bin_width_px = puzzle_width_px
        elif bin_width_px is None:
            med_w = int(np.median([w for _, w, *_ in rects]))
            bin_width_px = max(med_w * 6, 800)
        bin_width_px = max(bin_width_px, max_w + 2 * margin_px)
        bin_height_px = 2_000_000  # “infinit” pe verticală

    # --------------------------------------------------------------- #
    # 3) împachetare cu rectpack                                      #
    # --------------------------------------------------------------- #
    from rectpack import newPacker
    packer = newPacker(rotation=allow_rotation)
    for rid, w, h, *_ in rects:
        packer.add_rect(w + 2 * margin_px, h + 2 * margin_px, rid)
    packer.add_bin(bin_width_px, bin_height_px)
    packer.pack()

    placed = []          # (x, y, w, h, img, label)
    min_x = min_y = np.inf
    max_x2 = max_y2 = 0

    for _bin, x, y, w_p, h_p, rid in packer.rect_list():
        w, h = w_p - 2 * margin_px, h_p - 2 * margin_px
        x, y = x + margin_px, y + margin_px
        _, _, _, img, label, orig_w, orig_h = rects[rid]
        if (w, h) == (orig_h, orig_w):      # rectpack a rotit
            img = np.rot90(img)

        if grid_align:
            x = round(x / grid_px) * grid_px
            y = round(y / grid_px) * grid_px

        placed.append((x, y, w, h, img, label))
        min_x, min_y = min(min_x, x), min(min_y, y)
        max_x2, max_y2 = max(max_x2, x + w), max(max_y2, y + h)

    # info utilă – vezi dimensiunea reală
    if debug:
        print("după împachetare:", max_x2 + margin_px, "×", max_y2 + margin_px, "px")
    need_w = max_x2 + margin_px
    need_h = max_y2 + margin_px
    # ridică eroare dacă nu toate rect-urile într-un singur bin
    if canvas_px is not None and (need_w > canvas_px[0] or need_h > canvas_px[1]):
        raise ValueError(
            f"Canvasul {canvas_px} e prea mic: "
            f"necesar {need_w}×{need_h} px la tile_scale={tile_scale}."
        )
    # aliniere la colțul (margin_px, margin_px) dacă avem canvas fix
    if canvas_px is not None:
        dx, dy = margin_px - min_x, margin_px - min_y
        placed = [(x + dx, y + dy, w, h, img, lab) for x, y, w, h, img, lab in placed]

    total_w, total_h = bin_width_px, (bin_height_px if canvas_px is not None else max_y2 + margin_px)

    # --------------------------------------------------------------- #
    # 4) desen colaj cu Matplotlib                                    #
    # --------------------------------------------------------------- #
    fig = plt.figure(figsize=(total_w / dpi, total_h / dpi), dpi=dpi)
    # FigureCanvas(fig)

    for x, y, w, h, img, label in placed:
        ax = fig.add_axes([x / total_w,
                        1 - (y + h) / total_h,
                        w / total_w,
                        h / total_h],
                        frameon=False)
        ax.imshow(img, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
            
    output_path = os.path.join("demo",f"{global_values.IMG_IDX:04d}.png")
    global_values.IMG_IDX = global_values.IMG_IDX + 1
    fig.savefig(output_path, dpi=dpi)    

    plt.show()
    return fig

