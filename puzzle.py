import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from collections import defaultdict

from fragment import *
from side import *
from sides_comparison import *
from utils import *
from puzzle import *
from tqdm import tqdm
from global_values import *
from groups import *
# from utils import mahalanobis_merger

def get_matches_accuracy(gt_comparisons, groups, fragments):
    comparisons = []

    for g in groups:
        n = len(g.grid)
        m = len(g.grid[0])

        for i in range(n):
            for j in range(m):
                fr1_idx = g.grid[i][j]
                if fr1_idx is not None:
                    if j + 1 < m:
                        side1_idx = find_side_idx_of_orientation(fragments[fr1_idx].rotation, 1)
                        fr2_idx = g.grid[i][j + 1]
                        if fr2_idx is not None:
                            side2_idx = find_side_idx_of_orientation(fragments[fr2_idx].rotation, 3)
                            comparisons.append((fr1_idx, fr2_idx, side1_idx, side2_idx))

                    if i + 1 < n:
                        side1_idx = find_side_idx_of_orientation(fragments[fr1_idx].rotation, 2)
                        fr2_idx = g.grid[i + 1][j]
                        if fr2_idx is not None:
                            side2_idx = find_side_idx_of_orientation(fragments[fr2_idx].rotation, 0)
                            comparisons.append((fr1_idx, fr2_idx, side1_idx, side2_idx))

    correct = 0
    wrong = 0

    nr_of_comp = int((2 * 4 + 3 * ((global_values.COL_NR - 2) * 2 + (global_values.ROW_NR - 2) * 2) + 4 * ((global_values.COL_NR - 2) * (global_values.COL_NR - 2))) / 2)

    for s_comp in comparisons:
        match_found = False
        for gt_comp in gt_comparisons:
            if s_comp == gt_comp or \
               (s_comp[0] == gt_comp[1] and s_comp[1] == gt_comp[0] and s_comp[2] == gt_comp[3] and s_comp[3] == gt_comp[2]):
                correct += 1
                match_found = True
                break  

        if not match_found:
            wrong += 1

    accuracy = (correct / nr_of_comp) * 100
    false_perc = (wrong / nr_of_comp) * 100 if comparisons else 0

    return accuracy, false_perc



def sort_sides_comparisons(sides_comparisons: List[SidesComparison]):
        return sorted(sides_comparisons, key=lambda x: x.score)




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
                            if pasted_group_additional_rotation > 0:
                                print(f"group {anchor_group_idx} merged with group {pasted_group_idx} - rotation {pasted_group_additional_rotation}")
                                show_all_groups(groups, fragments, fragment_idx_to_group_idx, 1)
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

        vote_stats = defaultdict(lambda: [0, 0.0])
        best_comp_for_vote_key = {}

        for _, voter in voting_df.iterrows():
            voting_fragment_idx = voter['fragment_idx']
            voting_side_idx = voter['side_idx']

            best_comp = None
            best_score = None

            for _, candidate in voting_df.iterrows():
                candidate_fragment_idx = candidate['fragment_idx']

                if fragment_idx_to_group_idx[voting_fragment_idx] != fragment_idx_to_group_idx[candidate_fragment_idx] and candidate_fragment_idx != voting_fragment_idx:
                    candidate_side_idx = candidate['side_idx']
                    comp = get_comparison(voting_fragment_idx, candidate_fragment_idx, voting_side_idx, candidate_side_idx)
                    if comp is not None and (best_score is None or best_score > comp.score):
                        vote_group_idx = fragment_idx_to_group_idx[comp.side1.fragment_idx]
                        candidate_group_idx = fragment_idx_to_group_idx[comp.side2.fragment_idx]
                        shifted_anchor_group, shifted_pasted_group, pasted_group_additional_rotation = simulate_merge_positions(fragments, comp, groups[vote_group_idx], groups[candidate_group_idx])
                        if does_merge_fit_within_bounds(shifted_anchor_group):
                            if check_groups_shapes_for_merging(shifted_anchor_group, shifted_pasted_group):
                                posibilities_remain = True

                                if check_all_group_matchings_scores(
                                    one_match_condition, group_condition, fragments,
                                    pasted_group_additional_rotation,
                                    shifted_anchor_group, shifted_pasted_group,
                                    one_match_th, group_th):

                                    best_score = comp.score
                                    best_comp = comp

            if best_comp is None:
                continue

            vote_group_idx = fragment_idx_to_group_idx[best_comp.side1.fragment_idx]
            candidate_group_idx = fragment_idx_to_group_idx[best_comp.side2.fragment_idx]

            shifted_anchor_group, shifted_pasted_group, pasted_group_additional_rotation = simulate_merge_positions(
                fragments, best_comp, groups[vote_group_idx], groups[candidate_group_idx]
            )

            candidate_group_origin = groups[candidate_group_idx].fragment_positions[best_comp.side2.fragment_idx]
            target_pos_in_anchor = shifted_pasted_group.fragment_positions[best_comp.side2.fragment_idx]

            final_row_offset = target_pos_in_anchor[0] - candidate_group_origin[0]
            final_col_offset = target_pos_in_anchor[1] - candidate_group_origin[1]

            vote_key = (vote_group_idx, candidate_group_idx, final_row_offset, final_col_offset, pasted_group_additional_rotation)

            # print(f"group {vote_group_idx} (fragment {voting_fragment_idx} side {voting_side_idx}) voted for {candidate_group_idx} with offset {final_row_offset}, {final_col_offset}")

            vote_stats[vote_key][0] += 1
            vote_stats[vote_key][1] += best_score

            if vote_key not in best_comp_for_vote_key:
                best_comp_for_vote_key[vote_key] = best_comp

        vote_list = sorted(vote_stats.items(), key=lambda x: (-x[1][0], x[1][1] / x[1][0]))
        # if vote_list:
        #     for v in vote_list:
        #         print(v)
        # else:
        #     print("vote list empty")
        was_merged = False

        for vote_key, (count, sum_score) in vote_list:
            vote_group_idx, candidate_group_idx, offset_row, offset_col, rotation = vote_key
            mean_score = sum_score / count
            comp = best_comp_for_vote_key[vote_key]

            shifted_anchor_group, shifted_pasted_group, pasted_group_additional_rotation = simulate_merge_positions(
                fragments, comp, groups[vote_group_idx], groups[candidate_group_idx]
            )

            if does_merge_fit_within_bounds(shifted_anchor_group):
                if check_groups_shapes_for_merging(shifted_anchor_group, shifted_pasted_group):

                    if check_all_group_matchings_scores(one_match_condition, group_condition, fragments, pasted_group_additional_rotation, shifted_anchor_group, shifted_pasted_group, one_match_th, group_th):

                        print(f"GROUP {vote_group_idx} votes for GROUP {candidate_group_idx} with offset ({offset_row},{offset_col}), rotation {rotation} --> {count} votes, mean_score={mean_score:.6f}")

                        groups[vote_group_idx] = merge_groups(
                            fragments, pasted_group_additional_rotation,
                            shifted_anchor_group, shifted_pasted_group,
                            fragment_idx_to_group_idx)

                        update_after_merge(groups, fragments, fragment_idx_to_group_idx, candidate_group_idx)
                        show_all_groups(groups, fragments, fragment_idx_to_group_idx, 0)

                        was_merged = True
                        break

        if not was_merged:
            one_match_th *= 1.2
            group_th *= 1.2
            # print(f"one match th {one_match_th} group th {group_th}")

        if group_th >= 1:
            return groups, fragments, fragment_idx_to_group_idx

    return groups, fragments, fragment_idx_to_group_idx








def draw_red_border(fragment:Fragment, side: Side):
    fragment_value = fragment.value.copy()

    fragment_value[side.side_indexes_of_fragment[:,0],side.side_indexes_of_fragment[:,1]] = [255, 0, 0, 255]
    return fragment_value





def rotate_fragment(fragments, side, side_type):
    image = fragments[side.fragment_idx].value
   
    h, w = image.shape[:2]

    p1 = side.side_indexes_of_fragment[0]
    p2 = side.side_indexes_of_fragment[-1]

    x, y = p2[0] - p1[0], p2[1] - p1[1]
    th1 = np.degrees(np.arctan2(y, x))

    if side_type == 1:
        p1 = [0, w-1]
        p2 = [h-1, w-1]
    else:
        p1 = [h-1, 0]
        p2 = [0, 0]
    x, y = p2[0] - p1[0], p2[1] - p1[1]
    th2 = np.degrees(np.arctan2(y, x))

    rotation_angle = th2 - th1
  

    if abs(rotation_angle) < 5:
        return image
    
    elif rotation_angle > 80 and rotation_angle < 100 or rotation_angle < -260 and rotation_angle > -280:
        image = cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE)

    elif abs(rotation_angle) > 170 and abs(rotation_angle) < 190 or rotation_angle < -170 and rotation_angle > -190:
        image = cv.rotate(image, cv.ROTATE_180)
    
    elif rotation_angle < -80 and rotation_angle > -100  or rotation_angle > 260 and rotation_angle < 280:
        image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
    else:
        print(f"invalid rotation angle {rotation_angle}")
        return 0
  
    return image



def two_fragments_merger(fragments, side1, side2):
    rotated_fragment1 = rotate_fragment(fragments, side1, 1)
    rotated_fragment2 = rotate_fragment(fragments, side2, 2)


    new_fragment = np.hstack((rotated_fragment1, rotated_fragment2))
    return new_fragment


def merge_fragments_two_by_two(fragments: List[Fragment], sides_comparisons: List[SidesComparison]):
    banned_fragments_idx = []
    new_fragments = []
    new_fr_idx = 0
    for comp in sides_comparisons:
        if comp.side1.fragment_idx not in banned_fragments_idx and comp.side2.fragment_idx not in banned_fragments_idx:
            new_fragment_value = two_fragments_merger(fragments, comp.side1, comp.side2)

            new_fragment = Fragment(new_fragment_value, new_fr_idx)
            new_fragments.append(new_fragment)
            banned_fragments_idx.append(comp.side1.fragment_idx)
            banned_fragments_idx.append(comp.side2.fragment_idx)
            new_fr_idx += 1
    return new_fragments








