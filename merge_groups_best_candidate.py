from groups import *

def calculate_all_group_matchings_scores(fragments, pasted_group_additional_rotation, shifted_anchor_group: Group, shifted_pasted_group: Group):
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
                    total_score += neighbor_comp.score
                    total_matchings += 1

    if total_matchings == 0:
        return False
    average_score = total_score / total_matchings
    return average_score, total_matchings


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



def find_best_candidate_for_empty_spot(fragments, row, groups):
    anchor_group_idx = row['group_idx']
    empty_row, empty_col = row['row'], row['col']
    anchor_group = groups[anchor_group_idx]

    neighbours = [
        anchor_group.grid[empty_row - 1][empty_col] if empty_row > 0 else None,
        anchor_group.grid[empty_row][empty_col + 1] if empty_col + 1 < anchor_group.col_nr else None,
        anchor_group.grid[empty_row + 1][empty_col] if empty_row + 1 < anchor_group.row_nr else None,
        anchor_group.grid[empty_row][empty_col - 1] if empty_col > 0 else None
    ]

    best_score = float('inf')
    best_comp = None
    best_fragment_idx = None
    best_pasted_group_idx = None
    best_pasted_group_additional_rotation = None
    best_total_matchings = None
    for pasted_group_idx, pasted_group in enumerate(groups):
        if pasted_group_idx == anchor_group_idx:
            continue

        for pasted_fr_idx in pasted_group.used_fragments:
            for pasted_additional_rotation in range(0,4):
                if neighbours[0] is not None:
                    neighbours_side_idx = find_side_idx_of_orientation(fragments[neighbours[0]].rotation,2)
                    comp = get_comparison(neighbours[0], pasted_fr_idx, neighbours_side_idx, pasted_additional_rotation)
                elif neighbours[1] is not None:
                    neighbours_side_idx = find_side_idx_of_orientation(fragments[neighbours[1]].rotation,3)
                    comp = get_comparison(neighbours[1], pasted_fr_idx, neighbours_side_idx, pasted_additional_rotation)
                elif neighbours[2] is not None:
                    neighbours_side_idx = find_side_idx_of_orientation(fragments[neighbours[2]].rotation,0)
                    comp = get_comparison(neighbours[2], pasted_fr_idx, neighbours_side_idx, pasted_additional_rotation)
                elif neighbours[3] is not None:
                    neighbours_side_idx = find_side_idx_of_orientation(fragments[neighbours[3]].rotation,1)
                    comp = get_comparison(neighbours[3], pasted_fr_idx, neighbours_side_idx, pasted_additional_rotation)
                else:
                    continue
                if comp:
                    shifted_anchor_group, shifted_pasted_group, pasted_group_additional_rotation = simulate_merge_positions(fragments, comp, groups[anchor_group_idx], groups[pasted_group_idx])
                    if does_merge_fit_within_bounds(shifted_anchor_group):
                        if check_groups_shapes_for_merging(shifted_anchor_group, shifted_pasted_group):

                            score, total_matchings = calculate_all_group_matchings_scores(fragments, pasted_group_additional_rotation, shifted_anchor_group, shifted_pasted_group)
                            if score:
                                if score < best_score:
                                    best_score = score
                                    best_comp = comp
                                    best_fragment_idx = pasted_fr_idx
                                    best_pasted_group_idx = pasted_group_idx
                                    best_pasted_group_additional_rotation = pasted_group_additional_rotation
                                    best_total_matchings = total_matchings

    if best_comp:   
        return {
            'anchor_group_idx': anchor_group_idx,
            'empty_spot_neighbours': neighbours,
            'pasted_group_idx': best_pasted_group_idx,
            'fragment_idx': best_fragment_idx,
            'score': best_score,
            'comp': best_comp,
            'pasted_group_additional_rotation':best_pasted_group_additional_rotation,
            'total_matchings': best_total_matchings

            }
    return None



def solve_groups_safe(groups, fragments, fragment_idx_to_group_idx):



    while len(groups) > 1:
        edges_of_groups_df = edges_of_groups(groups)
        if edges_of_groups_df.empty:
            print("No empty spots with neighbours left.")
            break
        #### more neighbours first, then score
        merge_candidates = []
        max_neighbours = edges_of_groups_df['nr_of_neighbours'][0] + 1
        while not merge_candidates and max_neighbours >= 2:
            # print(f"\n looking at merging with {max_neighbours} neighbours")
            max_neighbours -= 1
            for _, row in edges_of_groups_df.iterrows():
                if row['nr_of_neighbours'] == max_neighbours:
                    candidate = find_best_candidate_for_empty_spot(fragments, row, groups)
                    if candidate is not None:
                        merge_candidates.append(candidate)
        if not merge_candidates:
            print("No valid merge candidates found.")
            break

        merge_candidates.sort(key=lambda c: c['score'])
        # print([round(c['score'], 6) for c in merge_candidates])
        best = merge_candidates.pop(0)
        comp = best['comp']
        anchor_group_idx = best['anchor_group_idx']
        pasted_group_idx = best['pasted_group_idx']
        print(f"Merged group {anchor_group_idx} and {pasted_group_idx} with total score: {best['score']} using: {comp}")
        shifted_anchor_group, shifted_pasted_group, pasted_group_additional_rotation = simulate_merge_positions(fragments, comp, groups[anchor_group_idx], groups[pasted_group_idx])
        groups[anchor_group_idx] = merge_groups(fragments, pasted_group_additional_rotation, shifted_anchor_group, shifted_pasted_group, fragment_idx_to_group_idx)
        update_after_merge(groups, fragments, fragment_idx_to_group_idx, pasted_group_idx)
        # show_all_groups(groups, fragments, fragment_idx_to_group_idx, 0)

    return groups, fragments, fragment_idx_to_group_idx

########################################################################################################
### reparare


def undo_small_groups(groups, fragment_idx_to_group_idx):
    biggest_gr = None
    biggest_gr_idx = None
    for g_idx, g in enumerate(groups):
        if biggest_gr_idx is None or biggest_gr < len(g.used_fragments):
            biggest_gr = len(g.used_fragments)
            biggest_gr_idx = g_idx
    print(f"biggest_group: {biggest_gr_idx} with {biggest_gr} fragments")

    if biggest_gr == 1:
        return groups, fragment_idx_to_group_idx, biggest_gr_idx

    new_groups = []
    new_fragment_idx_to_group_idx = {}
    new_biggest_group_idx = 0  

    new_groups.append(groups[biggest_gr_idx])
    for fr_idx in groups[biggest_gr_idx].used_fragments:
        new_fragment_idx_to_group_idx[fr_idx] = 0

    for g_idx, g in enumerate(groups):
        if g_idx != biggest_gr_idx:
            for fr_idx in g.used_fragments:
                new_group = Group(fr_idx)
                new_groups.append(new_group)
                new_fragment_idx_to_group_idx[fr_idx] = len(new_groups) - 1

    return new_groups, new_fragment_idx_to_group_idx, new_biggest_group_idx




def edges_of_groups_repair(groups, biggest_group_idx):
    data = []

    group_idx = biggest_group_idx
    group = groups[group_idx]

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



def find_best_candidate_for_empty_spot_repair(fragments, row, groups, biggest_group_idx):
    anchor_group_idx = row['group_idx']
    empty_row, empty_col = row['row'], row['col']
    anchor_group = groups[anchor_group_idx]

    neighbours = [
        anchor_group.grid[empty_row - 1][empty_col] if empty_row > 0 else None,
        anchor_group.grid[empty_row][empty_col + 1] if empty_col + 1 < anchor_group.col_nr else None,
        anchor_group.grid[empty_row + 1][empty_col] if empty_row + 1 < anchor_group.row_nr else None,
        anchor_group.grid[empty_row][empty_col - 1] if empty_col > 0 else None
    ]

    best_score = float('inf')
    best_comp = None
    best_fragment_idx = None
    best_pasted_group_idx = None
    best_pasted_group_additional_rotation = None
    best_total_matchings = None
    for pasted_group_idx, pasted_group in enumerate(groups):
        if pasted_group_idx == anchor_group_idx:
            continue

        for pasted_fr_idx in pasted_group.used_fragments:
            for pasted_additional_rotation in range(0,4):
                if neighbours[0] is not None:
                    neighbours_side_idx = find_side_idx_of_orientation(fragments[neighbours[0]].rotation,2)
                    comp = get_comparison(neighbours[0], pasted_fr_idx, neighbours_side_idx, pasted_additional_rotation)
                elif neighbours[1] is not None:
                    neighbours_side_idx = find_side_idx_of_orientation(fragments[neighbours[1]].rotation,3)
                    comp = get_comparison(neighbours[1], pasted_fr_idx, neighbours_side_idx, pasted_additional_rotation)
                elif neighbours[2] is not None:
                    neighbours_side_idx = find_side_idx_of_orientation(fragments[neighbours[2]].rotation,0)
                    comp = get_comparison(neighbours[2], pasted_fr_idx, neighbours_side_idx, pasted_additional_rotation)
                elif neighbours[3] is not None:
                    neighbours_side_idx = find_side_idx_of_orientation(fragments[neighbours[3]].rotation,1)
                    comp = get_comparison(neighbours[3], pasted_fr_idx, neighbours_side_idx, pasted_additional_rotation)
                else:
                    continue
                if comp:
                    shifted_anchor_group, shifted_pasted_group, pasted_group_additional_rotation = simulate_merge_positions(fragments, comp, groups[anchor_group_idx], groups[pasted_group_idx])
                    if does_merge_fit_within_bounds(shifted_anchor_group):
                        if check_groups_shapes_for_merging(shifted_anchor_group, shifted_pasted_group):

                            score, total_matchings = calculate_all_group_matchings_scores(fragments, pasted_group_additional_rotation, shifted_anchor_group, shifted_pasted_group)
                            if score:
                                if score < best_score:
                                    best_score = score
                                    best_comp = comp
                                    best_fragment_idx = pasted_fr_idx
                                    best_pasted_group_idx = pasted_group_idx
                                    best_pasted_group_additional_rotation = pasted_group_additional_rotation
                                    best_total_matchings = total_matchings

    if best_comp:   
        return {
            'anchor_group_idx': anchor_group_idx,
            'empty_spot_neighbours': neighbours,
            'pasted_group_idx': best_pasted_group_idx,
            'fragment_idx': best_fragment_idx,
            'score': best_score,
            'comp': best_comp,
            'pasted_group_additional_rotation':best_pasted_group_additional_rotation,
            'total_matchings': best_total_matchings

            }
    return None

def solve_groups_safe_reparing(groups, fragments, fragment_idx_to_group_idx, biggest_group_idx):

    while len(groups) > 1:
        edges_of_groups_df = edges_of_groups_repair(groups, biggest_group_idx)
        if edges_of_groups_df.empty:
            print("No empty spots with neighbours left.")
            break

        merge_candidates = []
        max_neighbours = edges_of_groups_df['nr_of_neighbours'][0] + 1
        while not merge_candidates and max_neighbours >= 2:
            max_neighbours -= 1
            for _, row in edges_of_groups_df.iterrows():
                if row['nr_of_neighbours'] == max_neighbours:
                    candidate = find_best_candidate_for_empty_spot_repair(fragments, row, groups, biggest_group_idx)
                    if candidate is not None:
                        if candidate['pasted_group_idx'] == biggest_group_idx:
                            continue
                        merge_candidates.append(candidate)

        if not merge_candidates:
            print("No valid merge candidates found.")
            break

        merge_candidates.sort(key=lambda c: c['score'])
        best = merge_candidates.pop(0)
        comp = best['comp']
        anchor_group_idx = best['anchor_group_idx']
        pasted_group_idx = best['pasted_group_idx']
        print(f"Merged group {anchor_group_idx} and {pasted_group_idx} with total score: {best['score']} using: {comp}")

        shifted_anchor_group, shifted_pasted_group, pasted_group_additional_rotation = simulate_merge_positions(
            fragments, comp, groups[anchor_group_idx], groups[pasted_group_idx]
        )
        groups[anchor_group_idx] = merge_groups(fragments, pasted_group_additional_rotation, shifted_anchor_group, shifted_pasted_group, fragment_idx_to_group_idx)
        update_after_merge(groups, fragments, fragment_idx_to_group_idx, pasted_group_idx)
        show_all_groups(groups, fragments, fragment_idx_to_group_idx, 0)
        print(f"biggest group: {biggest_group_idx}")
    return groups, fragments, fragment_idx_to_group_idx

##########################################################################################################

def update_edges_of_groups(groups, edges_of_groups_df, pasted_group_idx, anchor_group_idx):
    ## save empty_spots of unmerging groups
    new_edges_of_groups = []

    for _, e in edges_of_groups_df.iterrows():
        if e['group_idx'] != pasted_group_idx and e['group_idx'] != anchor_group_idx:
            if e['group_idx'] > pasted_group_idx:
                e['group_idx'] -= 1

            new_edges_of_groups.append({
                'group_idx': e['group_idx'],
                'nr_of_neighbours': e['nr_of_neighbours'],
                'row': e['row'],
                'col': e['col']
            })

    
    if len(groups) > anchor_group_idx:
        group = groups[anchor_group_idx]
        rows, cols = len(group.grid), len(group.grid[0])
        for i in range(rows):
            for j in range(cols):
                if group.grid[i][j] is None:
                    neighbour_count = group.neighbours_grid[i][j]
                    new_edges_of_groups.append({
                        'group_idx': anchor_group_idx,
                        'nr_of_neighbours': neighbour_count,
                        'row': i,
                        'col': j
                    })

    
    if new_edges_of_groups:
        new_edges_of_groups = pd.DataFrame(new_edges_of_groups)
        new_edges_of_groups = new_edges_of_groups.sort_values(by='nr_of_neighbours', ascending=False)
    else:
        new_edges_of_groups = pd.DataFrame(columns=['group_idx', 'nr_of_neighbours', 'row', 'col'])

    return new_edges_of_groups

def update_merge_candidates(fragments, groups, new_edges_of_groups, merge_candidates, pasted_group_idx, anchor_group_idx):

    # filter out merge candidates that involve either of the merged groups, and update to correct idx
    new_merge_candidates = []
    for c in merge_candidates:
        if c['anchor_group_idx'] != pasted_group_idx and c['pasted_group_idx'] != pasted_group_idx:
            if c['anchor_group_idx'] != anchor_group_idx and c['pasted_group_idx'] != anchor_group_idx:
                if c['anchor_group_idx'] > pasted_group_idx:
                    c['anchor_group_idx'] -= 1
                if c['pasted_group_idx'] > pasted_group_idx:
                    c['pasted_group_idx'] -= 1
                new_merge_candidates.append(c)


    for _, row in new_edges_of_groups.iterrows():
        best_candidate = find_best_candidate_for_empty_spot(fragments, row, groups)
        if best_candidate is not None:
            new_merge_candidates.append(best_candidate)

    return new_merge_candidates


def solve_groups(groups, fragments, fragment_idx_to_group_idx):
    edges_of_groups_df = edges_of_groups(groups)
    if edges_of_groups_df.empty:
        print("No empty spots with neighbours left.")
        return

    merge_candidates = []
    for _, row in edges_of_groups_df.iterrows():
        candidate = find_best_candidate_for_empty_spot(fragments, row, groups)
        if candidate is not None:
            merge_candidates.append(candidate)

    while len(groups) > 1:
        
        if not merge_candidates:
            print("No valid merge candidates found.")
            break

        while merge_candidates:
            max_neighbour = edges_of_groups_df['nr_of_neighbours'].max()
            best_candidates = [c for c in merge_candidates if sum(1 for x in c['empty_spot_neighbours'] if x is not None) == int(max_neighbour)]
            best = min(best_candidates, key=lambda c: c['score'])


            merge_candidates.remove(best)
            comp = best['comp']
            anchor_group_idx = best['anchor_group_idx']
            pasted_group_idx = best['pasted_group_idx']

            print(f"Merged group {anchor_group_idx} and {pasted_group_idx} with total score: {best['score']} using: {comp}")

            shifted_anchor_group, shifted_pasted_group, pasted_group_additional_rotation = simulate_merge_positions(fragments, comp, groups[anchor_group_idx], groups[pasted_group_idx])
            groups[anchor_group_idx] = merge_groups(fragments, pasted_group_additional_rotation, shifted_anchor_group, shifted_pasted_group, fragment_idx_to_group_idx)

            update_after_merge(groups, fragments, fragment_idx_to_group_idx, pasted_group_idx)
            edges_of_groups_df = edges_of_groups(groups)

            # edges_of_groups_df = update_edges_of_groups(groups, edges_of_groups_df, pasted_group_idx, anchor_group_idx)
            merge_candidates = update_merge_candidates(fragments, groups, edges_of_groups_df, merge_candidates, pasted_group_idx, anchor_group_idx)
            show_all_groups(groups, fragments, fragment_idx_to_group_idx, 0)

 
    return groups, fragments, fragment_idx_to_group_idx
