import cv2 as cv

def rotate_image(image, rotation):
    
    if rotation == 1:
        image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
    elif rotation == 2:
        image = cv.rotate(image, cv.ROTATE_180)
    elif rotation == 3:
        image = cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE)

    return image


def find_pasted_group_moving_distance_and_rotation(fragments, comp):  
    anchor_rotation = fragments[comp.side1.fragment_idx].rotation
    pasted_rotation = fragments[comp.side2.fragment_idx].rotation

    anchor_side_idx = comp.side1.side_idx
    pasted_side_idx = comp.side2.side_idx


    #### side_orientation is the normalization from side_idx to 0 - top, 1 - right etc
    anchor_side_orientation = (anchor_side_idx + anchor_rotation) % 4
    pasted_side_orientation = (pasted_side_idx + pasted_rotation) % 4

    row_dist = 0
    col_dist = 0
    pasted_group_additional_rotation = 0


    ## after normalizing so 0, 1, 2, 3 means top, right, bottom, left, find the additional new rotation for the pasted group. 
    # the whole group will bet rotated 90 clockwise for pasted_group_additional_rotation
    if anchor_side_orientation == 0:
        row_dist -= 1
        pasted_group_additional_rotation = (4 + 2 - pasted_side_orientation) % 4

    elif anchor_side_orientation == 1:
        col_dist += 1
        pasted_group_additional_rotation = (4 + 3 - pasted_side_orientation) % 4

    elif anchor_side_orientation == 2:
        row_dist += 1
        pasted_group_additional_rotation = (4 - pasted_side_orientation) % 4

    else:
        col_dist -= 1
        pasted_group_additional_rotation = (4 + 1 - pasted_side_orientation) % 4
    
    return row_dist, col_dist, pasted_group_additional_rotation


def rotate_fragments_positions(group_to_rotate, rotation):

    H = group_to_rotate.row_nr
    W = group_to_rotate.col_nr

    for fr_idx, (row, col) in group_to_rotate.fragment_positions.items():

        if rotation == 1:  
            group_to_rotate.fragment_positions[fr_idx] = [col, H - 1 - row] 
        elif rotation == 2:
            group_to_rotate.fragment_positions[fr_idx] = [H - 1 - row, W - 1 - col] 
        elif rotation == 3: 
            group_to_rotate.fragment_positions[fr_idx] = [W - 1 - col, row] 
        elif rotation == 0:
            continue
        else:
            print("rotation not between 0-3")
    return group_to_rotate
