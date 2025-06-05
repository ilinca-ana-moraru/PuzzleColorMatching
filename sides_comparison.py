from side import * 
import numpy as np
import global_values 
from scipy.special import erf
from rotation import *
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List
from fragment import *





class SidesComparison:
    def __init__(self, side1 : Side, side2: Side, score = None):
        self.side1 = side1
        self.side2 = side2
        self.model = global_values.MODEL
        self.device = global_values.DEVICE
        self.buddy_score = None

        if global_values.GRAD_SCORING == True:
            self.grad_scoring()
        else:
            self.score = score


    def grad_scoring(self):

        self.reversed_side1_value = self.side1.value[::-1]
        self.color_points_distances = abs(self.reversed_side1_value - self.side2.value)
        color_score = self.color_points_distances/ 255

        if global_values.DIFF_GBR == True:
            grayscale_weights = 3 * np.array([0.2989, 0.5870, 0.1140])
            color_score *= grayscale_weights
            color_score =np.linalg.norm(color_score, axis = 1)

        if global_values.DIFF_GRAY == True:
            color_score =np.linalg.norm(color_score, axis = 1)


        self.color_score = erf(4 * color_score - 2)/2 + 0.5 ## input[0,1] -> [-2, 2] output[-1,1] -> [0,1]
        self.color_score = np.sum(color_score)/len(self.side1.value)



        self.reversed_side1_grad = self.side1.grad[::-1]
        grad_match = (self.reversed_side1_grad - self.side2.grad)
        grad_match = erf(4 * grad_match - 2)/2 + 0.5  ## input[0,1] -> [-2, 2] output[-1,1] -> [0,1]
        # grad_match[grad_match < 0.2] = 0.0
        self.grad_match = np.sum(grad_match)/len(self.side1.value)



        self.grad_presence = np.sum(erf(4 * self.reversed_side1_grad - 2)/2 + 0.5 + erf(4 * self.side2.grad - 2)/2 + 0.5)/len(self.side1.value) * 50

        self.grad_score = self.grad_match/ (self.grad_presence + 0.000001)

        self.score = 1/(self.grad_presence + 0.000001)* np.sqrt(self.color_score**2 + self.grad_match**2)


        # print(f"color score: {self.color_score} grad score: {self.grad_score} grad match: {self.grad_match} grad presence: {self.grad_presence}")
        self.prudent_score = self.score
        for i in color_score:
            if i > 0.3:
                self.prudent_score *= 5
        # for i in grad_match:
        #     if i > 0.3:
        #         self.score *= 3


        # self.DLR, self.DRL =  mahalanobis_merger(self,fragments)
        # self.score = self.DLR + self.DRL


    def __str__(self):
        return (f"Sides Comp: Score={self.score} Buddy_Score:{self.buddy_score} Fragment_idx1={self.side1.fragment_idx}, Side_idx1={self.side1.side_idx}; fragment_idx2={self.side2.fragment_idx}, side_idx2={self.side2.side_idx}")
    



def preprocess_for_model(img):
    img = img[:, :, :3] 
    img = cv.resize(img, (64, 64))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img, dtype=torch.float32)


def extract_tile(rotated_image, row, col, TILE_H, TILE_W):
    y_start = row * TILE_H
    y_end = y_start + TILE_H
    x_start = col * TILE_W
    x_end = x_start + TILE_W

    return rotated_image[y_start:y_end, x_start:x_end, :]


def create_fragment_rotation_dictionary(fragments, tile_h, tile_w):

    full_h = global_values.ROW_NR * tile_h
    full_w = global_values.COL_NR * tile_w

    full_image = np.zeros((full_h, full_w, 3), dtype=fragments[0].value.dtype)

    for idx, frag in enumerate(fragments):
        row = idx // global_values.COL_NR
        col = idx % global_values.COL_NR

        y_start = row * tile_h
        y_end = y_start + tile_h
        x_start = col * tile_w
        x_end = x_start + tile_w

        full_image[y_start:y_end, x_start:x_end, :] = frag.value[:,:,:3]

    rotated_images = []
    for r in range(4):
        rotated = rotate_image(full_image, r)
        rotated_images.append(rotated)

    fr_rotation_dict = {}

    for idx in tqdm(range(len(fragments))):
        row = idx // global_values.COL_NR
        col = idx % global_values.COL_NR

        rotations = []
        for r in rotated_images:
            tile = extract_tile(r, row, col, tile_h, tile_w)
            rotations.append(tile)

        fr_rotation_dict[idx] = rotations

    return fr_rotation_dict


def create_fragment_rotation_tensor_dictionary(fragment_rotation_dictionary):
    fr_rotation_tensor_dict = {}

    for fr_idx, rotations in fragment_rotation_dictionary.items():
        tensor_rotations = []
        for img in rotations:
            tensor = preprocess_for_model(img)
            tensor_rotations.append(tensor)
        fr_rotation_tensor_dict[fr_idx] = tensor_rotations

    return fr_rotation_tensor_dict


def create_sides_comparisons(fragments: List[Fragment]):
    sides_comparisons = []
    for fr_idx1 in tqdm(range(len(fragments) - 1)):
        for side_idx1 in range(len(fragments[fr_idx1].sides)):
            side1 = fragments[fr_idx1].sides[side_idx1]
            
            if all(len(side1.value) >= len(fragments[fr_idx1].sides[side_idx].value) for side_idx in range(len(fragments[fr_idx1].sides))):
                for fr_idx2 in range(fr_idx1 + 1, len(fragments)):
                    for side_idx2 in range(len(fragments[fr_idx2].sides)):
                        side2 = fragments[fr_idx2].sides[side_idx2]
                        if len(side1.value) == len(side2.value):
                            if  global_values.ROTATING_PIECES or ((side1.side_idx == 2 and side2.side_idx == 0) or (side1.side_idx == 1 and side2.side_idx == 3) \
                            or (side1.side_idx == 0 and side2.side_idx == 2) or (side1.side_idx == 3 and side2.side_idx == 1)):
                                sides_comparisons.append(SidesComparison(fragments, side1, side2))
                                # print(f"fragment {fr_idx1} side {side_idx1} VS fragment {fr_idx2} side {side_idx2}")

    
    return sides_comparisons  

def create_sides_comparisons_with_score_nn(model, device, fragment_rotation_tensor_dictionary, pairs_list, batch_size=800):
    model.eval()
    all_scores = []
    sides = []

    with torch.no_grad():
        # Loop over batches of pairs
        for i in tqdm(range(0, len(pairs_list), batch_size)):
            batch_pairs = pairs_list[i:i+batch_size]

            fr1_batch = []
            fr2_batch = []

            # Prepare this batch only
            for side1, side2 in batch_pairs:
                nr_of_fr1_rotations = (4 + 1 - side1.side_idx) % 4
                nr_of_fr2_rotations = (4 + 3 - side2.side_idx) % 4

                # No preprocess_for_model anymore → just take precomputed tensors!
                fr1_tensor = fragment_rotation_tensor_dictionary[side1.fragment_idx][nr_of_fr1_rotations]
                fr2_tensor = fragment_rotation_tensor_dictionary[side2.fragment_idx][nr_of_fr2_rotations]

                # Flip second tensor
                fr2_tensor = torch.flip(fr2_tensor, dims=[2])

                fr1_batch.append(fr1_tensor)
                fr2_batch.append(fr2_tensor)
                sides.append( (side1, side2) )

            # Stack current batch
            fr1_batch = torch.stack(fr1_batch).to(device)
            fr2_batch = torch.stack(fr2_batch).to(device)

            # Forward pass
            output = model(fr1_batch, fr2_batch).squeeze()
            scores = torch.sigmoid(output).cpu().numpy()
            all_scores.extend(scores)

    # Return list of (side1, side2, score)
    results = []
    for idx, score in enumerate(all_scores):
        results.append( (sides[idx][0], sides[idx][1], score) )
    
    sides_comparisons = []
    for side1, side2, score in results:
        sc = SidesComparison(side1, side2, score)
        sides_comparisons.append(sc)

    return sides_comparisons



def create_sides_pairs_for_nn(fragments: List[Fragment]):
    pairs_list = []
    for fr_idx1 in tqdm(range(len(fragments) - 1)):
        for side_idx1 in range(len(fragments[fr_idx1].sides)):
            side1 = fragments[fr_idx1].sides[side_idx1]
            
            if all(len(side1.value) >= len(fragments[fr_idx1].sides[side_idx].value) for side_idx in range(len(fragments[fr_idx1].sides))):
                for fr_idx2 in range(fr_idx1 + 1, len(fragments)):
                    for side_idx2 in range(len(fragments[fr_idx2].sides)):
                        side2 = fragments[fr_idx2].sides[side_idx2]
                        if len(side1.value) == len(side2.value):
                            if  global_values.ROTATING_PIECES or ((side1.side_idx == 2 and side2.side_idx == 0) or (side1.side_idx == 1 and side2.side_idx == 3) \
                            or (side1.side_idx == 0 and side2.side_idx == 2) or (side1.side_idx == 3 and side2.side_idx == 1)):
                                pairs_list.append((side1, side2))
    
    return pairs_list




def calculate_buddy_score(fragments,sides_comparisons):
    best_score = [[None for _ in range(4)] for _ in range(len(fragments))]
    for s in tqdm(sides_comparisons):
        if best_score[s.side1.fragment_idx][s.side1.side_idx] is None or s.score < best_score[s.side1.fragment_idx][s.side1.side_idx]:
            best_score[s.side1.fragment_idx][s.side1.side_idx] = s.score

        if best_score[s.side2.fragment_idx][s.side2.side_idx] is None or s.score < best_score[s.side2.fragment_idx][s.side2.side_idx]:
            best_score[s.side2.fragment_idx][s.side2.side_idx] = s.score

    for s in sides_comparisons:
        buddy1 = s.score/best_score[s.side1.fragment_idx][s.side1.side_idx]
        buddy2 = s.score/best_score[s.side2.fragment_idx][s.side2.side_idx]
        s.buddy_score = buddy1 + buddy2 - 1
    return sides_comparisons

def sort_sides_comparisons(sides_comparisons: List[SidesComparison]):
        return sorted(sides_comparisons, key=lambda x: x.score)







class SiameseCNN(nn.Module):
    def __init__(self):
        super(SiameseCNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # Input: 3x64x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: 64x32x32

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: 128x16x16

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)   # Output: 256x8x8
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )


    def forward(self, piece1, piece2):
        feat1 = self.feature_extractor(piece1)
        feat2 = self.feature_extractor(piece2)

        diff = torch.abs(feat1 - feat2)
        diff = diff.view(diff.size(0), -1)

        out = self.classifier(diff)
        return out



