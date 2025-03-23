import os
import cv2 as cv
from fragment import *

def divide_image(image_path, output_folder, n, m):
    os.makedirs(output_folder, exist_ok=True)
    rgb_image = cv.imread(image_path, cv.IMREAD_COLOR)  
    rgb_image = rgb_image[..., ::-1]
    rgba_image = cv.cvtColor(rgb_image, cv.COLOR_BGR2BGRA)
    h, w = rgba_image.shape[:2] 

    tile_h, tile_w = h // n, w // m  

    fragments = []

    for i in range(n):
        for j in range(m):
            x, y = j * tile_w, i * tile_h  
            cropped_fragment = rgba_image[y:y + tile_h, x:x + tile_w]  
            
            fragment_path = os.path.join(output_folder, f"fragment_{i*m + j}.jpg")
            cv.imwrite(fragment_path, cropped_fragment[..., [2, 1, 0, 3]])
            fragment = Fragment(cropped_fragment, i*m + j)
            # print(fragment.contour)
            # print("-------------------------------------------")
            fragments.append(fragment)

    return fragments  