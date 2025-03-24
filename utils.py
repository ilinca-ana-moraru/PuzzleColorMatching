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



def find_centroid(image):
    binary_mask =  (image[:, :, 3])
    moments = cv.moments(binary_mask)  
    if moments["m00"] != 0:
        cx = int(moments["m10"] // moments["m00"])  
        cy = int(moments["m01"] // moments["m00"]) 
    else:
        cx = 0
        cy = 0
    return cx, cy


from PIL import Image

def fix_border(image):

    width, height = image.shape[:2]

    for x in range(1, width - 1):
        image[x, 0] = image[x, 1]  
    
    for x in range(1, width - 1):
        image[x, height - 1] = image[x, height - 2]  

    for y in range(1, height - 1):
        image[0, y] = image[1, y] 
    
    for y in range(1, height - 1):
        image[width - 1, y] = image[width - 2, y] 

    return image