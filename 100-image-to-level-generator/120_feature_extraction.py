import numpy as np
import cv2
from pathlib import Path

def extract_features(img_meta, opts = None):
    
    output = {}
    output['img_in_tiles_by_size'] = turn_into_tiles(img_meta['raw_img_data']['opencv'])

    save_tiles_as_images_for_debugging(output, img_meta)

    return output


def save_tiles_as_images_for_debugging(output, img_meta):

    output_base_path = "./output/"
    output_path = output_base_path + img_meta["file_info"]["addition_from_start_path"]

    if (img_meta["file_info"]["addition_from_start_path"] != ""):
        output_path += "/"

    output_path += img_meta["file_info"]["filename"] + "/"
    output_path += "tiles/"

    # Make the output path first if it doesn't exist.
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Write the tiles as images!
    for tile_size in output['img_in_tiles_by_size']:
        r,c = 0,0
        for row_of_tiles in output['img_in_tiles_by_size'][tile_size]:
            for tile in row_of_tiles:
                cv2.imwrite( output_path + str(tile_size) 
                        + "_" + str(r) + "x" 
                        + str(c) + ".png", cv2.cvtColor(tile, cv2.COLOR_RGB2BGR) )
                c += 1
            r += 1

def turn_into_tiles(opencv_img, opts = None):

    tiles_by_size = {}

    if opts is None:
        opts = {
            "tile_sizes": [16, 32, 100]
        }

    (r_max, c_max, rgb_max) = opencv_img.shape

    for tile_size in opts["tile_sizes"]:
        tiles_by_size[tile_size] = []
        for r in range(0, r_max, tile_size):
            r_arr = []
            for c in range(0, c_max, tile_size):
                r_arr.append(opencv_img[r:(r+tile_size), c:(c+tile_size)])
            tiles_by_size[tile_size].append(r_arr)

    return tiles_by_size
