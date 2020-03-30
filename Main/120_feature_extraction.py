import numpy as np
import cv2
from pathlib import Path

def extract_features(img_meta, opts = None):
    
    output = {}
    output['img_for_tile_by_tilesize'] = turn_into_tiles(img_meta['raw_img_data']['opencv'])
    output['histogram_for_tile_by_tilesize'] = turn_tiles_into_histogram(output['img_for_tile_by_tilesize'])

    if opts is None or (opts is not None and opts['output'] is not None and opts['output']['save_tiles'] is not False):
        save_tiles_as_images_for_debugging(output, img_meta)

    return output

def turn_tiles_into_histogram(img_for_tile_by_tilesize):

    histogram_for_tile_by_tilesize = {}

    for tile_size in img_for_tile_by_tilesize:
        histogram_for_tile_by_tilesize[tile_size] = []
        for row_of_tiles in img_for_tile_by_tilesize[tile_size]:
            row = []
            for image in row_of_tiles:
                row.append(get_histogram_of_image(image))
            histogram_for_tile_by_tilesize[tile_size].append(row) 
    return histogram_for_tile_by_tilesize


def get_histogram_of_image(image):
    histogram = cv2.calcHist(image, [0,1,2], None, [8,8,8],
		                [0,256,0,256,0,256])
    histogram = cv2.normalize(histogram, histogram).flatten()
    return histogram


def save_tiles_as_images_for_debugging(output, img_meta):

    output_path = img_meta["output_info"]["output_path"] + "tiles/"

    # Make the output path first if it doesn't exist.
    Path(output_path).mkdir(parents=True, exist_ok=True)
    Path(output_path +'stitched/').mkdir(parents=True, exist_ok=True)

    border_width = 2
    # Write the tiles as images!
    for tile_size in output['img_for_tile_by_tilesize']:
        r = 0
        rows_of_tiles_debug_images = []
        
        for row_of_tiles in output['img_for_tile_by_tilesize'][tile_size]:
            c = 0
            for tile in row_of_tiles:
                cv2.imwrite( output_path + str(tile_size) 
                        + "_" + str(r) + "x" 
                        + str(c) + ".png", tile) #cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
                c += 1
            r += 1

            row_of_tiles_debug_images = list(map(lambda tile: cv2.copyMakeBorder(tile, border_width, border_width, border_width, border_width, cv2.BORDER_CONSTANT, None, [255,255,255]), row_of_tiles))
            rows_of_tiles_debug_images.append(cv2.hconcat(row_of_tiles_debug_images))

        cv2.imwrite( output_path +'stitched/' + str(tile_size)+"px_debug_show_tilespacing.png", cv2.vconcat(rows_of_tiles_debug_images))
                

def turn_into_tiles(opencv_img, opts = None):

    tiles_by_size = {}

    if opts is None:
        opts = {
            "tile_sizes": [16, 32, 50]
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
