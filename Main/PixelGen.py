import json
import numpy as np
from pathlib import Path
import cv2
import os

def get_histogram_of_image(image):
    histogram = cv2.calcHist(image, [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    histogram = cv2.normalize(histogram, histogram).flatten()
    return histogram

def compare_game_tiles(tile_repr, sprite_repr, asset_type):
    results = {}
    for sprite_name in sprite_repr:
        sprite_img = sprite_repr[sprite_name]
        if asset_type == 'histogram':
            results[sprite_name] = cv2.compareHist(tile_repr, sprite_img, cv2.HISTCMP_BHATTACHARYYA) #HISTCMP_CORREL #HISTCMP_BHATTACHARYYA
        if asset_type == 'img':
            rmax = np.max([tile_repr.shape[0],sprite_img.shape[0]])
            #cmax = np.max([tile_repr.shape[1],sprite_img.shape[1]])
            cmax = 1
            max_shape = (rmax,cmax,3)
            tile_img = tile_repr
            
            if tile_repr.shape != max_shape:
                tile_img = np.resize(tile_repr, max_shape)
            if sprite_img.shape != max_shape:
                sprite_img =  np.resize(sprite_img, max_shape)

            results[sprite_name] = np.sum(np.absolute(np.array(tile_img) - np.array(sprite_img)))

    results = sorted([(v, k) for (k, v) in results.items()], reverse = False)
    if(results[0][1] != 'sky'):
        qqqqqqq = 0
    return results[0][1]

def get_best_tile_ASCII(tile_repr, sprite_repr, sprite_info, asset_type):
    best_sprite_name = compare_game_tiles(tile_repr, sprite_repr, asset_type)
    
    for sprite in sprite_info:
        spritList = sprite_info[sprite]
        for i in range(0, len(spritList)):
            if(best_sprite_name == spritList[i]):
                return sprite
    return ' '

def generate(img, sprites, spriteAsciiMap, tile_size = 16, asset_type = 'histogram'):
    opencv_img = img
    (r_max, c_max, rgb_max) = opencv_img.shape
    
    # extract_features
    # turn_into_tiles
    tiles = []
    for r in range(0, r_max, tile_size):
        r_arr = []
        for c in range(0, c_max, tile_size):
            r_arr.append(opencv_img[r:(r+tile_size), c:(c+tile_size)])
        tiles.append(r_arr)    
    
    # turn_tiles_into_histogram
    histogram_for_tile = []
    for row_of_tiles in tiles:
        row = []
        for image in row_of_tiles:
            row.append(get_histogram_of_image(image))
        histogram_for_tile.append(row)
    
    # convert_features_to_games     
    # turn_gaming_sprites_into_histograms 
    # Conversions utilizing histograms
    sprites_histogram = {}
    for sprite_name in sprites:
        sprite_img = sprites[sprite_name][1]
        sprite_img_cv = np.array(sprite_img) 
        sprite_img_cv = sprite_img_cv[:, :, ::-1].copy()          
        sprites_histogram[sprite_name] = get_histogram_of_image(sprite_img_cv)    
        
    # match_assets_into_game_levels
    image_output = []
    for row_of_tiles in histogram_for_tile:
        row = ''
        for tile_repr in row_of_tiles:
            row += (get_best_tile_ASCII(tile_repr, sprites_histogram, spriteAsciiMap, asset_type))
        image_output.append(row)
        
    return image_output