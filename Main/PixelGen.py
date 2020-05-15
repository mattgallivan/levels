import json
import numpy as np
from pathlib import Path
import cv2
import os
import EvaluatePixel

def get_histogram_of_image(image):
    histogram = cv2.calcHist(image, [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    histogram = cv2.normalize(histogram, histogram).flatten()
    return histogram

def generate(img, sprites, spriteAsciiMap, tile_size = 16, asset_type = 'histogram'):
    opencv_img = img
    (row_max, column_max, rgb_max) = opencv_img.shape
    
    # Convert to tiles
    tiles = []
    for row in range(0, row_max, tile_size):
        row_array = []
        for col in range(0, column_max, tile_size):
            row_array.append(opencv_img[row:(row+tile_size), col:(col+tile_size)])
        tiles.append(row_array)
    
    # Get histograms of sprites
    sprites_images = {}
    sprites_histogram = {}
    for sprite_name in sprites:
        sprite_img = sprites[sprite_name][1]
        sprite_img_cv = np.array(sprite_img)
        sprite_img_cv = sprite_img_cv[:, :, ::-1].copy()
        sprites_images[sprite_name] = sprite_img
        sprites_histogram[sprite_name] = get_histogram_of_image(sprite_img_cv)
        
    # Find best game ascii for each tile
    image_output = []
    for row_of_tiles in tiles:
        row = ''
        for tile in row_of_tiles:
            bestAscii = ''
            results = {}
            if asset_type == 'img':
                for sprite_name in sprites_images:
                    sprite_img = sprites_images[sprite_name]
                    tile = cv2.resize(tile, (tile_size,tile_size))
                    sprite_img = cv2.resize(sprite_img, (tile_size,tile_size))
                    results[sprite_name] = (np.square(tile - sprite_img)).mean(axis=None)
                    #results[sprite_name] = EvaluatePixel.evaluate(tile, sprite_img)
                results = sorted([(v, k) for (k, v) in results.items()], reverse = False)
            if asset_type == 'histogram':
                for sprite_name in sprites_histogram:
                    sprite_histogram = sprites_histogram[sprite_name]
                    tile_histogram = get_histogram_of_image(tile)
                    results[sprite_name] = cv2.compareHist(tile_histogram, sprite_histogram, cv2.HISTCMP_BHATTACHARYYA)
                    #HISTCMP_CORREL #HISTCMP_BHATTACHARYYA
                results = sorted([(v, k) for (k, v) in results.items()], reverse = False)
                    
            bestAsciiName = results[0][1]
            bestAscii = ' '
            for sprite in spriteAsciiMap:
                spritList = spriteAsciiMap[sprite]
                for i in range(0, len(spritList)):
                    if(bestAsciiName == spritList[i]):
                        bestAscii = sprite
            row += bestAscii
        image_output.append(row + "\n")
        
    return image_output