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


def get_average_pix(image):
    height, width, channels = image.shape
    avrg = [0, 0, 0]
    for x2 in range(0, width):
	    for y2 in range(0, height):
		    pixelVal = image[x2,y2]
		    avrg = [sum(x) for x in zip(avrg, pixelVal)] 
    
    tile_size = height*width
    avrg = [int(avrg[0]/tile_size), int(avrg[1]/tile_size), int(avrg[2]/tile_size)]
    return avrg


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
    sprites_avrg = {}
    for sprite_name in sprites:
        sprite_img = sprites[sprite_name][1]
        sprite_img_cv = np.array(sprite_img)
        sprites_images[sprite_name] = sprite_img
        sprites_histogram[sprite_name] = get_histogram_of_image(sprite_img_cv)
        sprites_avrg[sprite_name] = get_average_pix(sprite_img_cv)
        
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
                    #results[sprite_name] = EvaluatePixel.evaluate_cv(tile, sprite_img)
                results = sorted([(v, k) for (k, v) in results.items()], reverse=False)
            if asset_type == 'histogram':
                tile_histogram = get_histogram_of_image(tile)
                for sprite_name in sprites_histogram:
                    sprite_histogram = sprites_histogram[sprite_name]
                    results[sprite_name] = cv2.compareHist(tile_histogram, sprite_histogram, cv2.HISTCMP_BHATTACHARYYA)
                    #HISTCMP_CORREL #HISTCMP_BHATTACHARYYA
                results = sorted([(v, k) for (k, v) in results.items()], reverse = False)
            if asset_type == 'avrg':
                tile_avrg = get_average_pix(tile)
                for sprite_name in sprites_avrg:
                    sprite_avrg = sprites_avrg[sprite_name]
                    results[sprite_name] = abs(tile_avrg[0] - sprite_avrg[0]) + abs(tile_avrg[1] - sprite_avrg[1]) + abs(tile_avrg[2] - sprite_avrg[2])
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