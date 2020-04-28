import math
from PIL import Image


def evaluate(original_img, level_img):
    original_pixels = original_img.load()
    level_pixels = level_img.load()
    
    widthO, heightO = original_img.size
    widthL, heightL = level_img.size
    smallerY = min(heightO, heightL)
    smallerX = min(widthO, widthL)
    
    sumVal = 0
    counter = 0
    for x in range(0,smallerX):
        for y in range(0,smallerY):
            qqq = original_pixels[x,y]
            sumVal += (original_pixels[x,y][0] - level_pixels[x,y][0])**2
            sumVal += (original_pixels[x,y][1] - level_pixels[x,y][1])**2
            sumVal += (original_pixels[x,y][2] - level_pixels[x,y][2])**2
            counter+=1
            
    sumVal = math.sqrt(sumVal/counter)
    return sumVal