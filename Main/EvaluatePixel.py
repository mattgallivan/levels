import math
from PIL import Image


def evaluate(original_img, level_img):
    original_pixels = original_img.load()
    level_pixels = level_img.load()
    
    widthO, heightO = original_img.size
    widthL, heightL = level_img.size
    Y = min(heightO, heightL)
    X = min(widthO, widthL)
    Z = max(len(original_pixels[0,0]), len(level_pixels[0,0]))
    
    sumVal = 0
    for x in range(0,X):
        for y in range(0,Y):
            for z in range(0,Z):
                sumVal += (original_pixels[x,y][z] - level_pixels[x,y][z])**2

    sumVal = math.sqrt(sumVal/(X*Y*Z))
    return sumVal


#def evaluate_cv(original_img, level_img):
    #original_pixels = original_img
    #level_pixels = level_img
    
    #(rowO, columnO, rgbO) = original_img.shape
    #(rowL, columnL, rgbL) = level_img.shape
    #Y = min(rowO, rowL)
    #X = min(columnO, columnL)
    #Z = max(rgbO, rgbL)
    
    #sumVal = 0
    #for x in range(0,X):
        #for y in range(0,Y):
            #for z in range(0,Z):
                #sumVal += abs(original_pixels[x, y][z] - level_pixels[x, y][z])  # **2

    ##sumVal = math.sqrt(sumVal / (X * Y * Z))
    #sumVal = sumVal / (X * Y * Z)
    #return sumVal