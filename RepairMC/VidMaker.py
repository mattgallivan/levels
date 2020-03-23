import numpy as np
from PIL import Image, ImageDraw
import cv2
import glob

def makeVid(images):
    img_array = []
    filename = "./temp.jpeg"
    for i in range(0,len(images)):
        img = images[i]
        img.save(filename, "JPEG")
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
     
         
    out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()