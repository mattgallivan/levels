import glob
import os
from PIL import Image


pixelSize = 16
level_width = 40
level_height = 14
level_start = 50

resize = False
crop = True

# Read in an image:
for filename in glob.glob('./input_images_and_levels/Mario_Levels/*.png'):
    imageName = os.path.splitext(os.path.basename(filename))[0]
    inputImage_pil=Image.open(filename)

    processString=""

    if(resize):
	    dsize=(pixelSize * level_width, pixelSize * level_height)
	    inputImage_pil=inputImage_pil.resize(dsize)
	    processString += "_Resize"
    if(crop):
	    left, top, right, bottom=level_start, 0, level_start + level_width, level_height
	    inputImage_pil=inputImage_pil.crop((pixelSize * left, pixelSize * top, pixelSize * right, pixelSize * bottom))
	    processString += "_Crop"

    w, h=inputImage_pil.size
    pixels=inputImage_pil.load()

    inputImage_pil.save("./input_images_and_levels/Perfect/" + imageName + processString + ".png", "PNG")

    processString += "_Sketch"
    for y in range(0, h, pixelSize):
	    for x in range(0, w, pixelSize):

		    # Go over each tile of the image and find the average color:
		    # pixelDict = {}
		    sumVal=[0, 0, 0]
		    for x2 in range(0, pixelSize):
			    for y2 in range(0, pixelSize):
				    pixelVal=pixels[x + x2, y + y2]
				    sumVal=[sum(x) for x in zip(sumVal, pixelVal)]
				    # if(pixelVal not in pixelDict.keys()):
					    # pixelDict[pixelVal] = 0
				    # pixelDict[pixelVal] += 1

		    # set the tile to that color
		    tile_size=pixelSize * pixelSize
		    sumVal=(int(sumVal[0] / tile_size), int(sumVal[1] / tile_size), int(sumVal[2] / tile_size))
		    # sumVal = max(pixelDict, key=pixelDict.get)

		    for x2 in range(0, pixelSize):
			    for y2 in range(0, pixelSize):
				    pixels[x + x2, y + y2]=sumVal

    # Save the new image
    inputImage_pil.save("./input_images_and_levels/Sketch/" + imageName + processString + ".png", "PNG")
