import glob
from PIL import Image

def visualize(level, sprites):
	# TODO: REPLACE WITH JSON READER
	visualization = {}
	visualization["S"] = "brick"
	visualization["?"] = "questionMark"
	visualization["Q"] = "emptyBlock"
	visualization["E"] = "goomba"
	visualization["<"] = "topLeftPipe"
	visualization[">"] = "topRightPipe"
	visualization["["] = "leftPipe"
	visualization["]"] = "rightPipe"
	visualization["o"] = "coin"
	visualization["B"] = "cannonTop"
	visualization["b"] = "cannonBottom"
	
	maxY = len(level)
	maxX = len(level[0])-1 # -1 because of the "\n"	
	image = Image.new("RGB", (maxX*16, maxY*16), color=(91, 153, 254))
	pixels = image.load()
			
	maxY = len(level)
	for y in range(0, maxY):
		maxX = len(level[y])
		for x in range(0, maxX):
			imageToUse = None
			if level[y][x] in visualization.keys():
				imageToUse = sprites[visualization[level[y][x]]]
			elif level[y][x]=="X":
				if y>maxY-2:
					imageToUse = sprites["ground"]
				else:
					imageToUse = sprites["stair"]
			if not imageToUse == None:
				pixelsToUse = imageToUse.load()
				for x2 in range(0, 16):
					for y2 in range(0, 16):
						#if pixelsToUse[x2,y2][3]>0:
						pixels[x*16+x2,y*16+y2] = pixelsToUse[x2,y2][0:-1]
			else:
				if(level[y][x] == " "):
					for x2 in range(0, 16):
						for y2 in range(0, 16):
							pixels[x*16+x2,y*16+y2] = (0,0,0)
	
	#image.save("./Better.jpeg", "JPEG")
	return image