import glob
from PIL import Image

def visualize(level, sprites, spriteAsciiMap):
	maxY = len(level)
	maxX = len(level[0])-1 # -1 because of the "\n"	
	image = Image.new("RGB", (maxX*16, maxY*16), color=(91, 153, 254))
	pixels = image.load()
			
	maxY = len(level)
	for y in range(0, maxY):
		maxX = len(level[y])
		for x in range(0, maxX):
			imageToUse = None
			if level[y][x] in spriteAsciiMap.keys():
				if(level[y][x] != "-"):
					asciiVal = spriteAsciiMap[level[y][x]][0]
					imageToUse = sprites[asciiVal][0]
			if level[y][x]=="X":
				if y>maxY-2:
					imageToUse = sprites["ground"][0]
				else:
					imageToUse = sprites["stair"][0]
			if not imageToUse == None:
				pixelsToUse = imageToUse.load()
				for x2 in range(0, 16):
					for y2 in range(0, 16):
						#if pixelsToUse[x2,y2][3]>0:
						try:
							qqq = pixelsToUse[x2,y2][0:-1]
							qq = pixels[x*16+x2,y*16+y2]
							pixels[x*16+x2,y*16+y2] = pixelsToUse[x2,y2][0:-1]
						except:
							continue
			else:
				if(level[y][x] == " "):
					for x2 in range(0, 16):
						for y2 in range(0, 16):
							pixels[x*16+x2,y*16+y2] = (0,0,0)
	
	return image