import sys
import os
import glob
from PIL import Image
import pathlib

#Load sprites
sprites = {}
for filename in glob.glob("./sprites/*.png"):
	im = Image.open(filename)
	splits = filename.split("/")
	name = splits[-1][:-4]
	sprites[name] = im
	# print (str(im.size))

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

#Visualize Output Level
def visualize_level(level_file, input_file, output_file):
	level = {}
	with open(input_file) as fp:
		y = 0
		for line in fp:
			level[y] = line
			y+=1

	#Level Dimensions
	level_width = 0
	level_height = 0

	#opening level
	current_file = open(level_file, "r")
	lines = []
	for line in current_file:
		line_read = line.split(',')
		line_to_write = [n for n in line_read]
		lines.append(line_to_write)

	lines_encoded = []
	for line in lines:
		line_after_encode=[]
		for eachline in line:
			for i in eachline:
				if i != "\n":
					line_after_encode.append(i)

				elif i == "\n":
					pass
			lines_encoded.append(line_after_encode)
	# print(lines_encoded)

	level_height = len(lines_encoded)
	level_width = len(lines_encoded[0])

	# print(level_width)
	# print(level_height)

	maxX = level_width
	maxY = level_height

	image = Image.new("RGB", (maxX*16, maxY*16), color=(91, 153, 254))
	pixels = image.load()

	for y in range(0, maxY):
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
						if pixelsToUse[x2,y2][3]>0:
							pixels[x*16+x2,y*16+y2] = pixelsToUse[x2,y2][0:-1]

	image.save(output_file, "JPEG")
