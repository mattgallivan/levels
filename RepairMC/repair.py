import sys
import os
import random
import glob
import pickle
from PIL import Image

# Inputs:
bW = 1		
bH = 1

#Load sprites
sprites = {}
for filename in glob.glob("./sprites/*.png"):
	im = Image.open(filename)
	splits = filename.split("\\")
	name = splits[-1][:-4]
	sprites[name] = im

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

# =================================Train=======================================

levels = []#list of dictionaries, each dictionary a level

#Load SMB Converted Level(s)
for levelFile in glob.glob("./Original/*.txt"):
	with open(levelFile) as fp:
		level = {}
		y = 0
		for line in fp:
			level[y] = line
			y+=1
		levels.append(level)

#Extract Markov Random Field Counts from Levels
markovCounts = {}# Dictionary of (x-1, y), (x-1, y+1), (x, y+1)
for level in levels: 
	minY = 0
	maxY = len(level)-1
	for y in range(0, maxY):
		minX = 0
		maxX = len(level[y])-1
		for x in range(0, maxX):
			
			current = level[y][x]
			key = ""
			if(y+1>maxY):
				north = " "
			else:
				north = level[y+1][x]
			if(y-1<minY):
				south = " "
			else:
				south = level[y-1][x]
			if(x+1>maxX):
				east = " "
			else:
				east = level[y][x+1]
			if(x-1>minX):
				west = " "
			else:
				west = level[y][x-1]
		
			key = north+east+south+west			
			key = key.replace("\n", " ")
			
			if not key in markovCounts.keys():
				markovCounts[key] = {}
			if not current in markovCounts[key].keys():
				markovCounts[key][current] = 0
			markovCounts[key][current] += 1.0

#Normalize markov counts
markovProbabilities = {}
for key in markovCounts.keys():
	markovProbabilities[key] = {}
	sumVal = 0
	for key2 in markovCounts[key].keys():
		sumVal+=markovCounts[key][key2]
	for key2 in markovCounts[key].keys():
		markovProbabilities[key][key2] =markovCounts[key][key2]/sumVal

pickle.dump(markovProbabilities, open("smbprobabilities.pickle", "wb"))

# =================================Repair=======================================

with open("./Bad.txt") as fp:
	Badlevel = {}
	y = 0
	for line in fp:
		Badlevel[y] = line
		y+=1

markovProbabilities = pickle.load(open("smbprobabilities.pickle", "rb"))

for qqq in range(0,100):
	minY = 0
	maxY = len(Badlevel)-1
	list1 = sorted(range(0, maxY), key = lambda x: random.random() )
	for y in list1:
		minX = 0
		maxX = len(Badlevel[y])-1
		list2 = sorted(range(0, maxX), key = lambda x: random.random() )
		for x in list2:
			
			current = Badlevel[y][x]
			key = ""
			if(y+1>maxY):
				north = " "
			else:
				north = Badlevel[y+1][x]
			if(y-1<minY):
				south = " "
			else:
				south = Badlevel[y-1][x]
			if(x+1>maxX):
				east = " "
			else:
				east = Badlevel[y][x+1]
			if(x-1>minX):
				west = " "
			else:
				west = Badlevel[y][x-1]
			
			key = north+east+south+west
			key = key.replace("\n", " ")
	
			if key in markovProbabilities.keys():	
				if current in markovProbabilities[key].keys():
					# key exists and location is valid
					continue
				else:
					# key exists and location in invalid
					#replaceTile = max(markovProbabilities[key], key=markovProbabilities[key].get)
					#Badlevel[y] = Badlevel[y][0:x] + replaceTile + Badlevel[y][x+1:]
					randomSample = random.uniform(0, 1)
					currValue = 0.0
					for key2 in markovProbabilities[key]:
						if randomSample>=currValue and randomSample<currValue+markovProbabilities[key][key2]:
							Badlevel[y] = Badlevel[y][0:x] + key2 + Badlevel[y][x+1:]
							break
						currValue+=markovProbabilities[key][key2]
			else:
				#key does not exist
				list_of_possible_key = {}
				for k in range(0,len(key)):
					for l in visualization.keys():
						if (key[k] != l):
							new_key = key[0:k]+l+key[k+1:]
							if new_key in markovProbabilities.keys():
								if level[y][x] in markovProbabilities[new_key].keys():
									list_of_possible_key[new_key] = markovProbabilities[new_key][level[y][x]]
				if len(list_of_possible_key)>0:
					replace_surrounding_tiles = max(list_of_possible_key, key=list_of_possible_key.get)
					if y-1 >= 0:
						level[y-1] = level[y-1][0:x] + replace_surrounding_tiles[0] + level[y-1][x+1:]
						level[y] = level[y][0:x+1] + replace_surrounding_tiles[1] + level[y][x+2:]
					if y+1 < len(level):
						level[y+1] = level[y+1][0:x] + replace_surrounding_tiles[2] + level[y+1][x+1:]
						level[y] = level[y][0:x-1] + replace_surrounding_tiles[3] + level[y][x:]

fileName = "./Better.txt"
	
with open(fileName, "w") as the_file:
	for y in range(0, maxY+1):
		the_file.write(Badlevel[y])	












		
#Visualize Output Level
level = {}
with open("./Bad.txt") as fp:
	y = 0
	for line in fp:
		level[y] = line
		y+=1

image = Image.new("RGB", (1600, 240), color=(91, 153, 254))
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
					if pixelsToUse[x2,y2][3]>0:
						pixels[x*16+x2,y*16+y2] = pixelsToUse[x2,y2][0:-1]

image.save("./Bad.jpeg", "JPEG")

#Visualize Output Level
level = {}
with open("./Better.txt") as fp:
	y = 0
	for line in fp:
		level[y] = line
		y+=1

image = Image.new("RGB", (1600, 240), color=(91, 153, 254))
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
					if pixelsToUse[x2,y2][3]>0:
						pixels[x*16+x2,y*16+y2] = pixelsToUse[x2,y2][0:-1]

image.save("./Better.jpeg", "JPEG")