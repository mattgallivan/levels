import sys
import os
import random
import glob
import pickle
from PIL import Image
import Visualize
import VidMaker
import Train
import Repair

# =================================Inputs=======================================

# Inputs:
randomSample = True
itterations = 10

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

# Sprite Location
spriteLocation = "./sprites/*.png"
#Load sprites
sprites = {}
for filename in glob.glob(spriteLocation):
	im = Image.open(filename)
	splits = filename.split("\\")
	name = splits[-1][:-4]
	sprites[name] = im
	
# =================================Train=======================================

levelLocations = "./Original/*.txt"
markovProbabilities = Train.train_MC(levelLocations)
pickle.dump(markovProbabilities, open("smbprobabilities.pickle", "wb"))

# =================================Repair=======================================

with open("./Bad.txt") as fp:
	Badlevel = {}
	y = 0
	for line in fp:
		Badlevel[y] = line
		y+=1
image = Visualize.visualize(Badlevel, sprites)
image.save("./Bad.jpeg", "JPEG")

markovProbabilities = pickle.load(open("smbprobabilities.pickle", "rb"))
Badlevel, VidImages = Repair.Repair(Badlevel, markovProbabilities, sprites)

fileName = "./Better.txt"
with open(fileName, "w") as the_file:
	for y in range(0, len(Badlevel)):
		the_file.write(Badlevel[y])

#Visualize Output Level
level = {}
with open("./Better.txt") as fp:
	y = 0
	for line in fp:
		level[y] = line
		y+=1
image = Visualize.visualize(level, sprites)
image.save("./Better.jpeg", "JPEG")

VidMaker.makeVid(VidImages)