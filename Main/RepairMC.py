import sys
import os
import random
from random import randrange
import glob
import pickle
from PIL import Image

def Repair(Badlevel, trainedMarkovChainLocation, spriteAsciiMap, method, randomSample=True, itterations=1, extraCase=False, randomOrder=False):
	
	markovProbabilities = pickle.load(open(trainedMarkovChainLocation + method + ".pickle", "rb"))	
	tiles = list(spriteAsciiMap.keys())
	
	for interationCounter in range(0,itterations):
		change = False
		minY = 0
		minX = 0
		maxY = len(Badlevel)
		maxX = len(Badlevel[0].rstrip("\n"))
		
		coordinates = []
		for y in range(minY, maxY):
			for x in range(minX, maxX):
				place = len(coordinates)
				if(place>0 and randomOrder):
					place = randrange(place)
				coordinates.insert(place, [x,y])
		
		for x,y in coordinates:
			current = Badlevel[y][x]
			if(y+1>maxY-1):
				north = " "
			else:
				north = Badlevel[y+1][x]
			if(y-1<minY):
				south = " "
			else:
				south = Badlevel[y-1][x]
			if(x+1>maxX-1):
				east = " "
			else:
				east = Badlevel[y][x+1]
			if(x-1<minX):
				west = " "
			else:
				west = Badlevel[y][x-1]
	
			key = ""			
			if(method == "NS"):
				key = north+south
			if(method == "EW"):
				key = east+west
			if(method == "SW"):
				key = south + west
			if(method == "NE"):
				key = north + east
			if(method == "NW"):
				key = north + west
			if(method == "NSEW"):
				key = north+east+south+west
			key = key.replace("\n", " ")
	
			if key in markovProbabilities.keys():	
				if current in markovProbabilities[key].keys():
					# key exists and location is valid
					continue
				else:
					# key exists and location in invalid
					if(randomSample):
						randomSample = random.uniform(0, 1)
						currValue = 0.0
						for key2 in markovProbabilities[key]:
							if randomSample>=currValue and randomSample<currValue+markovProbabilities[key][key2]:
								Badlevel[y] = Badlevel[y][0:x] + key2 + Badlevel[y][x+1:]
								break
							currValue+=markovProbabilities[key][key2]
					else:
						replaceTile = max(markovProbabilities[key], key=markovProbabilities[key].get)
						Badlevel[y] = Badlevel[y][0:x] + replaceTile + Badlevel[y][x+1:]
					change = True
			else:
				#key does not exist
				if(not extraCase):
					continue
				else:
					change = True
					
					list_of_possible_key = {}
					for k in range(0,len(key)):
						for l in tiles:
							if (key[k] != l):
								new_key = key[0:k]+l+key[k+1:]
								if new_key in markovProbabilities.keys():
									if Badlevel[y][x] in markovProbabilities[new_key].keys():
										list_of_possible_key[new_key] = markovProbabilities[new_key][Badlevel[y][x]]
					if len(list_of_possible_key)>0:
						replace_surrounding_tiles = max(list_of_possible_key, key=list_of_possible_key.get)
						if(y>0):
							Badlevel[y-1] = Badlevel[y-1][0:x] + replace_surrounding_tiles[0] + Badlevel[y-1][x+1:]
						if(x<len(Badlevel[y])-2):
							Badlevel[y] = Badlevel[y][0:x+1] + replace_surrounding_tiles[1] + Badlevel[y][x+2:]
						if(y<len(Badlevel)-1):					
							Badlevel[y+1] = Badlevel[y+1][0:x] + replace_surrounding_tiles[2] + Badlevel[y+1][x+1:]
						if(x>0):
							Badlevel[y] = Badlevel[y][0:x-1] + replace_surrounding_tiles[3] + Badlevel[y][x:]
					else:								
						for k in range(0,len(key)):
							for l in tiles:
								if (key[k] != l):
									new_key = key[0:k]+l+key[k+1:]
									if new_key in markovProbabilities.keys():
										for l in tiles:
											if Badlevel[y][x] != l and l in markovProbabilities[new_key].keys():
												list_of_possible_key[new_key+l] = markovProbabilities[new_key][l]
						if len(list_of_possible_key)>0:
							replace_surrounding_tiles = max(list_of_possible_key, key=list_of_possible_key.get)
							if(y>0):
								Badlevel[y-1] = Badlevel[y-1][0:x] + replace_surrounding_tiles[0] + Badlevel[y-1][x+1:]
							if(x<len(Badlevel[y])-2):
								Badlevel[y] = Badlevel[y][0:x+1] + replace_surrounding_tiles[1] + Badlevel[y][x+2:]
							if(y<len(Badlevel)-1):					
								Badlevel[y+1] = Badlevel[y+1][0:x] + replace_surrounding_tiles[2] + Badlevel[y+1][x+1:]
							if(x>0):
								Badlevel[y] = Badlevel[y][0:x-1] + replace_surrounding_tiles[3] + Badlevel[y][x:]				
		if(not change):
			break
	return Badlevel







def train_MC(trainingLevels, method, MC_model_location):
	#Extract Markov Random Field Counts from Levels
	markovCounts = {}
	for level in trainingLevels: 
		minY = 0
		minX = 0
		maxY = len(level)
		maxX = len(level[0].rstrip("\n"))
		for y in range(0, maxY):
			for x in range(0, maxX):
				
				current = level[y][x]
				if(y+1>maxY-1):
					north = " "
				else:
					north = level[y+1][x]
				if(y-1<minY):
					south = " "
				else:
					south = level[y-1][x]
				if(x+1>maxX-1):
					east = " "
				else:
					east = level[y][x+1]
				if(x-1<minX):
					west = " "
				else:
					west = level[y][x-1]
	
				key = ""			
				if(method == "NS"):
					key = north+south
				if(method == "EW"):
					key = east+west
				if(method == "SW"):
					key = south+west
				if(method == "NE"):
					key = north+east
				if(method == "NW"):
					key = north + west
				if(method == "NSEW"):
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
			
	pickle.dump(markovProbabilities, open(MC_model_location + method + ".pickle", "wb"))