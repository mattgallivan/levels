import sys
import os
import random
from random import randrange
import glob
import pickle
from PIL import Image

def Repair(Badlevel, markovProbabilities, randomSample = True, itterations = 1000, extraCase = False):
	for interationCounter in range(0,itterations):
		change = False
		maxY = len(Badlevel)-1	
		maxX = len(Badlevel[0])-1
		minY = 0
		minX = 0
		
		coordinates = []
		for y in range(0, maxY):	
			for x in range(0, maxX):
				place = len(coordinates)
				if(place>0):
					place = randrange(place)
				coordinates.insert(place, [x,y])
		
		for x,y in coordinates:
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
						for l in visualization.keys():
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
							for l in visualization.keys():
								if (key[k] != l):
									new_key = key[0:k]+l+key[k+1:]
									if new_key in markovProbabilities.keys():
										for l in visualization.keys():
											if level[y][x] != l and l in markovProbabilities[new_key].keys():
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