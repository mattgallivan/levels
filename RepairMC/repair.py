import sys
import os
import random
import glob
import pickle
from PIL import Image
import Visualize
import VidMaker
import Train

def Repair(Badlevel, markovProbabilities, sprites):
	randomSample = True
	itterations = 1000
	VidImages = []
	
	for interationCounter in range(0,itterations):
		change = False
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
						VidImages.append(Visualize.visualize(Badlevel, sprites))
						change = True
				else:
					continue
					#key does not exist
					#list_of_possible_key = {}
					#for k in range(0,len(key)):
						#for l in visualization.keys():
							#if (key[k] != l):
								#new_key = key[0:k]+l+key[k+1:]
								#if new_key in markovProbabilities.keys():
									#if Badlevel[y][x] in markovProbabilities[new_key].keys():
										#list_of_possible_key[new_key] = markovProbabilities[new_key][Badlevel[y][x]]
					#if len(list_of_possible_key)>0:
						#replace_surrounding_tiles = max(list_of_possible_key, key=list_of_possible_key.get)
						#if(y>0):
							#Badlevel[y-1] = Badlevel[y-1][0:x] + replace_surrounding_tiles[0] + Badlevel[y-1][x+1:]
						#if(x<len(Badlevel[y])-2):
							#Badlevel[y] = Badlevel[y][0:x+1] + replace_surrounding_tiles[1] + Badlevel[y][x+2:]
						#if(y<len(Badlevel)-1):					
							#Badlevel[y+1] = Badlevel[y+1][0:x] + replace_surrounding_tiles[2] + Badlevel[y+1][x+1:]
						#if(x>0):
							#Badlevel[y] = Badlevel[y][0:x-1] + replace_surrounding_tiles[3] + Badlevel[y][x:]
						#VidImages.append(Visualize.visualize(Badlevel, sprites))
		if(not change):
			break
	return Badlevel, VidImages