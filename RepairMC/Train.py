import glob
import pickle

def train_MC(levelLocations, trainingLevels):

	#Extract Markov Random Field Counts from Levels
	markovCounts = {}# Dictionary of (x-1, y), (x-1, y+1), (x, y+1)
	for level in trainingLevels: 
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
	
	return markovProbabilities