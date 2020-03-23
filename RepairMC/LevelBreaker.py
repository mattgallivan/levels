import random

def break_level(level, markovProbabilities, prob, visualization):
	randomSwap = False # if false then we will replace with empty space
	minY = 0
	maxY = len(level)-1
	list1 = sorted(range(0, maxY), key = lambda x: random.random() )
	for y in list1:
		minX = 0
		maxX = len(level[y])-1
		list2 = sorted(range(0, maxX), key = lambda x: random.random() )
		for x in list2:
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

			randVal = random.uniform(0,1)
			if(randVal < prob):
				if(key not in markovProbabilities.keys()):
					# already broken
					continue
				
				# we want to make it worse so add things that have never been there in the training data
				validKeys = markovProbabilities[key].keys()
				invalidKeys = []
				for key in visualization.keys():
					if(key not in validKeys):
						invalidKeys.append(key)
				if(len(invalidKeys)==0):
					# somehow everything is valid
					continue
			
				replaceTile = " "
				if(randomSwap):
					replaceTile = random.choice(list(invalidKeys))
				level[y] = level[y][0:x] + replaceTile + level[y][x+1:]				
		
	return level