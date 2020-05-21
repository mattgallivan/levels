import pickle

def evaluate(level, evalLocation):
	dictionary = pickle.load(open(evalLocation + ".pickle", "rb"))
	Good = 0
	Bad = 0
	
	maxY = len(level)
	maxX = len(level[0].rstrip("\n"))
	for y in range(0, maxY):
		for x in range(0, maxX):
			key = ""
			for i in range(-1,2):
				for j in range(-1, 2):
					if(y+i>maxY-1 or y+i<0 or x+j>maxX-1 or x+j<0):
						key += " "
					else:
						key += level[y+i][x+j]

			key = key.replace("\n", " ")
			if key in dictionary.keys():
				Good += 1
			else:
				Bad += 1
	return Good/(Good+Bad) * 100



def trainEval(trainingLevels, evalLocation):
	dictionary = {}
	
	for level in trainingLevels:
		maxY = len(level)
		maxX = len(level[0].rstrip("\n"))
		for y in range(0, maxY):
			for x in range(0, maxX):
				key = ""
				for i in range(-1,2):
					for j in range(-1, 2):
						if(y+i>maxY-1 or y+i<0 or x+j>maxX-1 or x+j<0):
							key += " "
						else:
							key += level[y+i][x+j]
	
				key = key.replace("\n", " ")
				if key in dictionary.keys():
					dictionary[key] += 1
				else:
					dictionary[key] = 1
	
	pickle.dump(dictionary, open(evalLocation + ".pickle", "wb"))