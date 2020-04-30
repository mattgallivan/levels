def evaluate(level, markovProbabilities):
	Good = 0
	Bad = 0

	minY = 0
	maxY = len(level)-1
	list1 = range(0, maxY)
	for y in list1:
		minX = 0
		maxX = len(level[y])-1
		list2 = range(0, maxX)
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
	
			if key in markovProbabilities.keys():	
				if current in markovProbabilities[key].keys():
					Good += 1
				else:
					Bad += 1
			else:
				Bad += 1
	return Good/(Good+Bad)