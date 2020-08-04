
def evaluate(testLevel, newLevel):
	Good = 0
	Bad = 0
	
	maxY = len(testLevel)
	maxX = len(testLevel[0].rstrip("\n"))
	for y in range(0, maxY):
		for x in range(0, maxX):
			if(testLevel[y][x] == newLevel[y][x]):
				Good += 1
			else:
				Bad += 1
	return Good/(Good+Bad) * 100