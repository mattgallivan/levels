import os
import json
import glob
import pickle
from PIL import Image

import Inputs
import RepairMC
import EvaluateMC
import Visualize
import PixelGen


# Inputs ================================================================================
# Actual image(s):
inputImage = Image.open("TestImg.png")

# Locations and Methods:
dataLocation = "./data/games/"
gameOptions = os.listdir(dataLocation)
generateMethods = ['CNN', 'Pixel']
repairMethods = ['AutoEncoder', 'MarkovChain']
# TODO: May be some other hyperparameters we want to set here

#user Input
selectedGame = gameOptions[1]
selectedGenMethod = generateMethods[1]
selectedRepairMethod = repairMethods[1]

# Game data and game pretrained models (should be files):
asciiLevels, sprites, spriteAsciiMap = Inputs.Get_All_Inputs(dataLocation, selectedGame)
trainedModelLocations = dataLocation + selectedGame + "/trainedModels/"
markovProbabilities = pickle.load(open(trainedModelLocations + "smbprobabilities.pickle", "rb"))
trainedCNN = []
trainedAutoEncoder = []

# Generate the level from the images======================================================
# inputImage => generatedLevel
generatedLevel = []
if(selectedGenMethod == 'CNN'):
    x = 0

if(selectedGenMethod == 'Pixel'):
    generatedLevel = PixelGen.generate(inputImage, sprites, spriteAsciiMap, 16, 'img')

# Evaluation 1 ===========================================================================
# generatedLevel => (values)
generatedImage = Visualize.visualize(generatedLevel, sprites, spriteAsciiMap)
EvaluateMC.evaluate(generatedLevel, markovProbabilities)

# Repair the levels ======================================================================
# generatedLevel => repairedLevel
repairedLevel = []
if(selectedRepairMethod == 'AutoEncoder'):
    x = 0

if(selectedRepairMethod == 'MarkovChain'):
    repairedLevel = RepairMC.Repair(generatedLevel, markovProbabilities)

# Evaluation 2 ===========================================================================
# repairedLevel => (values)
repairedImage = Visualize.visualize(repairedLevel, sprites, spriteAsciiMap)

# Plotting ===============================================================================