import os
import json
import glob
import pickle
import cv2
from PIL import Image

import Inputs
import RepairMC
import RepairAE
import EvaluateMC
import EvaluatePixel
import Visualize
import PixelGen
import CNNGen


# Inputs ================================================================================
# Actual image(s):
imageName = "TestImg"
imageFile = "TestImg.jpeg"
inputImage_pil = Image.open(imageFile)
inputImage_cv = cv2.imread(imageFile)

# Locations and Methods:
dataLocation = "./data/games/"
gameOptions = os.listdir(dataLocation)
generateMethods = ['CNN', 'Pixel']
repairMethods = ['AutoEncoder', 'MarkovChain']
pixelMethods = ['img', 'histogram']
pixelSize = 16
# TODO: May be some other hyperparameters we want to set here

#user Input
selectedGame = gameOptions[1]
selectedGenMethod = generateMethods[1]
selectedRepairMethod = repairMethods[1]
selectedMpixelMethods = pixelMethods[0]

# Game data and game pretrained models (should be files):
asciiLevels, sprites, spriteAsciiMap = Inputs.Get_All_Inputs(dataLocation, selectedGame)
trainedModelLocations = dataLocation + selectedGame + "/trainedModels/"
markovProbabilities = pickle.load(open(trainedModelLocations + "smbprobabilities.pickle", "rb"))
trainedCNN = []
trainedAutoEncoder = []
tempFileLocation = "./Temp_for_AE/"

# Generate the level from the images======================================================
# inputImage => generatedLevel
generatedLevel = []
if(selectedGenMethod == 'CNN'):
    generatedLevel = CNNGen.generate(inputImage_cv, pixelSize)

if(selectedGenMethod == 'Pixel'):
    generatedLevel = PixelGen.generate(inputImage_cv, sprites, spriteAsciiMap, pixelSize, selectedMpixelMethods)
    
# Evaluation 1 ===========================================================================
# generatedLevel => (values)
generatedImage = Visualize.visualize(generatedLevel, sprites, spriteAsciiMap)
generatedImage.save("./generatedLevel.jpeg", "JPEG")
consistencyGen = EvaluateMC.evaluate(generatedLevel, markovProbabilities)
closenessGen = EvaluatePixel.evaluate(inputImage_pil, generatedImage)

# Repair the levels ======================================================================
# generatedLevel => repairedLevel
repairedLevel = []
if(selectedRepairMethod == 'AutoEncoder'):
    repairedLevel = RepairAE.Repair(generatedLevel, tempFileLocation, imageName, spriteAsciiMap)

if(selectedRepairMethod == 'MarkovChain'):
    repairedLevel = RepairMC.Repair(generatedLevel, markovProbabilities)

# Evaluation 2 ===========================================================================
# repairedLevel => (values)
repairedImage = Visualize.visualize(repairedLevel, sprites, spriteAsciiMap)
repairedImage.save("./repairedImage.jpeg", "JPEG")
consistencyRepair = EvaluateMC.evaluate(repairedLevel, markovProbabilities)
closenessRepair = EvaluatePixel.evaluate(inputImage_pil, repairedImage)

# Plotting ===============================================================================
print("Conisitency After Gen: " + str(consistencyGen))
print("Conisitency After Repair: " + str(consistencyRepair))
print("Closeness After Gen: " + str(closenessGen))
print("Closeness After Repair: " + str(closenessRepair))