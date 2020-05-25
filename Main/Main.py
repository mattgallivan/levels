import os
import json
import glob
import pickle
import cv2
import shutil
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
imageName = "Sample"
imageFile = imageName+".jpeg"
inputImage_pil = Image.open(imageFile)
inputImage_cv = cv2.imread(imageFile)

w,h = inputImage_pil.size

pixelSize = 16

# for now it streches or contracts image but maybe cropping would be better or should have an option for either
outputLevelWidth = w//pixelSize
outputLevelHeight = h//pixelSize

dsize = (pixelSize*outputLevelWidth, pixelSize*outputLevelHeight)
inputImage_pil = inputImage_pil.resize(dsize)
inputImage_cv = cv2.resize(inputImage_cv, dsize)
inputImage_pil.save("./output_images_and_levels/a-originalImage.jpeg", "JPEG")

# Locations and Methods:
dataLocation = "./data/games/"
gameOptions = sorted(os.listdir(dataLocation))
generateMethods = ['CNN', 'Pixel']
pixelMethods = ['img', 'histogram']
repairMethods = ['AutoEncoder', 'MarkovChain']
# TODO: May be some other hyperparameters we want to set here

#user Input
selectedGame = gameOptions[1]
selectedGenMethod = generateMethods[1]
selectedPixelMethods = pixelMethods[0]
selectedRepairMethod = repairMethods[0]
trainModels = False

# Game data and game pretrained models (should be files):
asciiLevels, sprites, spriteAsciiMap = Inputs.Get_All_Inputs(dataLocation, selectedGame)
trainedModelLocations = dataLocation + selectedGame + "/trainedModels/"
trainedMarkovChain = trainedModelLocations + "smbprobabilities.pickle"
trainedCNN = trainedModelLocations + "cnn_model"
patch_width = 20
patch_height = 14 # Anything other than 14 will need a new stiching method
CNN_epochs = 1
CNN_batch = 16
trainedAutoEncoder = []
tempFileLocation = "./Temp_for_AE/"
# if os.path.exists(tempFileLocation):
#     shutil.rmtree(tempFileLocation)
# os.makedirs(tempFileLocation)

if(trainModels):
    RepairMC.train_MC(asciiLevels, trainedMarkovChain)
    CNNGen.train_model(asciiLevels, pixelSize, sprites, spriteAsciiMap, trainedCNN, CNN_epochs, CNN_batch, patch_width, patch_height)

markovProbabilities = pickle.load(open(trainedMarkovChain, "rb"))

# Generate the level from the images======================================================
# inputImage => generatedLevel
generatedLevel = []
if(selectedGenMethod == 'CNN'):
    generatedLevel = CNNGen.generate(inputImage_cv, pixelSize, spriteAsciiMap, trainedCNN, patch_width, patch_height)

if(selectedGenMethod == 'Pixel'):
    generatedLevel = PixelGen.generate(inputImage_cv, sprites, spriteAsciiMap, pixelSize, selectedPixelMethods)
    
# Evaluation 1 ===========================================================================
# generatedLevel => (values)
generatedImage = Visualize.visualize(generatedLevel, sprites, spriteAsciiMap)
generatedImage.save("./output_images_and_levels/b-generatedLevel.jpeg", "JPEG")
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
repairedImage.save("./output_images_and_levels/c-repairedImage.jpeg", "JPEG")
consistencyRepair = EvaluateMC.evaluate(repairedLevel, markovProbabilities)
closenessRepair = EvaluatePixel.evaluate(inputImage_pil, repairedImage)

# Plotting ===============================================================================
print("Conisitency After Gen: " + str(consistencyGen))
print("Conisitency After Repair: " + str(consistencyRepair))
print("Closeness After Gen: " + str(closenessGen))
print("Closeness After Repair: " + str(closenessRepair))