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

# Locations and Methods:
dataLocation = "./data/games/"
gameOptions = sorted(os.listdir(dataLocation))
print(gameOptions)
generateMethods = ['CNN', 'Pixel']
repairMethods = ['AutoEncoder', 'MarkovChain']
pixelMethods = ['img', 'histogram', 'avrg']
MCMethods = ["NSEW", "NS", "EW", "SW", "NE"]

# Inputs ================================================================================
# Actual image(s):
imageName = "TestImg"
imageFile = imageName+".jpeg"
inputImage_pil = Image.open(imageFile)
inputImage_cv = cv2.imread(imageFile)

# for now it streches or contracts image but maybe cropping would be better or should have an option for either
w,h = inputImage_pil.size
pixelSize = 16
outputLevelWidth = w//pixelSize
outputLevelHeight = h//pixelSize
outputLevelWidth = 202
outputLevelHeight = 14

dsize = (pixelSize*outputLevelWidth, pixelSize*outputLevelHeight)
inputImage_pil = inputImage_pil.resize(dsize)
inputImage_cv = cv2.resize(inputImage_cv, dsize)
inputImage_pil.save("./output_images_and_levels/a-originalImage.jpeg", "JPEG")

# TODO: May be some other hyperparameters we want to set here

#user Input
selectedGame = gameOptions[1]
selectedGenMethod = generateMethods[1]
selectedRepairMethod = repairMethods[1]

selectedPixelMethods = pixelMethods[2]
selectedMCMethod = MCMethods[0]

# Training Models=========================================================================
# Training Info:
trainModels = True
asciiLevels, sprites, spriteAsciiMap = Inputs.Get_All_Inputs(dataLocation, selectedGame)
trainedModelLocations = dataLocation + selectedGame + "/trainedModels/"

# Hyperparameters
patch_width = 2
patch_height = 2 # Anything other than 14 will need a new stiching method for the CNN
CNN_epochs = 20
CNN_batch = 16

# Trained Model Locations
trainedCNN = trainedModelLocations + "cnn_model" + "_" + str(patch_width) + "_" + str(patch_height) + ".pth"
trainedMarkovChain = trainedModelLocations + "smbprobabilities"
trainedEval = trainedModelLocations + "evalDictionary"
trainedAutoEncoder = trainedModelLocations + "ae_model" + ".pth"
tempFileLocation = "./Temp_for_AE/"

# Training Methods if required:
if(trainModels):
    for m in MCMethods:
        RepairMC.train_MC(asciiLevels, m, trainedMarkovChain)
    #CNNGen.train_model(asciiLevels, pixelSize, sprites, spriteAsciiMap, trainedCNN, CNN_epochs, CNN_batch, patch_width, patch_height)

EvaluateMC.trainEval(asciiLevels, trainedEval)

# Generate the level from the images======================================================
# inputImage => generatedLevel
generatedLevel = []
if(selectedGenMethod == 'CNN'):
    generatedLevel = CNNGen.generate(inputImage_cv, pixelSize, spriteAsciiMap, trainedCNN, patch_width, patch_height)

if(selectedGenMethod == 'Pixel'):
    generatedLevel = PixelGen.generate(inputImage_cv, sprites, spriteAsciiMap, pixelSize, selectedPixelMethods)
    
# Evaluation 1 ===========================================================================
# generatedLevel => (values)
generatedImage = Visualize.visualize(generatedLevel, sprites, spriteAsciiMap, pixelSize)
generatedImage.save("./output_images_and_levels/b-generatedLevel.jpeg", "JPEG")
consistencyGen = EvaluateMC.evaluate(generatedLevel, trainedEval)
closenessGen = EvaluatePixel.evaluate(inputImage_pil, generatedImage)

# Repair the levels ======================================================================
# generatedLevel => repairedLevel
repairedLevel = []
if(selectedRepairMethod == 'AutoEncoder'):
    repairedLevel = RepairAE.Repair(generatedLevel, tempFileLocation, imageName, spriteAsciiMap)

if(selectedRepairMethod == 'MarkovChain'):
    repairedLevel = generatedLevel
    repairedLevel = RepairMC.Repair(repairedLevel, trainedMarkovChain, spriteAsciiMap, selectedMCMethod)

# Evaluation 2 ===========================================================================
# repairedLevel => (values)
repairedImage = Visualize.visualize(repairedLevel, sprites, spriteAsciiMap, pixelSize)
repairedImage.save("./output_images_and_levels/c-repairedImage.jpeg", "JPEG")
consistencyRepair = EvaluateMC.evaluate(repairedLevel, trainedEval)
closenessRepair = EvaluatePixel.evaluate(inputImage_pil, repairedImage)

# Plotting ===============================================================================
print("Conisitency After Gen: " + str(consistencyGen))
print("Conisitency After Repair: " + str(consistencyRepair))
print("Closeness After Gen: " + str(closenessGen))
print("Closeness After Repair: " + str(closenessRepair))