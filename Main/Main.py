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

# Locations and Methods:
dataLocation = "./data/games/"
gameOptions = sorted(os.listdir(dataLocation))
print(gameOptions)
selectedGame = gameOptions[1]


generateMethods = ['CNN', 'Pixel']
repairMethods = ['AutoEncoder', 'MarkovChain', 'Multi1', 'Multi2']
pixelMethods = ['img', 'histogram', 'avrg']
MCMethods = ["NSEW", "NS", "EW", "SW", "NE"]

# Inputs ================================================================================
# Actual image(s):
imageName = "1"
imageFile = imageName + ".jpeg"
inputImage_pil = Image.open(imageFile)
inputImage_cv = cv2.imread(imageFile)

processString = imageName

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

# Training Models=========================================================================
# Training Info:
trainModels = False
asciiLevels, sprites, spriteAsciiMap = Inputs.Get_All_Inputs(dataLocation, selectedGame)
trainedModelLocations = dataLocation + selectedGame + "/trainedModels/"

# Hyperparameters
patch_width = 2
patch_height = 2  # Anything other than 2, 7, or 14 will need a new stiching method for the CNN
CNN_epochs = 50
CNN_batch = 16

# Trained Model Locations
trainedCNN = trainedModelLocations + "cnn_model" + "_" + str(patch_width) + "_" + str(patch_height) + ".pth"
trainedMarkovChain = trainedModelLocations + "smbprobabilities"
trainedEval = trainedModelLocations + "evalDictionary"
trainedAutoEncoder = trainedModelLocations + "ae_model" + ".pth"
tempFileLocation = "./Temp_for_AE/"
if os.path.exists(tempFileLocation):
    shutil.rmtree(tempFileLocation)
os.makedirs(tempFileLocation)

# Training Methods if required:
if(trainModels):
    for m in MCMethods:
        RepairMC.train_MC(asciiLevels, m, trainedMarkovChain)
    CNNGen.train_model(asciiLevels, pixelSize, sprites, spriteAsciiMap, trainedCNN, CNN_epochs, CNN_batch, patch_width, patch_height)

EvaluateMC.trainEval(asciiLevels, trainedEval)


# Actual System=============================================================================

# user Input
#selectedGenMethod = generateMethods[1]
#selectedRepairMethod = repairMethods[1]
#selectedPixelMethod = pixelMethods[2]
selectedMCMethod = MCMethods[0]
for selectedGenMethod in generateMethods:
    if(selectedGenMethod == 'Pixel'):
        pixelMethodsList = pixelMethods
    else:
        pixelMethodsList = ['img']

    for selectedPixelMethod in pixelMethodsList:
        for selectedRepairMethod in repairMethods:

            processString = ("./output_images_and_levels/" + imageName + "_to_" + selectedGame + "_Gen-" + selectedGenMethod + selectedPixelMethod + "_Rep-" + selectedRepairMethod)
            if not os.path.exists(processString):
                os.makedirs(processString)
            inputImage_pil.save(processString + "/" + "a_Original_Resized.jpeg", "JPEG")
            
            # Generate the level from the images======================================================
            # inputImage => generatedLevel
            generatedLevel = []
            if(selectedGenMethod == 'CNN'):
                generatedLevel = CNNGen.generate(inputImage_cv, pixelSize, spriteAsciiMap, trainedCNN, patch_width, patch_height)
            
            if(selectedGenMethod == 'Pixel'):
                generatedLevel = PixelGen.generate(inputImage_cv, sprites, spriteAsciiMap, pixelSize, selectedPixelMethod)
                
            # Evaluation 1 ===========================================================================
            # generatedLevel => (values)
            generatedImage = Visualize.visualize(generatedLevel, sprites, spriteAsciiMap, pixelSize)
            generatedImage.save(processString + "/" + "b_Generated.jpeg", "JPEG")
            consistencyGen = EvaluateMC.evaluate(generatedLevel, trainedEval)
            closenessGen = EvaluatePixel.evaluate(inputImage_pil, generatedImage)
            
            # Repair the levels ======================================================================
            # generatedLevel => repairedLevel
            repairedLevel=generatedLevel
            if(selectedRepairMethod == 'AutoEncoder'):
                repairedLevel = RepairAE.Repair(repairedLevel, tempFileLocation, imageName, spriteAsciiMap)
            
            if(selectedRepairMethod == 'MarkovChain'):
                repairedLevel = RepairMC.Repair(repairedLevel, trainedMarkovChain, spriteAsciiMap, selectedMCMethod)
            
            if(selectedRepairMethod == 'Multi1'):
                repairedLevel = RepairAE.Repair(repairedLevel, tempFileLocation, imageName, spriteAsciiMap)
                repairedLevel = RepairMC.Repair(repairedLevel, trainedMarkovChain, spriteAsciiMap, selectedMCMethod)
            
            if(selectedRepairMethod == 'Multi2'):
                repairedLevel = RepairMC.Repair(repairedLevel, trainedMarkovChain, spriteAsciiMap, selectedMCMethod)
                repairedLevel = RepairAE.Repair(repairedLevel, tempFileLocation, imageName, spriteAsciiMap)
            
            # Evaluation 2 ===========================================================================
            # repairedLevel => (values)
            repairedImage=Visualize.visualize(repairedLevel, sprites, spriteAsciiMap, pixelSize)
            repairedImage.save(processString + "/" + "c_Repaired.jpeg", "JPEG")
            consistencyRepair=EvaluateMC.evaluate(repairedLevel, trainedEval)
            closenessRepair=EvaluatePixel.evaluate(inputImage_pil, repairedImage)
            
            # Plotting ===============================================================================
            #print("Conisitency After Gen: " + str(consistencyGen))
            #print("Conisitency After Repair: " + str(consistencyRepair))
            #print("Closeness After Gen: " + str(closenessGen))
            #print("Closeness After Repair: " + str(closenessRepair))
            
            EvalFile = open(processString + "/" + "d_Eval.txt", "w")
            EvalFile.write("Conisitency After Gen: " + str(consistencyGen) + "\n")
            EvalFile.write("Conisitency After Repair: " + str(consistencyRepair) + "\n")
            EvalFile.write("Closeness After Gen: " + str(closenessGen) + "\n")
            EvalFile.write("Closeness After Repair: " + str(closenessRepair) + "\n")
            EvalFile.close()
