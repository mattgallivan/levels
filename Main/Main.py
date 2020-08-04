import os
import json
import glob
import pickle
import cv2
import shutil
import csv
from PIL import Image

import Inputs
import RepairMC
import RepairAE
import EvaluateMC
import EvaluatePixel
import EvaluateLevel
import Visualize
import PixelGen
import CNNGen

# Locations and Methods:
dataLocation = "./data/games/"
gameOptions = sorted(os.listdir(dataLocation))
print(gameOptions)
selectedGame = gameOptions[1]

generateMethods = ['CNN', 'Pixel']
repairMethods = ['AutoEncoder', 'MarkovChain', 'Multi1']
pixelMethods = ['img', 'histogram', 'avrg']
MCMethods = ["NSEW", "NS", "EW", "SW", "NE", "NW"]
pixelSize = 16

# Training Models=========================================================================
# Training Info:
trainModels = False
asciiLevels, sprites, spriteAsciiMap = Inputs.Get_All_Inputs(dataLocation, selectedGame)
asciiLevelsSMB2, spritesSMB2, spriteAsciiMapSMB2 = Inputs.Get_All_Inputs(dataLocation, "super-mario-bros-2-simplified")
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
    CNNGen.train_model(asciiLevels, pixelSize, sprites, spriteAsciiMap, trainedCNN, CNN_epochs, CNN_batch, patch_width, patch_height)

for m in MCMethods:
    RepairMC.train_MC(asciiLevels, m, trainedMarkovChain)
EvaluateMC.trainEval(asciiLevels, trainedEval)

# Actual System=============================================================================
# Actual image(s):

stats = {}
categories = ["SuperMarioBros2(J)-World", "mario"]
selectedCategory = categories[0]
for testFile in glob.glob("./input_test_files/" + selectedCategory + "*.txt"):

    #imageFile = "./input_images_and_levels/Sketch/mario-2-1--sketch-avg_Crop_Sketch.png"
    testLevel = []
    with open(testFile) as fp:
        y = 0
        for line in fp:
            testLevel.append(line)
            y += 1
    imageName = os.path.splitext(os.path.basename(testFile))[0]
    testImage = Visualize.visualize(testLevel, spritesSMB2, spriteAsciiMap, pixelSize)
    imageFile = "./input_test_files/" + imageName + ".png"
    testImage.save(imageFile, "PNG")
    
    inputImage_pil = Image.open(imageFile)
    inputImage_cv = cv2.imread(imageFile)

    # for now it streches or contracts image but maybe cropping would be better or should have an option for either
    w, h = inputImage_pil.size
    #outputLevelWidth = w // pixelSize
    #outputLevelHeight = h // pixelSize
    outputLevelWidth = 40
    outputLevelHeight = 14
    if(len(testLevel) != outputLevelHeight):
        continue

    # Strech:
    # dsize = (pixelSize * outputLevelWidth, pixelSize * outputLevelHeight)
    # inputImage_pil = inputImage_pil.resize(dsize)
    # inputImage_cv = cv2.resize(inputImage_cv, dsize)

    # Crop:
    level_start=50
    left, top, right, bottom=level_start, 0, level_start + outputLevelWidth, outputLevelHeight
    inputImage_pil=inputImage_pil.crop((pixelSize * left, pixelSize * top, pixelSize * right, pixelSize * bottom))
    inputImage_cv=inputImage_cv[0:0 + outputLevelHeight * pixelSize, level_start * pixelSize:level_start * pixelSize + outputLevelWidth * pixelSize]
    temp=[]
    for i in range(0, outputLevelHeight):
        newRow=""
        for j in range(level_start, level_start + outputLevelWidth):
            newRow += testLevel[i][j]
        temp.append(newRow)
    testLevel=temp

    outputFolder="./output_test_files/" + imageName + "_to_" + selectedGame + "/"
    if os.path.exists(outputFolder):
        shutil.rmtree(outputFolder)
    os.makedirs(outputFolder)
    #EvalFile=open(outputFolder + "Evaluations.txt", "a+")
    # user Input
    # selectedGenMethod = generateMethods[1]
    # selectedRepairMethod = repairMethods[1]
    # selectedPixelMethod = pixelMethods[2]
    selectedMCMethod=MCMethods[3]
    for selectedGenMethod in generateMethods:
        if(selectedGenMethod == 'Pixel'):
            pixelMethodsList=pixelMethods
        else:
            pixelMethodsList=['img']

        for selectedPixelMethod in pixelMethodsList:
            GenMethodInfostring = "Gen-" + selectedGenMethod + selectedPixelMethod
            GenProcessString = (outputFolder + GenMethodInfostring)
            if not os.path.exists(GenProcessString):
                os.makedirs(GenProcessString)
            inputImage_pil.save(GenProcessString + "/" + "a_Original_Resized.png", "PNG")
            
            # Generate the level from the images======================================================
            # inputImage => generatedLevel
            generatedLevel=[]
            if(selectedGenMethod == 'CNN'):
                generatedLevel=CNNGen.generate(inputImage_cv, pixelSize, spriteAsciiMap, trainedCNN, patch_width, patch_height)

            if(selectedGenMethod == 'Pixel'):
                generatedLevel=PixelGen.generate(inputImage_cv, sprites, spriteAsciiMap, pixelSize, selectedPixelMethod)

            # Evaluation 1 ===========================================================================
            # generatedLevel => (values)
            generatedImage = Visualize.visualize(generatedLevel, spritesSMB2, spriteAsciiMap, pixelSize)
            generatedImage.save(GenProcessString + "/" + "b_Generated.png", "PNG")
            consistencyGen=EvaluateMC.evaluate(generatedLevel, trainedEval)
            closenessGen=EvaluatePixel.evaluate(inputImage_pil, generatedImage)
            levelCompareGen = EvaluateLevel.evaluate(testLevel, generatedLevel)

            # Plotting ===============================================================================
            # print("Closeness After Gen: " + str(closenessGen))
            # print("Conisitency After Gen: " + str(consistencyGen))

            # EvalFile.write(GenMethodInfostring + " Closeness After Gen: " + str(closenessGen) + "\n")
            # EvalFile.write(GenMethodInfostring + " Conisitency After Gen: " + str(consistencyGen) + "\n")
            # EvalFile.write(GenMethodInfostring + " LevelCheck After Gen: " + str(levelCompareGen) + "\n")
            # EvalFile.write("\n")
            
            if not ("LevelCompare_" + GenMethodInfostring) in stats.keys():
                stats["LevelCompare_" + GenMethodInfostring] = []
            stats["LevelCompare_" + GenMethodInfostring] += [levelCompareGen]

            for selectedRepairMethod in repairMethods:
                RepMethodInfoString = GenMethodInfostring + "_Rep-" + selectedRepairMethod
                RepProcessString = (GenProcessString + "/" + RepMethodInfoString)
                if not os.path.exists(RepProcessString):
                    os.makedirs(RepProcessString)
                inputImage_pil.save(RepProcessString + "/" + "a_Original_Resized.png", "PNG")
                generatedImage.save(RepProcessString + "/" + "b_Generated.png", "PNG")

                # Repair the levels ======================================================================
                # generatedLevel => repairedLevel
                repairedLevel=generatedLevel
                if(selectedRepairMethod == 'AutoEncoder'):
                    repairedLevel=RepairAE.Repair(repairedLevel, tempFileLocation, imageName, spriteAsciiMap)

                if(selectedRepairMethod == 'MarkovChain'):
                    repairedLevel=RepairMC.Repair(repairedLevel, trainedMarkovChain, spriteAsciiMap, selectedMCMethod)

                if(selectedRepairMethod == 'Multi1'):
                    repairedLevel=RepairAE.Repair(repairedLevel, tempFileLocation, imageName, spriteAsciiMap)
                    repairedLevel=RepairMC.Repair(repairedLevel, trainedMarkovChain, spriteAsciiMap, selectedMCMethod)

                # Evaluation 2 ===========================================================================
                # repairedLevel => (values)
                repairedImage = Visualize.visualize(repairedLevel, spritesSMB2, spriteAsciiMap, pixelSize)
                repairedImage.save(RepProcessString + "/" + "c_Repaired.png", "PNG")
                consistencyRepair=EvaluateMC.evaluate(repairedLevel, trainedEval)
                closenessRepair=EvaluatePixel.evaluate(inputImage_pil, repairedImage)
                levelCompareRepair=EvaluateLevel.evaluate(testLevel, repairedLevel)

                # Plotting ===============================================================================
                # print("Closeness After Repair: " + str(closenessRepair))
                # print("Conisitency After Repair: " + str(consistencyRepair))

                # EvalFile.write(RepMethodInfoString + " Closeness After Repair: " + str(closenessRepair) + "\n")
                # EvalFile.write(RepMethodInfoString + " Conisitency After Repair: " + str(consistencyRepair) + "\n")
                # EvalFile.write(RepMethodInfoString + " LevelCheck After Repair: " + str(levelCompareRepair) + "\n")
                # EvalFile.write("\n")

                if not ("LevelCompare_" + RepMethodInfoString) in stats.keys():
                    stats["LevelCompare_" + RepMethodInfoString]=[]
                stats["LevelCompare_" + RepMethodInfoString] += [levelCompareRepair]

    #EvalFile.close()

with open("output_" + selectedCategory + ".csv", 'a') as outputFile:
    for key, val in stats.items():
        rowStr = str(key) + ','
        rowStr += ','.join([str(i) for i in val]) + '\n'
        outputFile.write(rowStr)
