import os
import glob

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image

import repair
from createleveltry2 import create_level
from join import join_input, join_output, join_output_deterministic
from generate_one_hot import generate_one_hot
from guzdial_autoencoder import GuzdialConvAutoEncoder
#from visualize_level import visualize_level

# select which model to use and make sure the appropriate path is selected
model = repair.ConvAutoEncoder
model_path = './autoencoder_weights.pth'
base_path = './repair_output/'
#input_path = './PCGML3/levels/mario-1-1.txt'

# this pipeline is developed to train the network then take a single level file 
# and run it through the repair autoencoder

# level path must be the path to a txt file
def Repair(level, output_path, levelname, spriteAsciiMap):
    generate_one_hot(level, output_path, levelname, spriteAsciiMap)
    #generate_one_hot(input_level_path, output_path)

    # 3. generate chunked output tensors 
    # input_path is a path to the directory containing the one-hot encodings of the level we wish to repair
    # input_path = './PCGML3/mario_1-1_broken/'
    input_path = output_path
    # output_path = './PCGML3/mario_1-1_broken_output/'
    output_path = base_path + 'output_tensors/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    repair.output(model, model_path, input_path, output_path)

    # 4. join the chunks
    chunk_dir = output_path
    # save the file to the output path, this should be a file not a directory
    # output_file = './PCGML3/mario_1-1_broken_output_joined.pth'
    output_file = base_path + 'joined.pth'
    # original file is needed to determine the output dimensions
    create_level(input_level_path, chunk_dir, output_file, asciiMapping)
    
    # 5. turn the tensor back to ASCII 
    repaired_level_tensor = torch.load(output_file)
    #repaired_level_textfile = base_path + 'joined.txt'
    chunk_decoded = join_output_deterministic(repaired_level_tensor, repaired_level_textfile, asciiMapping, save=True) 

    # 6. visualize the output
    #input_file = repaired_level_textfile
    #output_file = base_path + 'joined.jpeg'
    #visualize_level(input_level_path, input_file, output_file)

#pipeline(input_path)