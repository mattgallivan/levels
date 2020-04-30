import os
import glob
import shutil

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
from conv_fully_connected import ConvFullyConnected
from conv_fully_connected_experiment import ConvFullyConnectedExperiment
from visualize_level import visualize_level

# select which model to use and make sure the appropriate path is selected
model = ConvFullyConnected
model_path = './autoencoder_weights.pth'
base_path = './repair_output/'

# this pipeline is developed to train the network then take a single level file 
# and run it through the repair autoencoder

# level path must be the path to a txt file
def pipeline(level_path, train=False):
    if train:
        # 1. train the model
        data = repair.load_data()
        train_data, test_data = repair.split_data(data)
        # labels, data = repair.load_data_categorical()

        learning_rate = 1e-4
        # repair.train_categorical(labels, data, learning_rate, model, model_path)
        # repair.eval_categorical(labels, data, model, model_path)
        repair.train(train_data, learning_rate, model, model_path)
        repair.eval(test_data, model, model_path)

    # 2. generate chunked input tensors
    input_level_path = level_path
    output_path = base_path + 'input_tensors/'
    if os.path.exists(output_path):
       shutil.rmtree(output_path)
    os.makedirs(output_path)
    generate_one_hot(input_level_path, output_path)

    # 3. generate chunked output tensors 
    # input_path is a path to the directory containing the one-hot encodings of the level we wish to repair
    input_path = output_path
    output_path = base_path + 'output_tensors/'
    if os.path.exists(output_path):
       shutil.rmtree(output_path)
    os.makedirs(output_path)
    repair.output(model, model_path, input_path, output_path)

    # 4. join the chunks
    chunk_dir = output_path
    # save the file to the output path, this should be a file not a directory
    output_file = base_path + 'joined.pth'
    # original file is needed to determine the output dimensions
    create_level(input_level_path, chunk_dir, output_file)

    # 5. turn the tensor back to ASCII 
    repaired_level_tensor = torch.load(output_file)
    repaired_level_textfile = base_path + 'joined.txt'
    join_output(repaired_level_tensor, repaired_level_textfile, save=True) 

    # 6. visualize the output
    input_file = repaired_level_textfile
    output_file = base_path + 'joined.jpeg'
    visualize_level(input_level_path, input_file, output_file)

# pipeline('../data/games/super-mario-bros-simplified/game-levels-ascii/mario-1-3.txt', train=False)
# pipeline('./PCGML3/levels_broken/mario-1-3-broken.txt', train=False)