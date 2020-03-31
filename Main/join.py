import numpy as np
import os
import torch
import torch.nn as nn
import pathlib
import sys
import random
import pickle
from pathlib import Path

#tiles = ["X", "S", "-", "?", "Q", "E", "<", ">", "[", "]", "o", "B", "b"]
#tiles_len = len(tiles)

#get tensor here
# input_dir = 'chunked_data/output_tensors/'
# output_textfile_dir = 'chunked_data/output_textfiles/'
# original_dir = 'chunked_data/one hot tensors/'
# input_textfile_dir = 'chunked_data/input_textfiles/'
# test_file_name = 'tensor_2-1_900'
# test_original_file_name = 'one_hot_tensor_mario-2-1_900'

# output_tensor = torch.load(input_dir + test_file_name + '.pth')
# original_tensor = torch.load(original_dir + test_original_file_name + '.pth')

def join_input(tensor, location, file_name, asciiMapping, save=False):
    chunk_decoded = []

    tiles = list(asciiMapping.keys())
    tiles_len = len(tiles)     

    for vertical_iterator in range(tensor.shape[0]):
        line_decoded = []
        for horizontal_iterator in range(tensor.shape[1]):
            flag = 0
            for i in range(tensor.shape[2]):
                if tensor[vertical_iterator, horizontal_iterator, i] == 1:
                    tile = tiles[i]
                    flag = 1
            if flag == 0:
                #tile is empty
                tile = tiles[2]
            line_decoded.append(tile)
        chunk_decoded.append(line_decoded)

    #for i in chunk_decoded:
        #print(i)
        #print("")

    if save:
        with open(location + file_name + '.txt', "w") as the_file:
            for listt in chunk_decoded:
                for k in listt:
                    the_file.write(k)
                the_file.write("\n")

def join_output(tensor, location, file_name, save=False):
    chunk_decoded = []

    for vertical_iterator in range(tensor.shape[0]):
        line_decoded = []
        for horizontal_iterator in range(tensor.shape[1]):
            for i in range(tensor.shape[2]):
                one_hot = tensor[vertical_iterator, horizontal_iterator]
                if one_hot.sum() == 0:
                    tile = tiles[2]
                else: 
                    normed_probs = nn.functional.normalize(one_hot, p=1, dim=0)
                    if normed_probs.sum() != 1.0:
                        tile = tiles[2]
                    # sometimes the normalized probabilities do not sum to 1 correctly 
                    else: 
                        tile = np.random.choice(tiles, p=normed_probs.detach().numpy())
            line_decoded.append(tile)
        chunk_decoded.append(line_decoded)

    #for i in chunk_decoded:
        #print(i)
        #print("")

    if save: 
        with open(location + file_name + '.txt', "w") as the_file:
            for listt in chunk_decoded:
                for k in listt:
                    the_file.write(k)
                the_file.write("\n")

def join_output_deterministic(tensor, asciiMapping, save=False):
    chunk_decoded = []
    
    tiles = list(asciiMapping.keys())
    tiles_len = len(tiles)   

    for vertical_iterator in range(tensor.shape[0]):
        line_decoded = []
        for horizontal_iterator in range(tensor.shape[1]):
            for i in range(tensor.shape[2]):
                one_hot = tensor[vertical_iterator, horizontal_iterator]
                # if torch.max(one_hot) < 1e-3:
                #     tile = tiles[2]
                # else:
                tile = tiles[torch.argmax(one_hot)]
                if tile == "b":
                    tile = "-"
            line_decoded.append(tile)
        chunk_decoded.append(line_decoded)
    
    return chunk_decoded

    #for i in chunk_decoded:
        #print(i)
        #print("")

    #if save:
        #with open(file_path, "w") as the_file:
            #for listt in chunk_decoded:
                #for k in listt:
                    #the_file.write(k)
                #the_file.write("\n")

# print("input")
# join_input(original_tensor, input_textfile_dir)
# print("output")
# join_output(output_tensor, output_textfile_dir)

# output_tensor = torch.load('./PCGML3/overall_tensor_output.pth')
# join_output_deterministic(output_tensor, './PCGML3/', 'repaired_output', save=True)