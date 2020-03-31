import numpy as np
import os
import torch
import torch.nn as nn
import pathlib
import sys
import random
import pickle
from pathlib import Path

# one_hot_tensor_dir = os.listdir('./mario_1-1_output/')
# level_dir = "./levels/"
# one_hot_tensor_path = "./mario_1-1_output/"

def create_level(lines_encoded, one_hot_tensor_dir, output_path, asciiMapping):
    #representing tiles
    
    tiles = list(asciiMapping.keys())
    tiles_len = len(tiles)

    #Level Dimensions
    #level_width = 0
    #level_height = 0

    #opening level
    #lines = []
    #path = pathlib.Path(level_path)
    #if path.is_file():
        #current_file = open(path, "r")
        #lines = []
        #for line in current_file:
            #line_read = line.split(',')
            #line_to_write = [n for n in line_read]
            #lines.append(line_to_write)

    #lines_encoded = []
    #for line in lines:
        #line_after_encode=[]
        #for eachline in line:
            #for i in eachline:
                #if i != "\n":
                    #line_after_encode.append(i)

                #elif i == "\n":
                    #pass
            #lines_encoded.append(line_after_encode)
    #print(lines_encoded)

    level_height = len(lines_encoded)
    level_width = len(lines_encoded[0])

    print(level_width)
    print(level_height)

    overall_tensor = torch.zeros(level_height, level_width, tiles_len)

    horizontal_counter = 0
    vertical_counter = 0

    window_vertical = 8
    window_horizontal = 8
    tensor_iterator = 0
    count_vertical_position = 0
    count_horizontal_position = 0

    #load the files in sorted order
    files = os.listdir(one_hot_tensor_dir)
    for filename in sorted(files, key=lambda x: int(os.path.splitext(x)[0])):
        test_tensor = torch.load(one_hot_tensor_dir + '{filename}'.format(filename = filename))

        if vertical_counter < (level_height - window_vertical + 1):
            if horizontal_counter < (level_width - window_horizontal + 1):

                if tensor_iterator == 0 and vertical_counter == 0 and horizontal_counter == 0:
                    h_counter = 0
                    v_counter = 0
                    for vertical_iterator in range(test_tensor.shape[0]):
                        for horizontal_iterator in range(test_tensor.shape[1]):
                            overall_tensor[v_counter, h_counter, :] = test_tensor[vertical_iterator, horizontal_iterator, :]
                            h_counter+=1
                        v_counter+=1
                    tensor_iterator+=1
                    horizontal_counter+=1
                    count_horizontal_position = 8

                elif tensor_iterator != 0 and vertical_counter == 0 and horizontal_counter != 0:
                    h_counter = count_horizontal_position
                    v_counter = 0
                    for vertical_iterator in range(test_tensor.shape[0]):
                        overall_tensor[v_counter, h_counter, :] = test_tensor[vertical_iterator, 7, :]
                        v_counter+=1

                    if horizontal_counter == (level_width - window_horizontal):
                        count_horizontal_position = 0
                        horizontal_counter = 0
                        vertical_counter+=1
                        count_vertical_position = 8
                        tensor_iterator = 0
                    else:
                        count_horizontal_position += 1
                        horizontal_counter+=1
                        tensor_iterator+=1

                elif tensor_iterator == 0 and vertical_counter != 0 and horizontal_counter == 0:
                    h_counter = count_horizontal_position
                    v_counter = count_vertical_position

                    for horizontal_iterator in range(test_tensor.shape[1]):
                        overall_tensor[v_counter, h_counter, :] = test_tensor[7, horizontal_iterator, :]
                        h_counter+=1
                    count_horizontal_position = 8
                    horizontal_counter+=1
                    tensor_iterator+=1

                elif tensor_iterator != 0 and vertical_counter != 0 and horizontal_counter != 0:
                    h_counter = count_horizontal_position
                    v_counter = count_vertical_position

                    overall_tensor[v_counter, h_counter, :] = test_tensor[7, 7, :]
                    if horizontal_counter == (level_width - window_horizontal):
                        count_horizontal_position = 0
                        horizontal_counter = 0
                        vertical_counter+=1
                        count_vertical_position += 1
                        tensor_iterator = 0
                    else:
                        count_horizontal_position +=1
                        horizontal_counter+=1
                        tensor_iterator+=1

    torch.save(overall_tensor, output_path)
