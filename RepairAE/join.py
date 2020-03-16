import numpy as np
import os
import torch
import torch.nn as nn
import pathlib
import sys
import random
import pickle
from pathlib import Path

tiles = ["X", "S", "-", "?", "Q", "E", "<", ">", "[", "]", "o", "B", "b"]
tiles_len = len(tiles)

#get tensor here
input_dir = 'chunked_data/output_tensors/'
output_textfile_dir = 'chunked_data/output_textfiles/'
original_dir = 'chunked_data/one hot tensors/'
input_textfile_dir = 'chunked_data/input_textfiles/'
file_name = 'tensor_3-1_698'
original_file_name = 'one_hot_tensor_mario-3-1_698'

output_tensor = torch.load(input_dir + file_name + '.pth')
original_tensor = torch.load(original_dir + original_file_name + '.pth')

print(output_tensor)
print(output_tensor.shape[0])

def join(tensor, location):
    chunk_decoded = []

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

    for i in chunk_decoded:
        print(i)
        print("")

    with open(location + file_name + '.txt', "w") as the_file:
        for listt in chunk_decoded:
            for k in listt:
                the_file.write(k)
            the_file.write("\n")

join(output_tensor, output_textfile_dir)
join(original_tensor, input_textfile_dir)