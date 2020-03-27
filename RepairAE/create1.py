import numpy as np
import os
import torch
import torch.nn as nn
import pathlib
from pathlib import Path

level_dir = "./mario levels/"
one_hot_tensor_dir = "./one hot tensors/"

tiles = ["X", "S", "-", "?", "Q", "E", "<", ">", "[", "]", "o", "B", "b"]
tiles_len = len(tiles)
lines = []

for path in pathlib.Path(level_dir).iterdir():
    filename = path.stem
    if path.is_file():
        current_file = open(path, "r")
        #print(current_file.read())
        lines = []
        for line in current_file:
            line_read = line.split(',')
            line_to_write = [n for n in line_read]
            lines.append(line_to_write)

    lines_encoded = []
    for line in lines:
        line_after_encode=[]
        for eachline in line:
            for i in eachline:
                if i != "\n":
                    line_after_encode.append(i)

                elif i == "\n":
                    pass
            lines_encoded.append(line_after_encode)
    print(lines_encoded)

    #Dimensions
    print(len(lines_encoded))
    print(len(lines_encoded[0]))

    #Window Dimensions
    window_vertical = 8
    window_horizontal = 8
    count = 0
    count1=0
    for vertical_iterator in range(0, ((len(lines_encoded)-window_vertical)+1)):
        for horizontal_iterator in range(0,((len(lines_encoded[0])-window_horizontal)+1)):
            window_chunk = [[0 for i in range(window_vertical)] for j in range(window_horizontal)]
            i=0
            j=0
            for window_iterator_vertical in range(vertical_iterator, vertical_iterator + window_vertical):
                #print(window_iterator_vertical)
                for window_iterator_horizontal in range(horizontal_iterator, horizontal_iterator + window_horizontal):
                    #print(window_iterator_horizontal)
                    window_chunk[i][j] = lines_encoded[window_iterator_vertical][window_iterator_horizontal]
                    j +=1
                i+= 1
                j = 0

            #Chunk created
            print(window_chunk)
            count+=1

            #Creating tensor
            tensor_to_produce = torch.zeros(window_vertical, window_horizontal, tiles_len)
            for iter1 in range(0, window_vertical):
                for iter2 in range(0, window_horizontal):
                    item_at_index = window_chunk[iter1][iter2]
                    #print(item_at_index)
                    if item_at_index == '-':
                        tiles_index = tiles.index(item_at_index)
                        tensor_to_produce[iter1, iter2, tiles_index] = 0

                    elif item_at_index != '-':
                        tiles_index = tiles.index(item_at_index)
                        #print(tiles_index)
                        tensor_to_produce[iter1, iter2, tiles_index] = 1
                        #print(tensor_to_produce[tiles_index, iter1, iter2])
                        count1+=1

            print(tensor_to_produce)
            print(tensor_to_produce.shape)

            #saving tensor if required
            #torch.save(tensor_to_produce, os.path.join(one_hot_tensor_dir, 'one_hot_tensor_{name}_{id}.pth'.format(name = filename, id = count)))


    #Total chunks created
    print("Total number of window chunks created:")
    print(count)
    print("Total changes made:")
    print(count1)
