import numpy as np
import os
import torch
import torch.nn as nn
import pathlib
from pathlib import Path

level_dir = "./mario levels/"
tensor_dir = "./mario tensors/"
windows_dir = "./windows/"
#file1 = open("mario-1-1.txt", "r")

#lines = []
#for line in file1:
    #line_read = line.split(',')
    #line_to_write = [n for n in line_read]
    #lines.append(line_to_write)
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
        line_after_encode = []
        for eachline in line:
            for i in eachline:
                if i == "-":
                    line_after_encode.append(0)

                elif i != "-" and i != "\n":
                    line_after_encode.append(1)

                elif i == '\n':
                    pass
            lines_encoded.append(line_after_encode)
    print(lines_encoded)
    len_vertical = len(lines_encoded)
    print(len_vertical)
    len_horizontal = len(lines_encoded[0])
    print(len_horizontal)
    tensor_to_produce = torch.zeros(len_vertical, len_horizontal)
    #print(tensor_to_produce)
    i = 0
    j = 0
    for line in lines_encoded:
        for eachpos in line:
            if eachpos == 1:
                tensor_to_produce[j][i] = 1
                i+=1

            elif eachpos == 0:
                tensor_to_produce[j][i] = 0
                i+=1
        #print(tensor_to_produce[j][:])
        j+= 1
        i = 0
    print(tensor_to_produce)
    #torch.save(tensor_to_produce, os.path.join(tensor_dir, 'tensor_{name}.pth'.format(name = filename)))

    window_vertical = 14
    #window_horizontal = int(input("Enter horizontal window size:"))
    window_horizontal = 1
    window_tensor_concatenated = torch.zeros(window_vertical, window_horizontal)
    #window_tensor_concatenated = torch.empty()
    for i in range(0, (len_horizontal - window_horizontal)):
        window_tensor = torch.zeros(window_vertical, window_horizontal)

        if i == 0:
            window_tensor_concatenated = tensor_to_produce[:, i:i+window_horizontal]

        window_tensor = tensor_to_produce[:, i:i+window_horizontal]
        window_tensor_concatenated = torch.cat((window_tensor_concatenated, window_tensor), dim = 1)
    #torch.save(window_tensor_concatenated, os.path.join(windows_dir, 'window_concatenated_{name}.pth'.format(name = filename)))
        #print(window_tensor_concatenated)
    print(window_tensor_concatenated.shape)


    #print(tensor_to_produce.shape)
    current_file.close()
#torch.save(tensor_to_produce,"tensor_mario_1_1.pth")
