import os
import torch
import pathlib
from pathlib import Path

#tiles = ["X", "S", "-", "?", "Q", "E", "<", ">", "[", "]", "o", "B", "b"]
#tiles_len = len(tiles)
#lines = []

def generate_one_hot(level, output_path, levelname, asciiMapping):
    filename = levelname
    tiles = list(asciiMapping.keys())
    tiles_len = len(tiles)
    
    lines_encoded = []
    for line in level:
        line_after_encode=[]
        for i in line:
            if i != "\n":
                line_after_encode.append(i)
            elif i == "\n":
                pass
        lines_encoded.append(line_after_encode)
    #print(lines_encoded)

    #Dimensions
    #print(len(lines_encoded))
    #print(len(lines_encoded[0]))

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
            #print(window_chunk)
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

            #print(tensor_to_produce)
            #print(tensor_to_produce.shape)

            #saving tensor if required
            tensor_save_path = os.path.join(output_path, 'one_hot_tensor_{name}_{id}.pth'.format(name = filename, id = count))
            torch.save(tensor_to_produce, tensor_save_path)
            if(count>2000):
                break


    #Total chunks created
    print("Total number of window chunks created:")
    print(count)
    print("Total changes made:")
    print(count1)
    
    return lines_encoded

def generate_one_hot_stride_2(level, output_path, levelname, asciiMapping):
    filename = levelname
    tiles = list(asciiMapping.keys())
    tiles_len = len(tiles)

    lines_encoded = []
    for line in level:
        line_after_encode=[]
        for i in line:
            if i != "\n":
                line_after_encode.append(i)
            elif i == "\n":
                pass
        lines_encoded.append(line_after_encode)

    is_odd_length = False
    # if the length of the lines is odd, we need to temporarily add an extra character
    if len(lines_encoded[0]) % 2 != 0:
        for l in lines_encoded:
            l.append("-")
        is_odd_length = True

    #Dimensions
    # print(len(lines_encoded))
    # print(len(lines_encoded[0]))

    #Window Dimensions
    window_vertical = 8
    window_horizontal = 8
    count = 0
    count1 = 0
    for vertical_iterator in range(0, ((len(lines_encoded)-window_vertical)+1), 2):
        for horizontal_iterator in range(0,((len(lines_encoded[0])-window_horizontal)+1), 2):
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
            # print(window_chunk)
            count+=1

            #Creating tensor
            tensor_to_produce = torch.zeros(window_vertical, window_horizontal, tiles_len)
            tensor_to_produce_new = torch.zeros(window_vertical, window_horizontal)
            for iter1 in range(0, window_vertical):
                for iter2 in range(0, window_horizontal):
                    item_at_index = window_chunk[iter1][iter2]
                    tiles_index = tiles.index(item_at_index)
                    tensor_to_produce[iter1, iter2, tiles_index] = 1
                    tensor_to_produce_new[iter1, iter2] = tiles_index
                    count1+=1

            # print(tensor_to_produce)
            # print(tensor_to_produce.shape)

            # print(tensor_to_produce_new)
            # print(tensor_to_produce_new.shape)

            #saving tensor if required
            torch.save(tensor_to_produce, os.path.join(output_path, 'one_hot_tensor_{name}_{id}.pth'.format(name = filename, id = count)))
            # torch.save(tensor_to_produce_new, os.path.join(output_path, '{name} {id}.pth'.format(name = filename, id = count)))
            #torch.save(tensor_to_produce, os.path.join(one_hot_tensor_dir, '{id}.pth'.format(name = filename, id = count)))
            #torch.save(tensor_to_produce_new, os.path.join(one_hot_tensor_dir_new, '{id}.pth'.format(name = filename, id = count)))
        print(vertical_iterator)

    #Total chunks created
    print("Total number of window chunks created:")
    print(count)
    print("Total changes made:")
    print(count1)

    return lines_encoded, is_odd_length