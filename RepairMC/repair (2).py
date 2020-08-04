import sys
import os
import random
import pickle


markovProbabilities = pickle.load(open("smbprobabilities_Lode_Runner.pickle", "rb"))

list_of_loderunner_sprites = ["-", "#", "G", "M", ".", "b", "B", "E"]

def repairLevel(GenLvl, game):
    level = {}
    with open(GenLvl) as fp:
    	y = 0
    	for line in fp:
    		level[y] = line
    		y+=1
    if game == "Super Mario":
        pass
    elif game == "Loderunner":
        repair_loderunner(level)
    else:
        print("nothing to repair!!!!")

def repair_loderunner(level):
    for i in range(0,100):
        for y in range(0,len(level)):
    #            print(len(level[y]))
            for x in range(0, len(level[y])-1):
    #                print(y,x)
                north = " "
                west = " "
                east = " "
                south = " "
                if x == 0 and y == 0:
                    east = level[y][x+1]
                    south = level[y+1][x]
                if x > 0 and x < len(level[y])-2 and y == 0:
                    east = level[y][x+1]
                    south = level[y+1][x]
                    west = level[y][x-1]
                if x == len(level[y])-1 and y == 0:
                    south = level[y+1][x]
                    west = level[y][x-1]
                if x == 0 and y > 0 and y < len(level)-1:
                    north = level[y-1][x]
                    south = level[y+1][x]
                    east = level[y][x+1]
                if x > 0 and x < len(level[y])-2 and y > 0 and y < len(level)-1:
                    north = level[y-1][x]
                    south = level[y+1][x]
                    east = level[y][x+1]
                    west = level[y][x-1]
                if x == len(level[y])-1 and y > 0 and y < len(level)-1:
                    north = level[y-1][x]
                    south = level[y+1][x]
                    west = level[y][x-1]
                if x == 0 and y == len(level) - 1:
                    north = level[y-1][x]
                    east = level[y][x+1]
                if x > 0 and x < len(level[y])-2 and y == len(level) - 1:
                    north = level[y-1][x]
                    east = level[y][x+1]
                    west = level[y][x-1]
                if x == len(level[y])-2 and y == len(level) - 1:
                    north = level[y-1][x]
                    west = level[y][x-1]
                
                key = north+east+south+west
                key = key.replace('\n',' ')
                
                if key in markovProbabilities.keys():
                    # key exists
                    if not level[y][x] in markovProbabilities[key].keys():
                        # resample
                        replaceTile = max(markovProbabilities[key], key=markovProbabilities[key].get)
                        level[y] = level[y][0:x] + replaceTile + level[y][x+1:]
                else:
                    #key does not exist
                    list_of_possible_key = {}
                    for k in range(0,len(key)):
                        for l in list_of_loderunner_sprites:
                            if (key[k] != l):
                                new_key = key[0:k]+l+key[k+1:]
                                if new_key in markovProbabilities.keys():
                                    if level[y][x] in markovProbabilities[new_key].keys():
                                        list_of_possible_key[new_key] = markovProbabilities[new_key][level[y][x]]
                   
                    if len(list_of_possible_key)>0:
                        replace_surrounding_tiles = max(list_of_possible_key, key=list_of_possible_key.get)
                        if y-1 >= 0:
                            level[y-1] = level[y-1][0:x] + replace_surrounding_tiles[0] + level[y-1][x+1:]
                        level[y] = level[y][0:x+1] + replace_surrounding_tiles[1] + level[y][x+2:]
                        if y+1 < len(level):
                            level[y+1] = level[y+1][0:x] + replace_surrounding_tiles[2] + level[y+1][x+1:]
                        if x>0:
                            level[y] = level[y][0:x-1] + replace_surrounding_tiles[3] + level[y][x:]
                    else:
#                        pass
#                        if no combination of key is found with the center piece level[y][x]
                        for k in range(0,len(key)):
                            for l in list_of_loderunner_sprites:
                                if (key[k] != l):
                                    new_key = key[0:k]+l+key[k+1:]
                                    if new_key in markovProbabilities.keys():
                                        for l in list_of_loderunner_sprites:
                                            if level[y][x] != l and l in markovProbabilities[new_key].keys():
                                                list_of_possible_key[new_key+l] = markovProbabilities[new_key][l]
                        if list_of_possible_key:
                            replace_surrounding_tiles = max(list_of_possible_key, key=list_of_possible_key.get)
                            if y-1 >= 0:
                                level[y-1] = level[y-1][0:x] + replace_surrounding_tiles[0] + level[y-1][x+1:]
                            level[y] = level[y][0:x+1] + replace_surrounding_tiles[1] + level[y][x+2:]
                            if y+1 < len(level):
                                level[y+1] = level[y+1][0:x] + replace_surrounding_tiles[2] + level[y+1][x+1:]
                            if x>0:
                                level[y] = level[y][0:x-1] + replace_surrounding_tiles[3] + level[y][x:]
                            level[y] = level[y][0:x] + replace_surrounding_tiles[4] + level[y][x+1:]
#                   
    with open('./prev_repair_output1.txt',"w") as file:
        for y in range(0,len(level)):
            file.write(level[y])
        
        
repairLevel("./lode runner/Good/output_lode_runner1.txt", "Loderunner")