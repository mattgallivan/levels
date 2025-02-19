import json
import os

import numpy as np
from skimage import color
import torch
import torch.nn.functional as F

import Inputs
import Visualize


class Network(torch.nn.Module):

    def __init__(self, patch_width, patch_height, num_oh, px):
        super(Network, self).__init__()

        input_size = int(1 * (patch_width * px * patch_height * px * 3) / (1 * 1))
        output_size = patch_width * patch_height * num_oh

        img_w = patch_width * px
        img_h = patch_height * px
        rgb = 3

        self.cv1 = torch.nn.Conv2d(1, 4, 3, padding=1)
        self.pl1 = torch.nn.MaxPool2d(2, 2)
        self.cv2 = torch.nn.Conv2d(4, 8, 3, padding=1)
        self.pl2 = torch.nn.MaxPool2d(2, 2)
        self.cv3 = torch.nn.Conv2d(8, 16, 3, padding=1)
        self.pl3 = torch.nn.MaxPool2d(2, 2)
        self.cv4 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.pl4 = torch.nn.MaxPool2d(2, 2)
        self.dp1 = torch.nn.Dropout(0.5)
        inp = int(3 * img_w * img_h * rgb / (8 * 9))
        hidden = 500
        self.fc1 = torch.nn.Linear(inp, hidden)
        self.fc2 = torch.nn.Linear(hidden, patch_width * patch_height * num_oh)
        
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.num_oh = num_oh

    def forward(self, x):
        out = x.unsqueeze(1)
        out = self.pl1(F.relu(self.cv1(out)))
        out = self.pl2(F.relu(self.cv2(out)))
        out = self.pl3(F.relu(self.cv3(out)))
        out = self.pl4(F.relu(self.cv4(out)))
        out = self.dp1(out)
        out = out.view(out.shape[0], -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = out.reshape(x.shape[0], self.patch_height, self.patch_width, self.num_oh)
        return out


def train_model(levels, px, sprites, sprites_ascii_map, model_path, num_epochs, batch_size, patch_width, patch_height):
    # Load the game data for training
    #game_name = "super-mario-bros-simplified"
    #game_dir = "data/games"
    #lvl_dir = f"{game_dir}/{game_name}/game-levels-ascii"
    #json_dir = f"{game_dir}/{game_name}"    
    #with open(f"{json_dir}/gameMetadata.json") as f:
        #game_data = json.load(f)
    #_, sprites, sprites_ascii_map = Inputs.Get_All_Inputs(game_dir + "/", game_name)    
    
    tiles = list(sprites_ascii_map.keys())

    # Convert the levels and images to patches for training
    lvl_patches = []
    img_patches = []
    for lvl in levels:
        lvl_one_hot = load_level_as_one_hot(lvl, tiles)
        lvl_sprites = ["".join([tile for tile in row] + ["\n"]) for row in one_hot_to_level(lvl_one_hot, tiles)]
        img = np.array(Visualize.visualize(lvl_sprites, sprites, sprites_ascii_map, px))
        img = color.rgb2gray(img)
        lvl_patches.extend(array_to_patches(lvl_one_hot, patch_height, patch_width, 1))
        img_patches.extend(array_to_patches(img, patch_height, patch_width, px))
    
    # Train the model on the levels and images
    model = Network(patch_width, patch_height, len(tiles), px)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    x_train = torch.FloatTensor(np.array(img_patches))
    y_train = torch.FloatTensor(np.array(lvl_patches))
    train(model, criterion, optimizer, x_train, y_train, num_epochs, batch_size)

    torch.save(model.state_dict(), model_path)  

def generate(image, px, sprites_ascii_map, model_path, patch_width, patch_height):
    
    tiles = list(sprites_ascii_map.keys())
    
    model = Network(patch_width, patch_height, len(tiles), px)
    model.load_state_dict(torch.load(model_path))
    
    # Evaluate the input image using the network
    model.eval()
    image = color.rgb2gray(np.asarray(image))
    inp_patches = array_to_patches(image, patch_height, patch_width, px * patch_width)#[0:2]
    patched = []
    #patched_before = []
    
    
    
    
    
    for y in range(0, image.shape[0] - patch_height * px + 1, patch_height * px):
        patched_row = []
        for x in range(0, image.shape[1] - patch_width * px + 1, patch_width * px):
            input_Patch = image[y:y + patch_height * px, x:x + patch_width * px]
            out_patch = np.argmax(model(torch.FloatTensor([input_Patch])).squeeze(0).detach().numpy(), axis=2)
            patched_row.append(out_patch)
        patched.append(np.hstack(patched_row))
    result = np.vstack(patched)
    
    #qqqq = len(inp_patches)
    #for idy in range(0, len(inp_patches)//patch_height):
        #for idx in range(0, len(inp_patches), patch_width):
            #input_Patch = inp_patches[idx]
            #out_patch = np.argmax(model(torch.FloatTensor([input_Patch])).squeeze(0).detach().numpy(), axis=2)
            #patched.append(out_patch)
            ##patched_before.append(input_Patch)
        
    ## TODO: THIS striching SHOULD BE CHANGED:
    #result = np.hstack(patched)
    
    result = [[tiles[t] for t in row] + ['\n'] for row in result]
    #result_before = np.hstack(patched_before)
    #result_before = [[tiles[t] for t in row] + ['\n'] for row in result_before]

    #Chris added:
    newResult = []
    for i in range(0,len(result)):
        newRow = ""
        for j in range(0,len(result[i])):
            newRow += result[i][j]
        newResult.append(newRow)
    result = newResult
    
    return result
    

def train(model, criterion, optimizer, x_train, y_train, num_epochs=1, batch_size=32):
    model.train()
    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        permutation = torch.randperm(x_train.shape[0])
        for i in range(0, x_train.shape[0], batch_size):
            batch_indices = permutation[i:i + batch_size]
            batch_x, batch_y = x_train[batch_indices], y_train[batch_indices]
            output = model(batch_x)
            loss = criterion(output, batch_y)
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def array_to_patches(arr, height, width, step):
    patches = []
    wx = width * step
    wy = height * step    
    for x in range(0, arr.shape[1] - wx + step, step):
        for y in range(0, arr.shape[0] - wy + step, step):
            patch = arr[y:y + wy, x:x + wx]
            patches.append(patch)
    return patches


def load_level_as_one_hot(level_tiles, game_tiles):
    #with open(filepath, "r") as f:
    level_tiles = [
        [game_tiles.index(tile) for tile in line.strip()]
        for line in level_tiles.values()
    ]
    height = len(level_tiles)
    width = len(level_tiles[0])
    one_hot = np.eye(len(game_tiles))[np.array(level_tiles).reshape(-1)]
    level = one_hot.reshape(height, width, len(game_tiles))
    #level = level[1:, :]
    return level


def one_hot_to_level(one_hot, game_tiles):
    level = np.argmax(one_hot, axis=2)
    #print(level)
    level_ascii = []
    for row in level:
        row_ascii = []
        for tile in row:
            row_ascii.append(game_tiles[tile])
        level_ascii.append(row_ascii)
    return level_ascii
