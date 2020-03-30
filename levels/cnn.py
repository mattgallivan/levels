import os
import random

from matplotlib.image import imread
import numpy as np
from PIL import Image
from skimage import color
import torch
from torch.autograd import Variable
import torch.nn.functional as F


def level_to_image(level, sprites, wx, wy):
    image = Image.new("RGB", (16 * wx, 16 * wy))
    image_pixels = image.load()
    for ix in range(wx):
        for iy in range(wy):
            tile = level[iy][ix]
            sprite = sprites["tiles"][tile]
            if not sprite:
                continue
            sprite_pixels = sprite.load()
            for sx in range(sprite.width):
                for sy in range(sprite.height):
                    image_pixels[ix * sprite.width + sx, iy * sprite.height + sy] = sprite_pixels[sx, sy]
    return image


def load_image(filepath, use_grayscale):
    img = Image.open(filepath)
    # img.thumbnail((128, 128), Image.ANTIALIAS)
    if use_grayscale:
        arr = color.rgb2gray(np.asarray(img))
    else:
        arr = np.asarray(img.convert("RGB"))
        print(arr.shape)
    return arr


def load_level(filepath, game_tiles):
    with open(filepath, "r") as f:
        level_tiles = [
            [game_tiles.index(tile) for tile in line.strip()]
            for line in f
        ]
        height = len(level_tiles)
        width = len(level_tiles[0])
        one_hot = np.eye(len(game_tiles))[np.array(level_tiles).reshape(-1)]
        level = one_hot.reshape(height, width, len(game_tiles))
        level = level[1:, :]
        return level


def one_hot_to_level(one_hot, game_tiles):
    # print("MAX", np.mean(one_hot, axis=2))
    # level = np.argmax(one_hot, axis=2)
    # print(level)
    # print(one_hot)
    # level = np.apply_along_axis(
        # lambda prb: np.random.choice(range(len(prb)), p=prb / sum(prb)), 2, one_hot)    
    level = np.argmax(one_hot, axis=2)
    print(level)
    level_ascii = []
    for row in level:
        row_ascii = []
        for tile in row:
            row_ascii.append(game_tiles[tile])
        level_ascii.append(row_ascii)
    return level_ascii

game_path = "data/games/smb"
sprites = {
    "path": f"{game_path}/sprites",
    "tiles": {
        "X" : "ground.png",
        "S" : "brick.png",
        "-" : None,
        "?" : "questionMark.png",
        "Q" : "emptyBlock.png",
        "E" : "goomba.png",
        "<" : "topLeftPipe.png",
        ">" : "topRightPipe.png",
        "[" : "leftPipe.png",
        "]" : "rightPipe.png",
        "o" : "coin.png",
        "B" : "cannonTop.png",
        "b" : "cannonBottom.png",
    },
}
for tile, sprite_path in sprites["tiles"].items():
    if sprite_path:
        image = Image.open(f'{sprites["path"]}/{sprite_path}')
        sprites["tiles"][tile] = image

game_tiles = [
    "X",
    "S",
    "-",
    "?",
    "Q",
    "E",
    "<",
    ">",
    "[",
    "]",
    "o",
    "B",
    "b",
]


# Load the images as numpy arrays
image_dir = "data/games/smb/level-images"
images = {
    name.split(".")[0]: load_image(f"{image_dir}/{name}", True)
    for name in os.listdir(image_dir)
}
true_images = {
    name.split(".")[0]: load_image(f"{image_dir}/{name}", False)
    for name in os.listdir(image_dir)
}

# Load the levels as numpy arrays
level_dir = "data/games/smb/level-ascii"
levels = {
    name.split(".")[0]: load_level(f"{level_dir}/{name}", game_tiles)
    for name in os.listdir(level_dir)
}

# Slice them into patches
wx = 20
wy = 13
level_patches = []
for name, level in levels.items():
    for x in range(0, level.shape[1] - wx + 1, 1):
        for y in range(0, level.shape[0] - wy + 1, 1):
            patch = level[y:y + wy, x:x + wx]
            level_patches.append(patch)

wxi = wx * 16
wyi = wy * 16
image_patches = []
for name, level in levels.items():
    image = images[name]
    for x in range(0, image.shape[1] - wxi + 16, 16):
        for y in range(0, image.shape[0] - wyi + 16, 16):
            patch = image[y:y + wyi, x:x + wxi]
            image_patches.append(patch)

true_image_patches = []
for name, level in levels.items():
    image = true_images[name]
    for x in range(0, image.shape[1] - wxi + 16, 16):
        for y in range(0, image.shape[0] - wyi + 16, 16):
            patch = image[y:y + wyi, x:x + wxi, :]
            true_image_patches.append(patch)

# Train the network
class Simple(torch.nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(Simple, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = x.view(x.shape[0], -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = out.reshape(x.shape[0], wy, wx, 13)
        return out

class Conv(torch.nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(Conv, self).__init__()
        self.cv1 = torch.nn.Conv2d(1, 4, 3, padding=1)
        self.pl1 = torch.nn.MaxPool2d(2, 2)
        self.cv2 = torch.nn.Conv2d(4, 8, 3, padding=1)
        self.pl2 = torch.nn.MaxPool2d(2, 2)
        self.cv3 = torch.nn.Conv2d(8, 16, 3, padding=1)
        self.pl3 = torch.nn.MaxPool2d(2, 2)
        self.cv4 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.pl4 = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = x.unsqueeze(1)
        out = self.pl1(F.relu(self.cv1(out)))
        out = self.pl2(F.relu(self.cv2(out)))
        out = self.pl3(F.relu(self.cv3(out)))
        out = self.pl4(F.relu(self.cv4(out)))
        out = out.view(out.shape[0], -1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        out = out.reshape(x.shape[0], wy, wx, 13)
        return out

model = Conv(int((wxi * wyi) / 8), 250, wx * wy * 13)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

x_train = torch.FloatTensor(image_patches)
y_train = torch.FloatTensor(level_patches)

num_epochs = 20
batch_size = 16

'''
model.train()
for epoch in range(num_epochs):
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

torch.save(model.state_dict(), "model.pt")
'''
model.load_state_dict(torch.load("model.pt"))
model.eval()

# Generate an example level from an existing image

for idx in range(len(level_patches)):
    print(idx)
    xs = model(x_train[idx].unsqueeze(0))
    level_new = xs.squeeze(0).detach().numpy()
    level_old = y_train[idx].detach().numpy()
    img_old = true_image_patches[idx]

    level_ascii_new = one_hot_to_level(level_new, game_tiles)
    level_ascii_old = one_hot_to_level(level_old, game_tiles)

    img = level_to_image(level_ascii_new, sprites, wx, wy)

    print(img_old.shape)
    old = Image.fromarray(img_old)

    os.mkdir("outimg/" + str(idx))
    img.save("outimg/" + str(idx) + "/output.png", "PNG")
    old.save("outimg/" + str(idx) + "/original.png", "PNG")

    with open("outimg/" + str(idx) + "/original.txt", "w") as f:
        f.writelines("".join(row) + "\n" for row in level_ascii_old)

    with open("outimg/" + str(idx) + "/output.txt", "w") as f:
        f.writelines("".join(row) + "\n" for row in level_ascii_new)


# 2, 21, 102

'''
with open('example.txt', 'w') as file:
    file.writelines("".join(row) + "\n" for row in level_ascii_new)
    file.writelines("".join(row) + "\n" for row in level_ascii_old)
'''

# Compare to the original level
