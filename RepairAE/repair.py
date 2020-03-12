# adapted from https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/simple_autoencoder.py

import os
import glob

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')

# set hyperparameters
num_epochs = 100
batch_size = 128
learning_rate = 1e-3

# model dimensions 
level_width = 149
level_height = 14
encoding_dim = 32

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# load MNIST data
# dataset = MNIST('./data', transform=img_transform, download=True)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# load mario data
dir_name = 'PCGML/mario_tensors/*.pth'
dataset = {}
for f in glob.glob(dir_name):
    example = torch.load(f)
    level_name = f[-7:-4]
    save_image(example[:,:level_width], './mlp_img/image_{}_original.png'.format(level_name))
    dataset[level_name] = example

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(level_width * level_height, encoding_dim),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, level_width * level_height),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def to_level(x):
    # x = 0.5 * (x + 1)
    # x = x.clamp(0, 1)
    # x = x.view(x.size(0), 1, 28, 28)
    # return x
    x = x.round()
    x = x.view(level_height, level_width)
    return x

model = AutoEncoder()
loss_function =  nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5
)

for epoch in range(num_epochs):
    for name in dataset:
        level = dataset[name][:,:level_width]

        # reshape the img tensor
        level = level.flatten()
        level = Variable(level)

        # forward pass
        output = model(level)
        loss = loss_function(output, level)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # save the output occasionally
        if epoch + 1 == num_epochs:
            lvl = to_level(output.cpu().data)
            save_image(lvl, './mlp_img/image_{}_reconstructed.png'.format(name))

    # log training
    print('epoch [{}/{}], loss:{:.4f}'
        .format(epoch + 1, num_epochs, loss.data.item()))

torch.save(model.state_dict(), './autoencoder_weights.pth')