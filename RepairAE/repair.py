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

if not os.path.exists('./chunked_data'):
    os.mkdir('./chunked_data')

# set hyperparameters
num_epochs = 5
batch_size = 128
learning_rate = 1e-3

# model dimensions 
level_width = 8
level_height = 8
level_depth = 13
encoding_dim = 32

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# load MNIST data
# dataset = MNIST('./data', transform=img_transform, download=True)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# load mario data
input_dir_name = 'chunked_data/one hot tensors/'
output_dir_name = 'chunked_data/output_tensors/'
dataset = {}
for f in glob.glob(input_dir_name + '*.pth'):
    example = torch.load(f)
    level_name = f.split("-",1)[1].split(".",1)[0]
    # save_image(example[:,:level_width], './mlp_img_chunked/image_{}_original.png'.format(level_name))
    dataset[level_name] = example

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(level_width * level_height * level_depth, encoding_dim),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, level_width * level_height * level_depth),
            nn.ReLU(),
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
    # x = x.round()
    x = x.view(level_height, level_width, level_depth)
    return x

model = AutoEncoder()
loss_function =  nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5
)

for epoch in range(num_epochs):
    for name in dataset:
        level = dataset[name]
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
            # lvl = to_level(output.cpu().data)
            # save_image(lvl, './mlp_img/image_{}_reconstructed.png'.format(name))
            output = to_level(output)
            torch.save(output, '{}/tensor_{}.pth'.format(output_dir_name, name))

    # log training
    print('epoch [{}/{}], loss:{:.4f}'
        .format(epoch + 1, num_epochs, loss.data.item()))

torch.save(model.state_dict(), './autoencoder_weights.pth')