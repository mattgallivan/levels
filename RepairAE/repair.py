# adapted from https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/simple_autoencoder.py

import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

# set hyperparameters
num_epochs = 5 
batch_size = 128
learning_rate = 1e-3

# model dimensions 
image_dim = 28
encoding_dim = 32

# load data
dataset = None #TODO
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(image_dim * image_dim, encoding_dim),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, image_dim * image_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = AutoEncoder()
loss =  nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5
)

for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        # reshape the img tensor
        img = img.view(img.size(0), -1)
        img = Variable(img)