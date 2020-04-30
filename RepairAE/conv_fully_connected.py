import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

# originally attempted to replicate the architecture used in https://arxiv.org/pdf/1809.09419.pdf
embedding_dim = 512

class ConvFullyConnected(nn.Module):
    def __init__(self):
        super(ConvFullyConnected, self).__init__()
        # encoder
        self.conv1 = nn.Conv2d(13, 64, kernel_size=2)
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.linear1 = nn.Linear(128 * 5 * 5, embedding_dim)


        # decoder
        self.linear_trans1 = nn.Linear(embedding_dim,  128 * 5 * 5) 
        self.conv_trans1 = nn.ConvTranspose2d(128, 64, kernel_size=3)
        self.conv_trans2 = nn.ConvTranspose2d(64, 13, kernel_size=2)

        self.drop_out = nn.Dropout(p=0.0)


    def forward(self, x):
        # encode
        x = torch.tanh(self.conv1(x))
        # x = self.mp(x)
        x = torch.tanh(self.conv2(x))
        # x = self.mp(x)
        x = self.linear1(x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
        x = torch.tanh(x)

        # decode
        x = self.linear_trans1(x)
        x = torch.tanh(x.view(x.shape[0], 128, 5, 5))
        x = torch.tanh(self.conv_trans1(x))
        x = torch.relu(self.conv_trans2(x))
        return x