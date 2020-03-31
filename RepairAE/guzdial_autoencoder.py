import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

# originally attempted to replicate the architecture used in https://arxiv.org/pdf/1809.09419.pdf
encoding_dim = 64

class GuzdialConvAutoEncoder(nn.Module):
    def __init__(self):
        super(GuzdialConvAutoEncoder, self).__init__()
        # encoder
        self.conv1 = nn.Conv2d(13, 64, kernel_size=3, padding=(1,1))
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=(1,1))
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=(1,1))
        self.linear1 = nn.Linear(128 * 2 * 2, encoding_dim)


        # decoder
        self.linear_trans1 = nn.Linear(encoding_dim,  128 * 2 * 2) 
        # self.conv_trans1 = nn.ConvTranspose2d(128, 64, kernel_size=4)
        self.conv_trans1 = nn.ConvTranspose2d(128, 64, kernel_size=4)
        self.conv_trans2 = nn.ConvTranspose2d(64, 13, kernel_size=4)

        self.drop_out = nn.Dropout()


    def forward(self, x):
        # encode
        x = self.mp1(self.conv1(x))
        x = nn.functional.relu(x)
        # x = self.drop_out(x)
        x = self.mp2(self.conv2(x))
        x = nn.functional.relu(x)
        x = self.linear1(x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
        x = nn.functional.relu(x)

        # decode 
        x = self.linear_trans1(x)
        x = nn.functional.relu(x)
        x = self.conv_trans1(x.view(x.shape[0], 128, 2, 2))
        x = nn.functional.relu(x)
        x = self.conv_trans2(x)
        x = nn.functional.relu(x)
        return x