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
num_epochs = 50
batch_size = 64
learning_rate = 5e-4

# model dimensions 
level_width = 8
level_height = 8
level_depth = 13

# autoencoder dimensions
hidden_layer_dim1 = 32
hidden_layer_dim2 = 8

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
files = glob.glob(input_dir_name + '*.pth')
dataset = torch.zeros(len(files), level_width, level_height, level_depth)

i = 0
for f in files:
    example = torch.load(f)
    dataset[i] = example
    i += 1
    # level_name = f.split("-",1)[1].split(".",1)[0]
    # save_image(example[:,:level_width], './mlp_img_chunked/image_{}_original.png'.format(level_name))
    # dataset[level_name] = example

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(level_width * level_height * level_depth, hidden_layer_dim1),
            nn.ReLU(True),
            nn.Linear(hidden_layer_dim1, hidden_layer_dim2),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_layer_dim2, hidden_layer_dim1),
            nn.ReLU(True),
            nn.Linear(hidden_layer_dim1, level_width * level_height * level_depth),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# convolutional autoencoder dimensions
encoding_dim = 256

class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()
        # encoder
        self.conv1 = nn.Conv2d(13, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 3)
        # self.linear1 = nn.Linear(32 * 4 * 4, encoding_dim)
        # decoder
        # self.linear_trans1 = nn.Linear(encoding_dim,  32 * 4 * 4) 
        self.conv_trans1 = nn.ConvTranspose2d(64, 32, 3)
        self.conv_trans2 = nn.ConvTranspose2d(32, 13, 5)


    def forward(self, x):
        # encode
        x = self.conv1(x)
        x = nn.functional.relu(x)
        # x = x.flatten()
        # x = self.linear1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)

        # decode 
        # x = self.linear_trans1(x)
        # x = nn.functional.relu(x)
        # x = x.view(32, 4, 4)
        x = self.conv_trans1(x)
        x = nn.functional.relu(x)
        x = self.conv_trans2(x)
        x = nn.functional.relu(x)
        return x

def conv_transform(x):
    x = x.permute(2, 0, 1)
    return x.view(1, x.shape[0], x.shape[1], x.shape[2])

def conv_to_level(x):
    x = x.permute(0, 2, 3, 1)
    return x.view(x.shape[1], x.shape[2], x.shape[3])

def to_level(x):
    # x = 0.5 * (x + 1)
    # x = x.clamp(0, 1)
    # x = x.view(x.size(0), 1, 28, 28)
    # return x
    # x = x.round()
    x = x.view(level_height, level_width, level_depth)
    return x

model = ConvAutoEncoder()
loss_function =  nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5
)

for epoch in range(num_epochs):

    # permute the dataset before splitting into batches
    permutation = torch.randperm(dataset.shape[0])

    for i in range(0, dataset.shape[0], batch_size):
        batch_indices = permutation[i:i+batch_size]
        batch = dataset[batch_indices]
        batch = batch.permute(0, 3, 1, 2)
        batch = Variable(batch)

        # forward pass
        output = model(batch)
        loss = loss_function(output, batch)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # save the output occasionally
        # if epoch + 1 == num_epochs:
        #     # lvl = to_level(output.cpu().data)
        #     # save_image(lvl, './mlp_img/image_{}_reconstructed.png'.format(name))
        #     output = to_level(output)
        #     torch.save(output, '{}/tensor_{}.pth'.format(output_dir_name, name))

    # log training
    print('epoch [{}/{}], loss:{:.4f}'
        .format(epoch + 1, num_epochs, loss.data.item()))

level_func = conv_to_level
tranform_func = conv_transform

# save the output tensors
for f in files:
    example = torch.load(f)
    output = model(tranform_func(example))
    output = level_func(output)
    level_name = f.split("-",1)[1].split(".",1)[0]
    torch.save(output, '{}/tensor_{}.pth'.format(output_dir_name, level_name))

torch.save(model.state_dict(), './autoencoder_weights.pth')