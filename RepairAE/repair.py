# adapted from https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/simple_autoencoder.py

import os
import glob

import numpy as np
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image

from join import join_input, join_output, join_output_deterministic
from guzdial_autoencoder import GuzdialConvAutoEncoder

if not os.path.exists('./chunked_data'):
    os.mkdir('./chunked_data')

# set hyperparameters
num_epochs = 100
batch_size = 64

# model dimensions 
level_width = 8
level_height = 8
level_depth = 13

# autoencoder dimensions
hidden_layer_dim1 = 32
hidden_layer_dim2 = 8

def load_data():
    # load mario data
    input_dir_name = 'chunked_data/one hot tensors/'
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
    return dataset

def load_data_categorical():
    # load mario data
    indices_dir = './PCGML_new/one hot indices/'
    tensors_dir = './PCGML_new/one hot tensors/'

    index_files = glob.glob(indices_dir + '*.pth')
    # tensor_files = glob.glob(tensors_dir+ '*.pth')

    indices = torch.zeros(len(index_files), level_width, level_height)
    tensors = torch.zeros(len(index_files), level_width, level_height, level_depth)

    i = 0
    for f in index_files:
        chunk_id = f.split("/")[-1]
        example_index = torch.load(f)
        example_tensor = torch.load(tensors_dir + chunk_id)
        indices[i] = example_index
        tensors[i] = example_tensor
        i += 1 
    return (indices, tensors)

def split_data(data):
    data = torch.from_numpy(np.unique(data, axis=0))
    # split into training and test data
    # set the seed manually to consistently split data
    # torch.random.manual_seed(1337)
    indices = torch.randperm(data.shape[0])
    train_indices = indices[:6000]
    test_indices = indices[6000:]
    train_data = data[train_indices]
    test_data = data[test_indices]
    return (train_data, test_data)

def split_data_categorical(labels, data):
    unique_indices = np.unique(data, return_index=True, axis=0)
    data = data[unique_indices]
    labels = labels[unique_indices]
    # split into training and test data
    # set the seed manually to consistently split data
    # torch.random.manual_seed(1337)

    indices = torch.randperm(data.shape[0])
    train_indices = indices[:6000]
    test_indices = indices[6000:]

    train_data = data[train_indices]
    train_labels = labels[train_indices]
    test_data = data[test_indices]
    test_labels = data[test_indices]
    return (train_labels, train_data, test_labels, test_data)

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

class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()
        # encoder
        self.conv1 = nn.Conv2d(13, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        # decoder
        self.conv_trans1 = nn.ConvTranspose2d(128, 64, kernel_size=3)
        self.conv_trans2 = nn.ConvTranspose2d(64, 32, kernel_size=3)
        self.conv_trans3 = nn.ConvTranspose2d(32, 13, kernel_size=3)


    def forward(self, x):
        # encode
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))

        # decode 
        # x = self.linear_trans1(x)
        # x = nn.functional.relu(x)
        # x = x.view(32, 4, 4)
        x = nn.functional.relu(self.conv_trans1(x))
        x = nn.functional.relu(self.conv_trans2(x))
        x = nn.functional.relu(self.conv_trans3(x))
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

def train(data, learning_rate, model, model_path):
    model = model()
    model = model.train()
    loss_function =  nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate
    )

    for epoch in range(num_epochs):

        # permute the data before splitting into batches
        permutation = torch.randperm(data.shape[0])

        for i in range(0, data.shape[0], batch_size):
            batch_indices = permutation[i:i+batch_size]
            batch = data[batch_indices]
            batch = batch.permute(0, 3, 1, 2)
            batch = Variable(batch)

            # add some random noise to the input
            # batch_noisy = batch + torch.normal(mean=0.0, std=0.5, size=batch.shape)

            # forward pass
            output = model(batch)
            loss = loss_function(output, batch)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # log training
        print('epoch [{}/{}], loss:{:.4f}'
            .format(epoch + 1, num_epochs, loss.data.item()))
    torch.save(model.state_dict(), model_path)

def train_categorical(labels, data, learning_rate, model, model_path):
    model = model()
    model = model.train()
    loss_function =  nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )

    for epoch in range(num_epochs):

        # permute the data before splitting into batches
        permutation = torch.randperm(data.shape[0])

        for i in range(0, data.shape[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch = data[indices]
            batch_labels = labels[indices]
            batch = batch.permute(0, 3, 1, 2)
            batch = Variable(batch)

            # add some random noise to the input
            # batch_noisy = batch + torch.normal(mean=0.0, std=0.5, size=batch.shape)

            # forward pass
            output = model(batch)
            loss = loss_function(output, batch_labels.long())

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # log training
        print('epoch [{}/{}], loss:{:.4f}'
            .format(epoch + 1, num_epochs, loss.data.item()))
    torch.save(model.state_dict(), model_path)

def eval(data, model, model_path):
    model = model() 
    model = model.eval()
    model.load_state_dict(torch.load(model_path))
    loss_function =  nn.MSELoss()

    data = data.permute(0, 3, 1, 2)
    output = model(data)
    loss = loss_function(output, data)
    print("test loss:", loss.data.item())

def eval_categorical(labels, data, model, model_path):
    model = model() 
    model = model.eval()
    model.load_state_dict(torch.load(model_path))
    loss_function =  nn.CrossEntropyLoss()

    data = data.permute(0, 3, 1, 2)
    output = model(data)
    loss = loss_function(output, labels.long())
    print("test loss:", loss.data.item())

def output(model, model_path, input_path, output_path, textfile_dir):
    model = model()
    model.load_state_dict(torch.load(model_path))
    model = model.eval()
    level_func = conv_to_level
    tranform_func = conv_transform

    files = glob.glob(input_path + '*.pth')
    i = 0
    for f in files:
        example = torch.load(f)
        output = model(tranform_func(example))
        output = level_func(output)
        # visualize some tensors 
        if i % 100 == 0:
            print("input")
            join_input(example, input_path + textfile_dir, "input" + str(i))
            print("output")
            join_output_deterministic(output, output_path + textfile_dir, "output" + str(i), save=True)
        i += 1
        level_name = int(f.split("_")[-1].split(".")[0])
        torch.save(output, '{}/{}'.format(output_path, level_name))


# dataset = load_data()
# train_data, test_data = split_data(dataset)

# learning_rate = 1e-4
# train(train_data, GuzdialConvAutoEncoder, learning_rate)
# eval(test_data, GuzdialConvAutoEncoder)
# output()