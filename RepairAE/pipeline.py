import os
import glob

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

import repair
from createleveltry2 import create_level
from join import join_input, join_output, join_output_deterministic
from guzdial_autoencoder import GuzdialConvAutoEncoder
from visualize_level import visualize_level


model_path = './autoencoder_weights_cce.pth'

# 1. train the model
model = repair.ConvAutoEncoder
data = repair.load_data()
train_data, test_data = repair.split_data(data)
labels, data = repair.load_data_categorical()

learning_rate = 1e-3
# repair.train_categorical(labels, data, learning_rate, model, model_path)
# repair.eval_categorical(labels, data, model, model_path)
repair.train(train_data, learning_rate, model, model_path)
repair.eval(test_data, model, model_path)

# 2. generate chunked output tensors 
input_path = './PCGML3/mario_1-1_broken/'
output_path = './PCGML3/mario_1-1_broken_output/'
textfile_dir = 'textfiles/'
repair.output(model, model_path, input_path, output_path, textfile_dir)

# # 3. join the chunks
# one_hot_tensor_dir = os.listdir('./PCGML3/mario_1-1_broken_output/')
# level_dir = "./PCGML3/levels_broken/"
# one_hot_tensor_path = "./PCGML3/mario_1-1_broken_output/"
# create_level(one_hot_tensor_dir, level_dir, one_hot_tensor_path)

# # 4. turn the tensor back to ASCII 
# output_tensor = torch.load('./PCGML3/levels_broken/overall_tensor_output.pth')
# join_output_deterministic(output_tensor, './PCGML3/levels_broken/', 'repaired_output', save=True) 

# 5. visualize the output
# input_dir = './PCGML3/levels_broken/'
# output_dir = './PCGML3/levels_broken/'
# file_name = 'repaired_output'
# visualize_level(input_dir + 'mario-1-1-broken.txt', input_dir + file_name, output_dir + file_name)

