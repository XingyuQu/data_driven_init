'''
Loads ANN model and extracts feature vectors at the hidden layer and also plots histogram

Loads ANN model and copies the weights to an SNN model. 
'''


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from spiking_layer_ours import *
from Models import modelpool
from Preprocess import datapool
from torchvision.models.feature_extraction import create_feature_extractor
import os
import argparse
from funcs import *
import numpy as np
import time


parser = argparse.ArgumentParser(description='PyTorch ANN-SNN Conversion')

parser.add_argument('--t', default=300, type=int, help='T Latency length (Simulation time-steps)')
parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset name',
                    choices=['cifar10', 'cifar100', 'tiny-imagenet','imagenet','fashion'])
parser.add_argument('--model', default='vgg16', type=str, help='Model name',
                    choices=['small','vgg16', 'resnet18', 'resnet20','vgg16_no_bn',
                             'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg16_normed','alexnet',
                             'resnet18','resnet19', 'resnet20', 'resnet34', 'resnet50', 'resnet101', 'resnet152','cifarnet'])
parser.add_argument('--checkpoint', default='./saved_models', type=str, help='Directory for saving models')
parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
parser.add_argument('--wd', default=5e-4, type=float, help='Weight decay')

args = parser.parse_args()
args.mid = f'{args.dataset}_{args.model}'
savename = os.path.join(args.checkpoint, args.mid)+"_new"#+"no_bn"#+"no_bias"+"no_affine"#+"_no_affine"#+"_no_affine"+"_bn"
#print('%s_threshold_removed%d.npy'%(savename,2))
# Check Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Define Hyper-parameters 
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 300
batch_size =256



model = modelpool(args.model, args.dataset)
print(model)

criterion = nn.CrossEntropyLoss()

model.to(device)
train_loader, test_loader = datapool(args.dataset, batch_size,2,shuffle=True)
train_ann(train_loader, test_loader, model, 300, device, criterion, args.lr, args.wd, savename)      

