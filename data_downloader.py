# conda install pytorch using $ conda install pytorch torchvision -c pytorch

import torch
import torchvision
from torchvision import transforms, datasets
# object orientated aspects of pytorch
import torch.nn as nn
# functional aspects of pytorch (interchangeable with above just a funct approach rather than object approach)
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

# Data pre-processing

# load in the training and test set for Fashion MNIST using torchvision datasets function.
# this puts the data into a nice format for batching later on. Equivalent torchtext for the Stanford Sentiment Treebank
train_set = datasets.FashionMNIST(
    root="./data/input/",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()]),
)

test_set = datasets.FashionMNIST(
    root="./data/input/",
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()]),
)

print()
