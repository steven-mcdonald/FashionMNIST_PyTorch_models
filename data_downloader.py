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

# set the random seed in pytorch and numpy for reproducibility
seed_num = 42
torch.manual_seed(seed_num)
np.random.seed(seed_num)


def get_fashion_dataset():
    # load in the training and test set for Fashion MNIST using torchvision datasets function.
    # this puts the data into a nice format for batching later on.
    train_set = datasets.FashionMNIST(
        root="./data/input/",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]))

    test_set = datasets.FashionMNIST(
        root="./data/input/",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]))

    # split the training data further into training and validation sets.
    # split made reproducible  with the random seed.
    train_set, dev_set = torch.utils.data.random_split(train_set, [50000, 10000], generator=torch.Generator().manual_seed(42))

    return train_set, dev_set, test_set


def create_dataloader(data_set, batch_size=32):
    # Load the dataset as a DataLoader iterator with a given batch size
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size)
    return data_loader


if __name__ == '__main__':

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")

    train, dev, test = get_fashion_dataset()

    # Load the training data as a full batch
    train_fullbatch = create_dataloader(train, batch_size=len(train))
    # from the full batch data extract only the images without the labels
    # these will be the target images
    for data in train_fullbatch:
        train_fullbatch_X, _ = data
        break


    plt.imshow(train_fullbatch_X[0].view(28, 28))
    plt.show()

    print()
