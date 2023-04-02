# conda install pytorch using $ conda install pytorch torchvision -c pytorch

import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import transforms, datasets
# object orientated aspects of pytorch
import torch.nn as nn
# functional aspects of pytorch (interchangeable with above just a funct approach rather than object approach)
import torch.nn.functional as F
import torch.optim as optim

from image_plotter import image_plotter
from LinearNet import LinearNet
from DeepNet import DeepNet
from train import train
from test import test
from val import val

import numpy as np

# set the random seed in pytorch and numpy for reproducibility
seed_num = 42
torch.manual_seed(seed_num)
np.random.seed(seed_num)

classes_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                 'Ankle boot']


def get_fashion_dataset():
    # load in the training and test set for Fashion MNIST using torchvision datasets function.
    # this puts the data into a nice format for batching later on.
    train_val_set = datasets.FashionMNIST(
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
    train_set, val_set = torch.utils.data.random_split(train_val_set, [50000, 10000], generator=torch.Generator().manual_seed(42))

    return train_set, val_set, test_set


if __name__ == '__main__':

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # get datasets
    train_dataset, val_dataset, test_dataset = get_fashion_dataset()

    # set dataloaders
    num_threads = 4  # Loading the dataset is using 4 CPU threads
    batch_size = 128  # Using minibatches of 128 samples

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_threads)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_threads)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_threads)

    print("The train set contains {} images, in {} batches".format(len(train_loader.dataset), len(train_loader)))
    print("The validation set contains {} images, in {} batches".format(len(val_loader.dataset), len(val_loader)))
    print("The test set contains {} images, in {} batches".format(len(test_loader.dataset), len(test_loader)))

    # plot some training images with labels
    image_plotter(train_loader)

    # instantiate model
    # model = LinearNet(1 * 28 * 28, 10)
    model = DeepNet(10)

    # model to device
    model.to(device)
    # set loss function
    f_loss = torch.nn.CrossEntropyLoss()
    # set optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # set model to train
    model.train()
    # set number of training epochs
    EPOCHS = 6

    # train the model
    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, f_loss, optimizer, device)
        val_loss, val_acc = val(model, val_loader, f_loss, device)
        # print loss and accuracy after each epoch
        print(f"epoch {epoch}: train loss {train_loss:.4f} : val loss {val_loss:.4f} : val acc {val_acc:.4f}")

    # test the model
    test_loss, test_acc = test(model, test_loader, f_loss, device)

    print()

    model.eval()
    for i in range(10):
        x, y = test_dataset[i][0], test_dataset[i][1]
        with torch.no_grad():
            pred = model(x)
            predicted, actual = classes_names[pred[0].argmax(0)], classes_names[y]
            print(f'Predicted: "{predicted}", Actual: "{actual}"')

    print()
