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

from image_plotter import data_loader_img_view
from LinearNet import LinearNet
from train import train
from test import test
from val import val

import numpy as np

# set the random seed in pytorch and numpy for reproducibility
seed_num = 42
torch.manual_seed(seed_num)
np.random.seed(seed_num)


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
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")

    train_dataset, val_dataset, test_dataset = get_fashion_dataset()

    # Dataloaders
    num_threads = 4  # Loading the dataset is using 4 CPU threads
    batch_size = 128  # Using minibatches of 128 samples

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,  # <-- this reshuffles the data at every epoch
                                               num_workers=num_threads)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=num_threads)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_threads)

    print("The train set contains {} images, in {} batches".format(len(train_loader.dataset), len(train_loader)))
    print("The validation set contains {} images, in {} batches".format(len(val_loader.dataset), len(val_loader)))
    print("The test set contains {} images, in {} batches".format(len(test_loader.dataset), len(test_loader)))

    for i, data in enumerate(train_loader):
        X_sample, y_sample = data
        break

    nsamples = 10

    # create a dictionary to map the y values to clothing class labels
    label_keys = [x for x in range(10)]
    classes_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                     'Ankle boot']
    label_dict = dict(zip(label_keys, classes_names))

    labels = [label_dict[k] for k in y_sample.tolist()]

    # data_loader_img_view(X_sample, labels, nsamples)

    model = LinearNet(1 * 28 * 28, 10)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model.to(device)

    f_loss = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters())

    # model.parameters() are all the learnable parameters of the net
    # lr is the learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    EPOCHS = 6

    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, f_loss, optimizer, device)
        val_loss, val_acc = val(model, val_loader, f_loss, device)
        # print loss and accuracy after each epoch
        print(f"epoch {epoch}: train loss {train_loss:.4f} : val loss {val_loss:.4f} : val acc {val_acc:.4f}")

    test(model, test_loader, f_loss, device)

    print()

    model.eval()
    for i in range(10):
        x, y = test_dataset[i][0], test_dataset[i][1]
        with torch.no_grad():
            pred = model(x)
            predicted, actual = classes_names[pred[0].argmax(0)], classes_names[y]
            print(f'Predicted: "{predicted}", Actual: "{actual}"')

    print()
