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

import numpy as np
import matplotlib.pyplot as plt

# set the random seed in pytorch and numpy for reproducibility
seed_num = 42
torch.manual_seed(seed_num)
np.random.seed(seed_num)


def get_fashion_dataset():
    # load in the training and test set for Fashion MNIST using torchvision datasets function.
    # this puts the data into a nice format for batching later on.
    train_dev_set = datasets.FashionMNIST(
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
    train_set, dev_set = torch.utils.data.random_split(train_dev_set, [50000, 10000], generator=torch.Generator().manual_seed(42))

    return train_set, dev_set, test_set


class DatasetTransformer(torch.utils.data.Dataset):

    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.base_dataset[index]
        return self.transform(img), target

    def __len__(self):
        return len(self.base_dataset)


def create_dataloader(data_set, batch_size=32):
    # Load the dataset as a DataLoader iterator with a given batch size
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size)
    return data_loader


if __name__ == '__main__':

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")

    train_dataset, dev_dataset, test_dataset = get_fashion_dataset()

    # train_dataset = DatasetTransformer(train_dataset, transforms.ToTensor())
    # valid_dataset = DatasetTransformer(dev_dataset, transforms.ToTensor())
    # test_dataset = DatasetTransformer(test_dataset, transforms.ToTensor())

    # Dataloaders
    num_threads = 4  # Loading the dataset is using 4 CPU threads
    batch_size = 128  # Using minibatches of 128 samples

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,  # <-- this reshuffles the data at every epoch
                                               num_workers=num_threads)

    dev_loader = torch.utils.data.DataLoader(dataset=dev_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=num_threads)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_threads)

    print("The train set contains {} images, in {} batches".format(len(train_loader.dataset), len(train_loader)))
    print("The validation set contains {} images, in {} batches".format(len(dev_loader.dataset), len(dev_loader)))
    print("The test set contains {} images, in {} batches".format(len(test_loader.dataset), len(test_loader)))

    for i, data in enumerate(train_loader):
        print(i)
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

    print()
