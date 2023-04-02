import matplotlib.pyplot as plt
import numpy as np

label_keys = [x for x in range(10)]
classes_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                 'Ankle boot']
label_dict = dict(zip(label_keys, classes_names))


def image_plotter(dataloader):
    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    fig = plt.figure(figsize=(15, 5))
    for idx in np.arange(20):
        ax = fig.add_subplot(4, 5, idx+1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(images[idx]), cmap='gray')
        ax.set_title(label_dict[labels[idx].item()])
    plt.tight_layout()
    plt.savefig('data/plots/fashionMNIST_train_samples.png', bbox_inches='tight')
    plt.show()
    plt.show()
