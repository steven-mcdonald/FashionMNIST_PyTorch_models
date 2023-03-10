import matplotlib.pyplot as plt
import matplotlib.cm as cm


def data_loader_img_view(images, labels, nsamples=1):

    fig = plt.figure(figsize=(20, 5), facecolor='w')
    for i in range(nsamples):
        ax = plt.subplot(1, nsamples, i+1)
        plt.imshow(images[i, 0, :, :], vmin=0, vmax=1.0, cmap=cm.gray)
        ax.set_title(f"{labels[i]}", fontsize=15)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig('data/plots/fashionMNIST_samples.png', bbox_inches='tight')
    plt.show()
