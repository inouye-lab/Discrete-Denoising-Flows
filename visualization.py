import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def indices_cityscapes_to_rgb(indices):
    """
    Map index to color
    """
    colors = np.array([
        (0, 0, 0),
        (128, 64, 128),
        (70, 70, 70),
        (153, 153, 153),
        (107, 142, 35),
        (70, 130, 180),
        (220, 20, 60),
        (0, 0, 142)]
    )
    return colors[indices.astype(int)]


def save_grid_image(model, path, h, w, num_rows=6, dataset=None):
    """
    Sample (num_rows x num_rows) images from model and save then as a grid image
    """
    # sample from model
    samples = model.sample(num_rows ** 2).cpu().detach().numpy()
    samples = samples.reshape(-1, h, w)

    # put samples into grid structure
    img = np.zeros([num_rows * h, num_rows * w])
    for i, sample in enumerate(samples):
        x_pos = i % num_rows
        y_pos = i // num_rows
        img[y_pos * h:y_pos * h + h, x_pos * w:x_pos * w + w] = sample
    if dataset == 'cityscapes':
        img = indices_cityscapes_to_rgb(img)

    # save samples grid as image
    plt.clf()
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    plt.savefig(path)


def plot_2D_samples(path, model, num_samples, vocab_size):
    """
    Sample (num_samples) 2D samples from model and plot them as a 2D histogram
    """
    samples = model.sample(num_samples)

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 1, 1)

    learned_prob_table = np.histogramdd(samples, bins=vocab_size)
    ax1.imshow(learned_prob_table[0] / np.sum(learned_prob_table[0]),
               cmap=cm.get_cmap("Blues", 6),
               origin="lower",
               extent=[0, vocab_size, 0, vocab_size],
               interpolation="nearest")
    plt.tight_layout()
    plt.gcf().savefig(path, dpi=250)
