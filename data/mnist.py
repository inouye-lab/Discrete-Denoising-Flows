import torch
import torch.nn as nn
from torch.utils import data
from torchvision import datasets
from torchvision import transforms


class Binarize(nn.Module):
    """Binarize MNIST images"""

    def forward(self, img):
        rand = torch.rand(img.shape, device=img.device)
        return (rand < img).float().long()


def get_mnist_loaders(batch_size):
    data_loader_kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': True, 'batch_size': batch_size}
    ds_transforms = transforms.Compose([transforms.ToTensor(), Binarize()])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("/tmp/data", download=True, train=True, transform=ds_transforms),
        shuffle=True, **data_loader_kwargs)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST("/tmp/data", train=False, transform=ds_transforms),
                                              **data_loader_kwargs)

    return train_loader, test_loader
