import logging
import random

import numpy as np
import torch
import torch.utils.data as data

logger = logging.getLogger(__name__)


def entropy(P):
    return - sum([p * np.log(p) for p in P])


def sample_quantized_gaussian_mixture(batch_size):
    clusters = np.array([[2., 0.], [np.sqrt(2), np.sqrt(2)],
                         [0., 2.], [-np.sqrt(2), np.sqrt(2)],
                         [-2., 0.], [-np.sqrt(2), -np.sqrt(2)],
                         [0., -2.], [np.sqrt(2), -np.sqrt(2)]])
    assignments = torch.distributions.OneHotCategorical(
        logits=torch.zeros(8, dtype=torch.float32)).sample([batch_size])
    means = torch.matmul(assignments, torch.from_numpy(clusters).float())

    samples = torch.distributions.normal.Normal(loc=means, scale=0.1).sample()
    clipped_samples = torch.clamp(samples, -2.25, 2.25)
    quantized_samples = (torch.round(clipped_samples * 20) + 45).long()

    data_u = np.unique(quantized_samples.numpy(), axis=0).tolist()
    data = quantized_samples.numpy().tolist()
    P = [data.count(d) / len(data) for d in data_u]
    e = entropy(P)
    logger.info(f'Distribution Entropy: {e}   [BPD: {e / 2 / np.log(2)}]')

    return quantized_samples


class EightGaussiansDataset(data.Dataset):
    def __init__(self, length, vocab_size):
        # create data
        self.tensors = sample_quantized_gaussian_mixture(length)
        self.indv_samples = torch.from_numpy(np.unique(self.tensors, axis=0)).float()
        self.length = length
        self.vocab_size = vocab_size

    def __getitem__(self, index):
        idx = random.randint(0, self.tensors.shape[0] - 1)
        x = self.tensors[idx]
        return x

    def __len__(self):
        return self.length


def get_eightgaussians(num_classes, batch_size):
    train_dataset = EightGaussiansDataset(10_000, num_classes)
    trainloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return trainloader
