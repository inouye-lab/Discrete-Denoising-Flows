import torch
import torch.nn as nn
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
import utilities
import numpy as np

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

def get_snp_loaders(batch_size,kfolds,fold_num):
    snps_805_path = './data/datasets/805_SNP_1000G_real.hapt.zip' 
    
    all_binary_snps = utilities.preprocess_805_snp_data(snps_805_path)
    
    train_dataset, test_dataset = utilities.create_X_train_test(all_binary_snps,0.8,kfolds,fold_num)
    
    train_loader = data.DataLoader(torch.from_numpy(train_dataset), batch_size=batch_size, shuffle=False)
    test_loader = data.DataLoader(torch.from_numpy(test_dataset), batch_size=1, shuffle=False)
    return train_loader, test_loader
    
def get_cop_loaders(dataset,batch_size,kfolds,fold_num,device):
    
    if(dataset=='coph'):
        cop_path = './data/datasets/coup_data_4_2.npy' 
    elif(dataset=='copm'):
        cop_path = './data/datasets/coup_data_4_2_moderate_corr.npy' 
    elif(dataset=='copw'):
        cop_path = './data/datasets/coup_data_4_2_weak_corr.npy' 
    elif(dataset=='copn'):
        cop_path = './data/datasets/coup_data_4_2_no_corr.npy' 
    
    all_binary_cop = utilities.preprocess_cop(cop_path)
    
    train_dataset, test_dataset = utilities.create_X_train_test(all_binary_cop,0.8,kfolds,fold_num)
    

    train_loader = data.DataLoader(torch.from_numpy(train_dataset).to(device), batch_size=batch_size, shuffle=False)

    test_loader = data.DataLoader(torch.from_numpy(test_dataset).to(device), batch_size=1, shuffle=False)
    
    return train_loader, test_loader

def get_mushroom_loaders(batch_size,kfolds,fold_num):
    mushroom_path = './data/datasets/agaricus-lepiota.data' 
    
    all_mushroom = utilities.process_mushroom_data(mushroom_path)
    
    train_dataset, test_dataset = utilities.create_X_train_test(all_mushroom,0.8,kfolds,fold_num)
    
    train_loader = data.DataLoader(torch.from_numpy(train_dataset), batch_size=batch_size, shuffle=False)

    test_loader = data.DataLoader(torch.from_numpy(test_dataset), batch_size=1, shuffle=False)
    return train_loader, test_loader
    
def get_binary_mnist_loaders(batch_size,kfolds,fold_num):
    all_binary_mnist = utilities.preprocess_binary_mnist()
    
    train_dataset, test_dataset = utilities.create_X_train_test(all_binary_mnist,0.8,kfolds,fold_num)
  
    train_loader = data.DataLoader(torch.from_numpy(train_dataset), batch_size=batch_size, shuffle=False)

    test_loader = data.DataLoader(torch.from_numpy(test_dataset), batch_size=1, shuffle=False)
    return train_loader, test_loader
