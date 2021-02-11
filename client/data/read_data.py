import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

def read_data(filename, nr_examples=1000, batch_size=100, dataset_type='test'):
    """ Helper function to read and preprocess data for training with pytorch. """

    print("inside read data: filename: ", filename)

    if dataset_type == 'train':
        transform_data = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        training_phase = True
    else:
        transform_data = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        training_phase = False

    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=training_phase, download=True, transform=transform_data)

    n = len(dataset.__dict__['targets'])

    print("n: ", n,  ", nr_examples: ", nr_examples)
    subset1, subset2 = random_split(dataset, (nr_examples,n-nr_examples))

    dataloader = torch.utils.data.DataLoader(subset1, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    print("type dataloader: ", type(dataloader))
    return dataloader

