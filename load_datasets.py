import os
import sys
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split


def import_data(dataset_type):
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

    return dataset

def splitset(dataset, parts):
    n = len(dataset)

    l = []
    for i in range(parts):
        s = int(np.round(n / (parts - i)))
        n -= s
        l += [s]

    t = tuple(l)
    print(t)
    return random_split(dataset, t)


if __name__ == '__main__':

    nr_of_datasets = 10

    trainset = import_data(dataset_type='train')
    trainsets = splitset(trainset, nr_of_datasets)


    if not os.path.exists('data/10clients'):
        os.mkdir('data/10clients')

    for ind,ts in enumerate(trainsets):
        if not os.path.exists('data/10clients/client'+str(ind)):
            os.mkdir('data/10clients/client'+str(ind))
        torch.save(ts, 'data/10clients/client'+str(ind)+'/train.pth')

    testset = import_data(dataset_type='test')
    testsets = splitset(testset, nr_of_datasets)
    if not os.path.exists('data/10clients'):
        os.mkdir('data/10clients')

    for ind,ts in enumerate(testsets):
        if not os.path.exists('data/10clients/client'+str(ind)):
            os.mkdir('data/10clients/client'+str(ind))
        torch.save(ts, 'data/10clients/client'+str(ind)+'/test.pth')