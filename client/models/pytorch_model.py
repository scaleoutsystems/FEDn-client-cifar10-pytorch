from torch import nn
import torch
import torch.nn.functional as F

from models.pytorch_models import *
# Create an initial CNN Model
def create_seed_model(net='VGG16'):

    try:
        model = VGG(net)
    except:
        model = VGG('VGG16')


    #loss = nn.NLLLoss()
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-07)
    return model, loss, optimizer



