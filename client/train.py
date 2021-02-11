from __future__ import print_function
import sys
import yaml
import torch
import os
import collections

from data.read_data import read_data


def weights_to_np(weights):

    weights_np = collections.OrderedDict()
    for w in weights:
        weights_np[w] = weights[w].cpu().detach().numpy()
    return weights_np


def np_to_weights(weights_np):
    weights = collections.OrderedDict()
    for w in weights_np:
        weights[w] = torch.tensor(weights_np[w])
    return weights


def train(net, criterion, optimizer, device, data, settings):
    print("-- RUNNING TRAINING --", flush=True)

    # Import data
    trainset = torch.load(os.path.join(data,'train.pth'))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=False, num_workers=2)
    #training
    net.train()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


    return model

if __name__ == '__main__':

    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise(e)

    from fedn.utils.pytorchhelper import PytorchHelper
    from models.pytorch_model import create_seed_model

    if settings['device'] == 'cuda' or settings['device'] == 'gpu':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    print("device: ", device)

    helper = PytorchHelper()
    model, loss, optimizer = create_seed_model()
    model.to(device)
    model.load_state_dict(np_to_weights(helper.load_model(sys.argv[1])))
    model = train(model, loss, optimizer, device, '/app/data/', settings)
    ret = helper.save_model(weights_to_np(model.state_dict()), sys.argv[2])
    print("saved as: ", ret)
