import sys
from data.read_data import read_data
import json
import yaml
import torch
import os
import collections


def np_to_weights(weights_np):
    weights = collections.OrderedDict()
    for w in weights_np:
        weights[w] = torch.tensor(weights_np[w])
    return weights

def validate(model, settings, device):
    print("-- RUNNING VALIDATION --", flush=True)
    # The data, split between train and test sets. We are caching the partition in 
    # the container home dir so that the same data subset is used for 
    # each iteration.

    def evaluate(net, criterion, dataloader):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 1.0*correct / total
        loss = test_loss/total
        return float(loss), float(acc)

    # Training error (Client validates global model on same data as it trains on.)
    try:
        testset = torch.load("/app/data/test.pth")
        trainset = torch.load("/app/data/train.pth")

    except:
        pass

    test_loader = torch.utils.data.DataLoader(testset, batch_size=settings['batch_size'],
                                             shuffle=False, num_workers=2)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=settings['batch_size'],
                                              shuffle=False, num_workers=2)
    try:
        training_loss, training_acc = evaluate(model, loss, train_loader)
        test_loss, test_acc = evaluate(model, loss, test_loader)

    except Exception as e:
        print("failed to validate the model {}".format(e), flush=True)
        raise
    
    report = { 
                "classification_report": 'unevaluated',
                "training_loss": training_loss,
                "training_accuracy": training_acc,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
            }

    print("-- VALIDATION COMPLETE! --", flush=True)
    return report

if __name__ == '__main__':

    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise(e)

    from fedn.utils.pytorchhelper import PytorchHelper
    from models.pytorch_model import create_seed_model

    if settings['val_device'] == 'cuda' or settings['val_device'] == 'gpu':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    helper = PytorchHelper()
    model, loss, optimizer = create_seed_model()
    model.to(device)
    model.load_state_dict(np_to_weights(helper.load_model(sys.argv[1])))
    report = validate(model, settings, device)

    with open(sys.argv[2], "w") as fh:
        fh.write(json.dumps(report))

