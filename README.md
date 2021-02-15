# FEDn-client-cifar10-pytorch
This repository contains an example client (and compute package and seed file) for FEDn training of a PyTorch VGG16 model, without the top layer (last 3 fully connected layers). We have applied batch normalization between each Convolution and between each ReLU activation layer. 

The CIFAR-10 dataset is evenly divided (IID) into 10 clients (but this can be easily modified). The clients train the model locally one epoch in every round with a batch size of 32, using the Adam optimizer with learning rate 0.001 and usign pre-processed image augmentation -- random crop with padding 4 and horizontal flip.

## Prepare the client configuration

To attach to a FEDn network, first edit 'fedn-network.yaml' to set the endpoint for the controller/reducer. Then edit 'extra-hosts.yaml' to provice dns resolution for each combiner in the FEDn network (cpu version).

## Configure and start a client usign cpu device

The following will help you configure a client on a blank Ubuntu 20.04 LTS VM:    

<script src="https://gist.github.com/ahellander/9046dcd20e1721c7babca6fd8e646733.js"></script>

## Start a client on an Nvidia enabled host
Make sure that you have appropriate Nvidia drivers installed on the host. To start a client using Nvidia GPU:

<script src="https://gist.github.com/ahellander/41fe30e2938a8e63b08423b86c602245.js"></script>
