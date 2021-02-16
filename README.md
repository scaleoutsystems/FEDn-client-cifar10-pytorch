# FEDn-client-cifar10-pytorch
This repository contains an example client (and compute package and seed file) for FEDn training of a PyTorch VGG16 model, without the top layer (last 3 fully connected layers). We have applied batch normalization between each Convolution and between each ReLU activation layer. 

The CIFAR-10 dataset is evenly divided (IID) into 10 clients (but this can be easily modified). The clients train the model locally one epoch in every round with a batch size of 32, using the Adam optimizer with learning rate 0.001 and usign pre-processed image augmentation -- random crop with padding 4 and horizontal flip.

## Prepare the client configuration

To attach to a FEDn network, first edit 'fedn-network.yaml' to set the endpoint for the controller/reducer. Then edit 'extra-hosts.yaml' to provice dns resolution for each combiner in the FEDn network (cpu version).

If you first need to deploy a FEDn network, follow the instructions here: https://github.com/scaleoutsystems/fedn 

## Configure and start a client using cpu device

The following shell script will configure and start a client on a blank Ubuntu 20.04 LTS VM:    

```bash
#!/bin/bash

# Install Docker and docker-compose
sudo apt-get update
sudo apt-get install docker -y
sudo apt-get install docker-compose -y
sudo apt-get install git-lfs -y
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -G docker ubuntu

git clone https://github.com/scaleoutsystems/FEDn-client-cifar10-pytorch.git
pushd FEDn-client-cifar10-pytorch

# Download data and create split in 10 IID chunks
sudo apt install python3-pip -y
pip3 install -r requirements.txt
python3 load_datasets.py 

# Make sure you have edited extra-hosts.yaml to provide hostname mappings for combiners
# INDEX in 0...9 to select dataslice for this client.
sudo INDEX=1 docker-compose -f docker-compose.yaml -f extra-hosts.yaml up 
```

## Start a client on an Nvidia enabled host
Make sure that you have appropriate Nvidia drivers installed on the host. 

The follwing shell script will configure and start a client on a Ubuntu 20.04 LTS VM:

```bash
#!/bin/bash

git clone https://github.com/scaleoutsystems/FEDn-client-cifar10-pytorch.git
pushd FEDn-client-cifar10-pytorch

# Download the dataset and split in 10 IID chunks. 
sudo apt install python3-pip -y
pip3 install -r requirements.txt
python3 load_datasets.py 

# Build docker image
sudo docker build -f Dockerfile.gpu . -t cifar-client:latest

# Modify below as needed for setup and data slice to use, set combiner extra host, then start client
docker run --gpus all --add-host=combiner-aws-stockholm:13.53.197.49 -v /home/ubuntu/FEDn-client-cifar10-pytorch/data/10clients/client1:/app/data cifar-client /bin/bash -c "fedn run client -in fedn-network.yaml"
```
