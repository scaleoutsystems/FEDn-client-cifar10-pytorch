# FEDn-client-cifar10-pytorch

To attach to a FEDn network, first edit 'fedn-network.yaml' to set the endpoint for the controller. Then edit 'extra-hosts.yaml' to provice dns resolution for each combiner in the FEDn network.

## Configure and start a client usign cpu device

The following will help you configure a client on a blank Ubuntu 20.04 LTS VM:    

<script src="https://gist.github.com/ahellander/9046dcd20e1721c7babca6fd8e646733.js"></script>

## Start a client on an Nvidia enabled host
Make sure to have Nvidia drivers installed on the host. To start a client using Nvidia GPU:

<script src="https://gist.github.com/ahellander/41fe30e2938a8e63b08423b86c602245.js"></script>
