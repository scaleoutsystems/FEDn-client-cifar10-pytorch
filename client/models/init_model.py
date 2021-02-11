#from fedn.utils.pytorchmodel import PytorchModelHelper
from pytorch_model import create_seed_model
import torch
import numpy as np

if __name__ == '__main__':

	# Create a seed model and push to Minio
	model, _, _ = create_seed_model('VGG16')
	outfile_name = "../../seed/VGG16_torch.npz"
	np.savez_compressed(outfile_name, **model.state_dict())