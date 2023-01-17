from torchvision import datasets, transforms
import torch.utils.data
import torch
import sys
import argparse
import matplotlib.pyplot as plt
from utils512 import *
import numpy as np
import os

from tqdm import trange



parser = argparse.ArgumentParser(description='VAE training of LiDAR')
parser.add_argument('--data',       type=str,   default='',              help='Location of the orignal LiDAR')
parser.add_argument('--folder',     type=str,   default='',              help='Location of the orignal LiDAR')
parser.add_argument('--debug', action='store_true')
# args = parser.parse_args(args=['atlas_baseline=0, autoencoder=1,panos_baseline=0'])





# Encoder must be trained with all types of frames,dynmaic, static all

args = parser.parse_args()

original = np.load(args.data)[:1024,:,:,:]
print(original.shape)

if not os.path.exists('samplesVid/'+str(args.folder)):
	os.makedirs('samplesVid/'+str(args.folder))


i=0
for frame_num in trange( np.minimum(200, original.shape[0]) ):
	if(i<1024):
		i+=1
		frame = from_polar(torch.Tensor(original[frame_num:frame_num+1]).cuda())
		frame = frame.detach().cpu().numpy()
		# frame=from_polar(torch.Tensor(original[frame_num:frame_num+1,:,:,:]).cuda()).detach().cpu()
		plt.figure()
		plt.xlim([-0.15, 0.20])
		plt.ylim([-0.10, 0.20])
		plt.scatter(frame[:, 0], frame[:, 1], s=0.7, color='k')
		plt.savefig('samplesVid/'+str(args.folder)+'/'+str(frame_num)+'.jpg') 
	else:
		break
                                                              
