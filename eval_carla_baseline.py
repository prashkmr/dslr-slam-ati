#This is the code where we evaluate our method/model on CARLA on EMD/Chamfer / Result is 210 ans 1.
from __future__ import print_function

import argparse
from torchvision import datasets, transforms

from torch.utils.data import DataLoader, Dataset
import torch
import sys
from torchsummary import summary
import numpy as np
import os
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from models512 import *
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

import sklearn.metrics as metrics
from utils512 import *


parser = argparse.ArgumentParser(description='VAE training of LiDAR')
parser.add_argument('--batch_size',         type=int,   default=64,             help='size of minibatch used during training')
parser.add_argument('--use_selu',           type=int,   default=0,              help='replaces batch_norm + act with SELU')
parser.add_argument('--base_dir',           type=str,   default='runs/test',    help='root of experiment directory')
parser.add_argument('--no_polar',           type=int,   default=0,              help='if True, the representation used is (X,Y,Z), instead of (D, Z), where D=sqrt(X^2+Y^2)')
# parser.add_argument('--lr',                 type=float, default=1e-3,           help='learning rate value')
parser.add_argument('--z_dim',              type=int,   default=160,            help='size of the bottleneck dimension in the VAE, or the latent noise size in GAN')
parser.add_argument('--autoencoder',        type=int,   default=1,              help='if True, we do not enforce the KL regularization cost in the VAE')
parser.add_argument('--atlas_baseline',     type=int,   default=0,              help='If true, Atlas model used. Also determines the number of primitives used in the model')
parser.add_argument('--panos_baseline',     type=int,   default=0,              help='If True, Model by Panos Achlioptas used')
parser.add_argument('--kl_warmup_epochs',   type=int,   default=150,            help='number of epochs before fully enforcing the KL loss')
parser.add_argument('--ae_weight',          type=str,   default='',             help='Location of the weights')
parser.add_argument('--data',               type=str,   default='',             help='Loction of the dataset')

parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                    choices=['pointnet', 'dgcnn'],
                    help='Model to use, [pointnet, dgcnn]')
# parser.add_argument('--use_sgd', type=bool, default=False,
#                     help='Use SGD')
# parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
#                     help='learning rate (default: 0.001, 0.1 if using sgd)')
# parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
#                     help='SGD momentum (default: 0.9)')
parser.add_argument('--no_cuda', type=bool, default=False,
                    help='enables CUDA training')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#                     help='random seed (default: 1)')
# parser.add_argument('--eval', type=bool,  default=False,
#                     help='evaluate the model')
# parser.add_argument('--num_points', type=int, default=1024,
#                     help='num of points to use')
# parser.add_argument('--dropout', type=float, default=0.5,
#                     help='dropout rate')
parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                    help='Dimension of embeddings')
parser.add_argument('--k', type=int, default=20, metavar='N',
                    help='Num of nearest neighbors to use')


parser.add_argument('--debug', action='store_true')


'''
Expect two arguments: 
    1) path_to_model_folder
    2) epoch of model you wish to load
    3) metric to evaluate on 
e.g. python eval.py runs/test_baseline 149 emd
'''

#---------------------------------------------------------------
#Helper Function and classes
class Pairdata(Dataset):
    """
    Dataset of numbers in [a,b] inclusive
    """

    def __init__(self, lidar):
        super(Pairdata, self).__init__()
        
        self.lidar = lidar

    def __len__(self):
        return self.lidar.shape[0]

    def __getitem__(self, index):
        
        return index, self.lidar[index]

#---------------------------------------------------------------
args = parser.parse_args()

class Attention_loader_dytost(Dataset):
    """
    Dataset of numbers in [a,b] inclusive
    """

    def __init__(self, dynamic, static):
        super(Attention_loader_dytost, self).__init__()

        self.dynamic = dynamic
        self.static = static

    def __len__(self):
        return self.dynamic.shape[0]

    def __getitem__(self, index):
        
        return index, self.dynamic[index], self.static[index]





def directed_hausdorff(point_cloud1:torch.Tensor, point_cloud2:torch.Tensor, reduce_mean=True):
    """

    :param point_cloud1: (B, 3, N)
    :param point_cloud2: (B, 3, M)
    :return: directed hausdorff distance, A -> B
    """
    n_pts1 = point_cloud1.shape[2]
    n_pts2 = point_cloud2.shape[2]

    pc1 = point_cloud1.unsqueeze(3)
    pc1 = pc1.repeat((1, 1, 1, n_pts2)) # (B, 3, N, M)
    pc2 = point_cloud2.unsqueeze(2)
    pc2 = pc2.repeat((1, 1, n_pts1, 1)) # (B, 3, N, M)

    l2_dist = torch.sqrt(torch.sum((pc1 - pc2) ** 2, dim=1)) # (B, N, M)

    shortest_dist, _ = torch.min(l2_dist, dim=2)

    hausdorff_dist, _ = torch.max(shortest_dist, dim=1) # (B, )

    if reduce_mean:
        hausdorff_dist = torch.mean(hausdorff_dist)

    return hausdorff_dist






# reproducibility is good
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

nb_samples = 200
# out_dir = os.path.join(sys.argv[1], 'final_samples')
# maybe_create_dir(out_dir)
save_test_dataset = False

fast = True

# fetch metric
# # if 'emd' in sys.argv[3]: 
# #     loss = EMD
# # elif 'chamfer' in sys.argv[3]:
# #     loss = get_chamfer_dist
# else:
#     raise ValueError("{} is not a valid metric for point cloud eval. Either \'emd\' or \'chamfer\'"\
#             .format(sys.argv[2]))

# loss1 = EMD
# loss_fn1 = loss1()
loss = get_chamfer_dist
# size = 10 if 'emd' in sys.argv[3] else 5
size = 8

npydata = [3]
orig = []
pred = []

out = np.ndarray(shape=(3072,3,12,512))

totalcd = 0
totalhd = 0

with torch.no_grad():
  
  for i in npydata:
    ii = 0 
    # 1) load trained model
    # model = load_model_from_file(sys.argv[1], epoch=int(sys.argv[2]), model='gen')[0]
    # model = VAE(args).cuda()
    # model = VAE(args).cuda()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    model = VAE(args).to(device)
    # model = nn.DataParallel(model)
    # summary(model, (3,2048))
    # exit(1)

    model = model.cuda()
    # network=torch.load(args.ae_weight)
    # print(network.keys())
    # model.load_state_dict(network['state_dict'])
    # model.load_state_dict(network['gen_dict'])
    weight = torch.load(args.ae_weight)
    model.load_state_dict(weight['gen_dict'])
    # opt.load_state_dict(weight['optimizer'])
    
    model.eval() 

    # if 'panos' in sys.argv[1] or 'atlas' in sys.argv[1] : model.args.no_polar = 1 
    
    # 2) load data
    # print('test set reconstruction')
    # dataset = np.load('../lidar_generation/kitti_data/lidar_test.npz')
    # if fast: dataset = dataset[:100]
    # dataset_test = preprocess(dataset).astype('float32')

    # dataset preprocessing
    # print('loading Testing data')
    # dataset_train = np.load('../lidar_generation/kitti_data/lidar.npz')
    lidar_static    = np.load(args.data + "s{}.npy".format(str(i)))[:,:,2:14,::2].astype('float32')
    print(lidar_static.shape)
    lidar_dynamic   = np.load(args.data + "d{}.npy".format(str(i)))[:,:,2:14,::2].astype('float32')
    # lidar = from_polar_np(lidar).transpose(0, 2, 3, 1)
    # lidar_static = from_polar_np(lidar_static)
    # lidar_dynamic = from_polar_np(lidar_dynamic)
    # # print(lidar.shape)
    # lidar_static = lidar_static.reshape(-1, 3, 5120)
    # lidar_dynamic = lidar_dynamic.reshape(-1, 3, 5120)
    # # print(lidar.shape)
    # lidar_mask    = np.load("/content/drive/Shareddrives/Classification/lidar/dslr/data_custom/masks_d/testd{}_mask.npy".format(str(i)))[:,:2,:,:].astype('float32')
    # test_static    = preprocess(test_static).astype('float32')

    # mask   = np.load(os.path.join(args.data, 'mask/k4.npy'))[:,:2,:,::2].astype('float32')
    # mask    = (mask>0.5).astype('float32')
    # lidarSt = np.load("/content/drive/Shareddrives/Classification/lidar/dslr/data_custom/s/tests{}.npy".format(str(i)))[:1024,:,5:45,:].astype('float32')
    # lidarSt_mask    = np.load("/content/drive/Shareddrives/Classification/lidar/dslr/data_custom/masks_s/tests{}_mask.npy".format(str(i)))[:,:2,:,:].astype('float32')
    # test_dynamic   = preprocess(test_dynamic).astype('float32')

    test_loader    = Attention_loader_dytost(lidar_dynamic, lidar_static)
    
    loader = (torch.utils.data.DataLoader(test_loader, batch_size=args.batch_size,
                        shuffle=False, num_workers=1, drop_last=True)) #False))

    loss_fn = loss()
    # process_input = (lambda x : x) if model.args.no_polar else to_polar
    process_input = from_polar if args.no_polar else lambda x : x
    
    # noisy reconstruction
    for noise in [0]:#0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.][::(2 if fast else 1)]: 
        losses, losses1 = [], []
        # losses1 = []
        for batch in loader:
            lidar_dynamic = batch[1].cuda()
            lidar_static = batch[2].cuda()
            # print(lidar.shape)
            # lidar = lidar.permute(0, 2, 1)
            # print(lidar.shape)
            # mask  = batch[2].cuda()
            # lidarSt=batch[2].cuda()
            # lidar_mask = batch[3].cuda()
            # lidarSt_mask = batch[4].cuda()
            # batch = batch.cuda() 
            # batch_xyz = from_polar(batch)
            # stPair = from_polar(stPair)
            # dyPair = from_polar(dyPair)
            # noise_tensor = torch.zeros_like(stPair).normal_(0, noise)

            # means = dyPair.transpose(1,0).reshape((3, -1)).mean(dim=-1)
            # stds  = dyPair.transpose(1,0).reshape((3, -1)).std(dim=-1)
            # means, stds = [x.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) for x in [means, stds]]

            # # normalize data
            # norm_batch_xyz = (dyPair - means) / (stds + 1e-9)
            # # add the noise
            # input = norm_batch_xyz + noise_tensor

            # # unnormalize
            # input = input * (stds + 1e-9) + means

            # print(lidar.shape)
            recon, _,_  = model( lidar_dynamic )
            recon = recon
            # print(recon.shape)
            # recon = recon.permute(0, 2, 1)
            # recon = recon.cpu().numpy()
            # recon = recon.reshape(-1, 40, 256, 3)
            # print(recon.shape)

            # assert False
            # print(recon.shape)
            # orig.append(lidar_static.cpu().numpy())
            # pred.append(recon.cpu().numpy())
            # print(recon.reshape(-1,3,40, 256).shape)
            
            # recon_xyz = from_polar(recon)
            # print((recon.shape))
            # print((lidar.shape))
            # if args.panos_baseline:
            #   losses += [loss_fn(recon, lidarSt)]
            # else:  
            # print(lidar_static.shape, lidar_dynamic.shape)
            losses += [loss_fn(recon, lidar_static)]
            losses1 += [directed_hausdorff(from_polar(recon).reshape(-1,3,12*512), from_polar(lidar_static).reshape(-1,3,12*512))]
            # print(recon.shape)  [8,3,10240]
            out[ii*args.batch_size:(ii+1)*args.batch_size]   =    from_polar(recon).detach().cpu().numpy().reshape(-1, 3, 12, 512)
            ii+=1
        np.save( str(i) + '.npy', out)    
        print('Saved ', i)

        losses = torch.stack(losses).mean().item()
        losses1 = torch.stack(losses1).mean().item()
        totalcd += losses
        totalhd += losses1
        # losses1 += [loss_fn1.apply(recon, lidar)]
        # losses1 = torch.stack(losses1).mean().item()
        print('Chamfer Loss for {}: {:.4f}'.format(i, losses))
        print('Hauss   Loss for {}: {:.4f}'.format(i, losses1))
        # print('EMD Loss for {}: {:.4f}'.format(i, losses1))

        del recon, losses
print(totalcd/len(npydata))
print(totalhd/len(npydata))