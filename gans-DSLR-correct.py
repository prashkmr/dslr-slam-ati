import argparse
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from pydoc import locate
import tensorboardX
# from torch.utils.tensorboard import SummaryWriter
import shutil
from torchsummary import summary
import torchsummary
from utils512 import * 
from models512 import * 
from SLAM_ERROR import *






parser = argparse.ArgumentParser(description='GAN training of LiDAR')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--loss', type=int, default=1, help='0 == LSGAN, 1 == RaLSGAN')
parser.add_argument('--use_selu', type=int, default=0, help='replaces batch_norm + act with SELU')
parser.add_argument('--use_spectral_norm', type=int, default=1)
parser.add_argument('--use_round_conv', type=int, default=0)
parser.add_argument('--base_dir', type=str, default='runs/test/')
parser.add_argument('--dis_iters', type=int, default=1, help='disc iterations per 1 gen iter')
parser.add_argument('--no_polar', type=int, default=0)
parser.add_argument('--optim',  type=str, default='rmsprop')
parser.add_argument('--pose_dim',           type=int,   default=160,            help='size of the pose vector')
parser.add_argument('--data',  type=str, required= True, default='', help='Location of the dataset')
parser.add_argument('--log',  type=str, required= True, default='', help='Name of the log folder')
parser.add_argument('--atlas_baseline',     type=int,   default=0,              help='If true, Atlas model used. Also determines the number of primitives used in the model')
parser.add_argument('--panos_baseline',     type=int,   default=0,              help='If True, Model by Panos Achlioptas used')
parser.add_argument('--kl_warmup_epochs',   type=int,   default=150,            help='number of epochs before fully enforcing the KL loss')
parser.add_argument('--autoencoder',        type=int,   default=1,              help='if True, we do not enforce the KL regularization cost in the VAE')
parser.add_argument('--z_dim',              type=int,   default=160,            help='size of the bottleneck dimension in the VAE, or the latent noise size in GAN')
parser.add_argument('--reload',             type=str,   default='',             help='')
parser.add_argument('--warmup',             type=int,   default=30,             help='warmup epochs for LIDAR translation')
#parser.add_argument('--panos_baseline',     type=int,   default=0,              help='If True, Model by Panos Achlioptas used')
parser.add_argument('--gen_lr', type=float, default=1e-4)
parser.add_argument('--dis_lr', type=float, default=1e-4)
args = parser.parse_args()
DATA = args.data
RUN_SAVE_PATH = args.log
maybe_create_dir(args.base_dir+RUN_SAVE_PATH)
print_and_save_args(args, args.base_dir+RUN_SAVE_PATH)

# reproducibility is good
np.random.seed(0)
torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
shutil.copy(os.path.basename(__file__), args.base_dir + RUN_SAVE_PATH)


static  = []
dynamic = []
stMask  = []
dyMask  = []
def loadNP(index):
    global static, dynamic, stMask, dyMask
    print('Loading npys...')
    for i in trange(index):
        static.append( np.load( DATA  + "lidar/s{}.npy".format(i) ) [:,:,:,::4].astype('float32'))
        dynamic.append( np.load( DATA + "lidar/d{}.npy".format(i)  )[:,:,:,::4].astype('float32'))
        # stMask.append( np.load( DATA  + "mask/s{}.npy".format(i)  )[:,0:2,:,:].astype('float32'))
        # dyMask.append( np.load( DATA + "mask/d{}.npy".format(i)  )[:,0:2,:,:].astype('float32'))

loadNP(3)


# Logging
maybe_create_dir(os.path.join(args.base_dir, 'samples'))
writer = tensorboardX.SummaryWriter(log_dir=os.path.join(args.base_dir+RUN_SAVE_PATH, 'TB'))
writes = 0

#if args.atlas_baseline or args.panos_baseline: 
""" ed on 12 Gb GPU for z_dim in [128, 256, 512] """ 
# bs = [4, 8 if args.atlas_baseline else 6][min(1, 511 // args.z_dim)]
# factor = args.batch_size // 4
# args.batch_size = 64
# is_baseline = True
# args.no_polar = 0
#print('using batch size of %d, ran %d times' % (bs, factor))
# else:
#     factor, is_baseline = 1, False

# construct model and ship to GPU
dis = scene_discriminator(args.pose_dim).cuda()
gen = VAE(args).cuda()

summary(gen, (2,12,512))
# exit(0)

print(gen)
print(dis)

gen.apply(weights_init)
dis.apply(weights_init)    

triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y))
#  output = triplet_loss(anchor, positive, negative)





class Attention_loader(Dataset):
    """
    Dataset of numbers in [a,b] inclusive
    """
    def __init__(self,static1, static2, dynamic1):
        super(Attention_loader, self).__init__()

        self.static1 = static1
        self.static2 = static2
        self.dynamic1= dynamic1
 
    def __len__(self):
        return min(self.static1.shape[0], self.static2.shape[0])

    def __getitem__(self, index):
        
        return index, self.static1[index], self.static2[index], self.dynamic1[index]


def load(npyList):
    retList=[]
    for i in npyList:
        print(i)
        s1 = static[i] 
        s2 = static[(i+1)%len(npyList)]
        d1 = dynamic[i]
     

        data_train = Attention_loader(s1, s2, d1)

        train_loader  = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size,
                        shuffle=True, num_workers=4, drop_last=True)
        #del data_train
        retList.append(train_loader)
    print(retList)
    return retList

npyList = [i for i in range(3)]
npyList1 =load(npyList)





if args.optim.lower() == 'adam': 
    gen_optim = optim.Adam(gen.parameters(), lr=args.gen_lr, betas=(0.5, 0.999), weight_decay=0)
    dis_optim = optim.Adam(dis.parameters(), lr=args.dis_lr, betas=(0.5, 0.999), weight_decay=0)
elif args.optim.lower() == 'rmsprop': 
    gen_optim = optim.RMSprop(gen.parameters(), lr=args.gen_lr)
    dis_optim = optim.RMSprop(dis.parameters(), lr=args.dis_lr)


if args.reload !='':
    reload = torch.load(args.reload)
    gen.load_state_dict(reload['gen_dict'])
    dis.load_state_dict(reload['dis_dict'])
    gen_optim.load_state_dict(reload['gen_optim'])
    dis_optim.load_state_dict(reload['dis_optim'])
    print("reloaded")


loss_fn = lambda a, b : (a - b).abs().sum(-1).sum(-1).sum(-1)
# gan training
# ------------------------------------------------------------------------------------------------
for epoch in trange(0,1501):
    print('epochs: ',epoch)
    for i in npyList1:
        data_iter = iter(i)
        iters = 0
        real_d, fake_d, fake_g, losses_g, losses_d, delta_d = [[] for _ in range(6)]
        process_input = from_polar if args.no_polar else lambda x : x

        while iters < len(i):
            j = 0
            # if iters > 10 : break
            # print(iters)
        
            """ Update Discriminator Network """
            for p in dis.parameters():
                p.requires_grad = True

            while j < args.dis_iters and iters < len(i):
                j += 1; iters += 1

                inputs = data_iter.next()
                
                # width, height = 12, 12
                # axis_x = np.arange(0, width)
                # axis_y = np.arange(0, height)
                # grid_axes = np.array(np.meshgrid(axis_x, axis_y))
                # grid_axes = np.transpose(grid_axes, (1, 2, 0))
                # from scipy.spatial import Delaunay
                # tri = Delaunay(grid_axes.reshape([-1, 2]))
                # faces = tri.simplices.copy()
                # F = DiagramlayerToplevel().init_filtration(faces)
                # diagramlayerToplevel = DiagramlayerToplevel.apply
            
                # train with real data
                recon0, kl_cost0, z_real0 = gen(process_input(inputs[1][:,:,2:14,:].cuda()))
                recon1, kl_cost1, z_real1 = gen(process_input(inputs[2][:,:,2:14,:].cuda()))                

                real_out = dis(z_real0,z_real1)
            
                real_d += [real_out.mean().detach()]
                
                # train with fake data 
                # noise = torch.cuda.FloatTensor(args.batch_size, 100).normal_()
                recon2, kl_cost2, z_fake2 = gen(process_input(inputs[3][:,:,2:14,:].cuda()))

                fake_out = dis(z_fake2 , z_real1)
                # print(z_fake2.shape, z_real1.shape)   # torch.Size([64, 160]) torch.Size([64, 160])
                # print(fake_out.shape)                 # torch.Size([64, 1]) 
                # exit(0)

                fake_d += [fake_out.mean().detach()]
                
                if args.loss == 0 : 
                    dis_loss = (((real_out - fake_out.mean() - 1) ** 2).mean() + \
                                ((fake_out - real_out.mean() + 1) ** 2).mean()) / 2
                else:
                    dis_loss = (torch.mean((real_out - 1) ** 2) + torch.mean((fake_out - 0) ** 2)) / 2

                
            
                #---------------------------------
                #Add Contrastive loss to total discrimiantor Loss
                loss_contrastive = triplet_loss(z_real0, z_real1, z_fake2)
                dis_loss += loss_contrastive
                #---------------------------------
                
                losses_d += [dis_loss.mean().detach()]
                #delta_d  += [(real_out.mean() - fake.mean()).detach()]
                
                # top_loss_out0 = top_batch_cost(recon0.detach().cpu(), diagramlayerToplevel, F)
                # top_loss_hidden0 = top_batch_cost(z_real0.detach().cpu(), diagramlayerToplevel, F)
                
                # top_loss_out1 = top_batch_cost(recon1.detach().cpu(), diagramlayerToplevel, F)
                # top_loss_hidden1 = top_batch_cost(z_real1.detach().cpu(), diagramlayerToplevel, F)
                
                # top_loss_out2 = top_batch_cost(recon2.detach().cpu(), diagramlayerToplevel, F)
                # top_loss_hidden2 = top_batch_cost(z_real2.detach().cpu(), diagramlayerToplevel, F)
                
                # dis_loss = dis_loss+top_loss_out0+top_loss_hidden0+top_loss_out1+top_loss_hidden1+top_loss_out2+top_loss_hidden2
            
                dis_optim.zero_grad()
                dis_loss.backward()
                dis_optim.step()

            """ Update Generator network """
            for p in dis.parameters():
                p.requires_grad = False

            # noise = torch.cuda.FloatTensor(args.batch_size, 100).normal_()
            recon2, kl_cost2, z_fake2 = gen(process_input(inputs[3][:,:,2:14,:].cuda()))
            recon1, kl_cost1, z_real1 = gen(process_input(inputs[2][:,:,2:14,:].cuda()))
            recon0, kl_cost0, z_real0 = gen(process_input(inputs[1][:,:,2:14,:].cuda()))
                
            fake_out = dis(z_fake2, z_real1)

            fake_g += [fake_out.mean().detach()]        
            
            if args.loss == 0: 
                iters += 1
                inputs = inputs
            
                # raise SystemError
                real_out = dis(z_real0, z_real1)
                gen_loss = (((real_out - fake_out.mean() + 1) ** 2).mean() + \
                            ((fake_out - real_out.mean() - 1) ** 2).mean()) / 2
            else:
                gen_loss = torch.mean((fake_out - 1.) ** 2)

            #My original code
            # real_out = dis(z_real0, z_real1)

            # gen_loss = (((real_out - fake_out.mean() + 1) ** 2).mean() + \
            #                 ((fake_out - real_out.mean() - 1) ** 2).mean()) / 2
            
                

            recloss = loss_fn(recon2[:,:,:12,:], inputs[1][:,:,2:14,:].cuda()).mean(dim=0)
            gen_loss += recloss

            if (epoch >= args.warmup and epoch % 5 == 0):
                #  print('Slam')
                gt        = from_polar(inputs[1].cuda()).reshape(-1,3,8192).permute(0,2,1)
                recon_new = from_polar(recon2[:,:,:12,:]).reshape(-1,3,6144).permute(0,2,1)
                groundtruth_list=  [gt[i] for i in range(inputs[1].shape[0])]
                recon_list      =  [recon_new[i] for i in range(recon2.shape[0])] 
                slam_err        = Slam_error(groundtruth_list, recon_list)
                print('SLAM Error :', slam_err)
                gen_loss        += slam_err
                

            losses_g += [gen_loss.detach()]
            
            
        
            gen_optim.zero_grad()
            gen_loss.backward()
            gen_optim.step()

        print('recloss:',"       ", recloss.item())
        
        print_and_log_scalar(writer, 'real_out', real_d, writes)
        print_and_log_scalar(writer, 'fake_out', fake_d, writes)
        print_and_log_scalar(writer, 'fake_out_g', fake_g, writes)
        print_and_log_scalar(writer, 'delta_d', delta_d, writes)
        print_and_log_scalar(writer, 'losses_gen', losses_g, writes)
        print_and_log_scalar(writer, 'losses_dis', losses_d, writes)
        writes += 1

        # save some training reconstructions
 

        if (epoch) % 5 == 0 :

            
            state = {
            'epoch': epoch + 1, 
            'gen_dict': gen.state_dict(),
            'dis_dict': dis.state_dict(),
            'gen_optim': gen_optim.state_dict(),
            'dis_optim': dis_optim.state_dict()
            }
            torch.save(state, os.path.join(args.base_dir + RUN_SAVE_PATH, 'gen_{}.pth'.format(epoch)))
            print('saved models')
