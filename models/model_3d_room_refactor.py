import os

import numpy as np
import math
import random
import time

import torch
import torch.nn.functional as F

from torch import Tensor
from torch import nn
from torch.autograd import Variable
from torch.utils.data.sampler import WeightedRandomSampler
from models import database as db
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib
import matplotlib.pylab as plt

import pickle5 as pickle 

from models.model_utils import get_embedder
from dataloader.fast_sampler import CustomWeightedRandomSampler, BatchWeightedRandomSampler, DynamicWeightedRandomSampler
from utils.loss_utils.loss_recorder import LossRecoder
from dataprocessing.gen_2d_map_and_samples_pytorch import check_collision as check_collision_SE2, get_dist_fn as get_dist_fn_SE2
from dataprocessing.speed_sampling_gpu_refactor import check_collision as check_collision_SE3, get_dist_fn, preprepare_arm
import igl
from utils.util import read_hdf5
import shapely

import glob
from copy import deepcopy
from tqdm import tqdm
from dataloader.fast_dataloader import FastDataLoader
from .blocks import C_encoder_block, C_encoder_block_next, UnetBaiscBlock
    
class NN(nn.Module):
    
    def __init__(self, device, dim, args):#10
        super().__init__()
        self.dim = dim

        self.vmax = args.vmax
        h_size = args.h_size # 256 #512,256

        self.args = args
        self.full_coords_enc = args.full_coords_enc

        self.device = device

        #decoder

        self.scale = 10

        self.act = nn.ELU()

        self.nl1 = self.args.enc_d
        self.nl2 = self.args.dec_d
        
        p_dim = dim

        if self.args.pos_enc:
            self.p_embedder, p_dim = get_embedder(1, dim)

        basic_blocks = {
            'resnext': C_encoder_block_next,
            'resnet': C_encoder_block,
            'unet': UnetBaiscBlock
        }
        basic_block = basic_blocks[self.args.basic_block]

        if self.args.use_unet:
            self.enc0 = nn.Sequential(
                nn.Linear(p_dim, h_size),
                self.act
            )
            self.encoder_blocks = nn.ModuleList([
                UnetBaiscBlock(h_size, 32, 32),
                UnetBaiscBlock(32, 64, 64),
                UnetBaiscBlock(64, 128, 128),
                UnetBaiscBlock(128, 256, 256),
            ])

        else:
            self.encoder = nn.Sequential(
                nn.Linear(p_dim, h_size),
                self.act,
                *[basic_block(h_size, h_size, h_size) for _ in range(self.nl1-1)],
                nn.Linear(h_size, h_size),
            )

        if self.args.use_unet:
            self.dec0 = nn.Sequential(
                nn.Linear(2*h_size, 2*h_size),
                self.act,
            )
            self.decoder_blocks = nn.ModuleList([
                UnetBaiscBlock(1024, 256, 256),
                UnetBaiscBlock(512, 128, 128),
                UnetBaiscBlock(256, 64, 64),
                UnetBaiscBlock(128, 32, 32)
            ])
            self.dec1 = nn.Linear(32, 1)
        else:
            if self.args.symm_op_type == 'f':
                gen0_in_size = h_size
            else:
                gen0_in_size = 2*h_size
            self.gen0 = nn.Sequential(
                nn.Linear(gen0_in_size, 2*h_size),
                self.act,
            )
            self.generator = nn.Sequential(
                *[basic_block(2*h_size, 2*h_size, 2*h_size) for _ in range(self.nl2-1)],
                nn.Linear(2*h_size, h_size),
                self.act,
                nn.Linear(h_size, 1),
            )

    def init_weights(self, m):
        
        if type(m) == nn.Linear:
            stdv = (1. / math.sqrt(m.weight.size(1))/1.)*2
            #stdv = np.sqrt(6 / 64.) / self.T
            m.weight.data.uniform_(-stdv, stdv)
            m.bias.data.uniform_(-stdv, stdv)

    def symm_op(self, x):
        size = x.shape[0] // 2
        x0 = x[:size,...]
        x1 = x[size:,...]
        
        x_0 = torch.max(x0,x1)
        x_1 = torch.min(x0,x1)

        y = torch.cat((x_0, x_1), dim=1)
        return y
    
    def out(self, coords):
        
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        size = coords.shape[0]

        x0 = coords[:,:self.dim]
        x1 = coords[:,self.dim:]
        
        x = torch.vstack((x0,x1))
        
        if self.args.pos_enc:
            x = self.p_embedder(x)

        if self.args.use_unet:

            x = self.enc0(x)
            skip_connections = []

            for enc_block in self.encoder_blocks:
                x = enc_block(x)
                skip_connections.append(self.symm_op(x))
        else:
            x = self.encoder(x)
        
        x0 = x[:size,...]
        x1 = x[size:,...]
        
        if self.args.symm_op_type == 'min_max':
            x_0 = torch.max(x0,x1)
            x_1 = torch.min(x0,x1)

        if self.args.symm_op_type == 'mean':
            x = (torch.cat((x0, x1), 1) + torch.cat((x1, x0), 1)) / 2
        elif self.args.symm_op_type == 'f':
            # f = lambda x, y: (x + y) ** 2 / (4 * x * y + 1e-6)
            f = lambda x, y: (x + y) / 2
            x = f(x0, x1)
        else:
            x = torch.cat((x_0, x_1), 1)

        if self.args.use_unet:
            x = self.dec0(x)
            for dec_block in self.decoder_blocks:
                skip_x = skip_connections.pop()
                x = torch.cat((x, skip_x), dim=-1)
                x = dec_block(x)
            x = self.dec1(x)
        
        else:
            x = self.gen0(x)

            x = self.generator(x)

        x = self.vmax * torch.sigmoid(0.1*x)
        
        return x, coords

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output, coords = self.out(coords)
        
        return output, coords

class Model():
    def __init__(self, ModelPath, DataPath, dim, pos, logger, args, device='cpu'):

        self.Params = {}
        self.Params['ModelPath'] = ModelPath
        self.Params['DataPath'] = DataPath
        self.dim = dim
        self.pos = pos
        self.logger = logger
        self.args = args
        
        # Pass the JSON information
        self.Params['Device'] = device
        self.Params['Network'] = {}

    def gradient(self, y, x, create_graph=True):                                                               
                                                                                  
        grad_y = torch.ones_like(y)                                                                 
        grad_x = torch.autograd.grad(y, x, grad_y, only_inputs=True, retain_graph=True, create_graph=create_graph)[0]
        
        return grad_x
    
    def get_time_in_train(self, Xp):
        
        T = self.TravelTimes(Xp)
        return T

    def get_gradient_in_train(self, Xp):
        Xp = Xp.to(torch.device(self.Params['Device']))
       
        # Xp.requires_grad_(True)

        tau, Xp = self.network.out(Xp)
        dtau = self.gradient(tau, Xp)
        
        with torch.no_grad():
            # x = start; y = goal
            D = Xp[:,self.dim:]-Xp[:,:self.dim]
            T0 = torch.sqrt(torch.einsum('ij,ij->i', D, D)).unsqueeze(1) # ||x - y||
            T3 = tau**2

            V0 = D # y - x
            V1 = dtau[:,self.dim:] # dtau_goal
            
            Y1 = 1/(T0*tau)*V0           # (y - x) / (||x - y|| * tau)
            Y2 = T0/T3*V1                # (dtau_goal * ||x - y||) / tau**2


            Ypred1 = -(Y1-Y2)            # -dT(x, y) / dy
            Spred1 = torch.norm(Ypred1, dim=1, keepdim=True)  # 1/ S(y)
            Ypred1 = 1/Spred1**2*Ypred1  # -S^2(y) * dT(x, y) / dy

            V0=-D  # x - y
            V1=dtau[:,:self.dim] # dtau_start
            
            Y1 = 1/(T0*tau)*V0           # (x - y) / (||x - y|| * tau)
            Y2 = T0/T3*V1                # (dtau_start * ||x - y||) / tau**2

            Ypred0 = -(Y1-Y2)            # -dT(x, y) / dx
            Spred0 = torch.norm(Ypred0, dim=1, keepdim=True)  # 1/ S(x)
            Ypred0 = 1/Spred0**2*Ypred0  # -S^2(x) * dT(x, y) / dx
        
        return torch.cat((Ypred0, Ypred1),dim=1)
    
    def get_path_loss(self, Xp, dtau, tau, T, epoch):
        
        grad = self.grad_from_dtau(Xp, dtau, tau)
        
        if self.args.one_step:
            step = 1
        else:
            step = np.random.randint(1, 5, 1)[0]
        # TODO: do not clone
        coords = Xp.clone().detach().requires_grad_(True)
        # coords = Xp

        coords = coords + 0.03 / self.args.vmax * grad
        if step > 1:
            for i in range(step - 1):
                grad = self.get_gradient_in_train(coords)
                coords = coords + 0.03 / self.args.vmax * grad
        
        coords = coords.detach()

        start_prob = np.random.rand()
        
        c_size = coords.shape[0]
        no_coll_mask = self.check_collision(torch.cat((coords[:, :self.dim], coords[:, self.dim:]), dim=0))
        start_no_coll_mask = no_coll_mask[:c_size]
        goal_no_coll_mask = no_coll_mask[c_size:]
        # no_coll_mask = start_no_coll_mask & goal_no_coll_mask
        if start_prob > 0.5:
            no_coll_mask = start_no_coll_mask
        else:
            no_coll_mask = goal_no_coll_mask

        both_sg_no_coll_mask = start_no_coll_mask & goal_no_coll_mask

        coords_new = coords        
        coords_old = Xp.clone().detach()

        coords_wp_all = coords_new[no_coll_mask]
        coord_waypoint = coords_wp_all[:, :self.dim] if start_prob > 0.5 else coords_wp_all[:, self.dim:]
        
        return_loss = []
        
        with torch.no_grad():
            coords = coords_old[no_coll_mask]
            coords_sw = torch.cat([coords[:, :self.dim], coord_waypoint], dim=1)
            coords_wg = torch.cat([coord_waypoint, coords[:, self.dim:]], dim=1)
            coords_coll_wp_all = coords_new[~no_coll_mask]
            coords_coll_waypoint = coords_coll_wp_all[:, :self.dim] if start_prob > 0.5 else coords_coll_wp_all[:, self.dim:]
            coords_coll = coords_old[~no_coll_mask]
            coords_coll_sw = torch.cat([coords_coll[:, :self.dim], coords_coll_waypoint], dim=1)
            coords_coll_wg = torch.cat([coords_coll_waypoint, coords_coll[:, self.dim:]], dim=1)
            c_size = coords.shape[0]
            c_coll_size = coords_coll.shape[0]
            coords_w = torch.cat((coords_sw, coords_wg, coords_coll_sw, coords_coll_wg), dim=0)
            T_w_all = self.get_time_in_train(coords_w)
            T_w = T_w_all[:2*c_size]
            T_sw = T_w[:c_size]
            T_wg = T_w[c_size:]
            T_coll_w = T_w_all[2*c_size:]
            T_coll_sw = T_coll_w[:c_coll_size]
            T_coll_wg = T_coll_w[c_coll_size:]
            
            T_sum = T_sw + T_wg
            # nan will occur when start == goal, tau will be nan, gradient will be nan, (y - x) / (||x - y|| * tau) will be nan
            nan_mask = torch.isnan(T_sum)
            if nan_mask.any():
                print(f'nan occurs in path loss, totally {len(torch.nonzero(nan_mask))}.')
                T_sum = torch.where(nan_mask, torch.zeros_like(T_sum), T_sum)
            
            
            T_new = self.get_time_in_train(coords_new[both_sg_no_coll_mask])
        
        def ineq_loss(alpha, x, y, total=False):
            """
            loss for x < y or x <= y
            """
            penalty = torch.clamp(x - y, min=0)
            if not total:
                penalty = penalty[penalty > 0]
            size = penalty.shape[0] if penalty.shape[0] > 0 else 1
            loss = alpha * torch.sum(penalty) / size
            return loss
        
        total_loss_sum = False
        # T > T_new
        T_sg = T[both_sg_no_coll_mask]
        loss_path_1 = ineq_loss(self.beta1, T_new, T_sg, total=total_loss_sum)
        return_loss.append(loss_path_1)
        
        criterion = nn.L1Loss()
        T_sg = T[no_coll_mask]
        # T == T_sw + T_wg, L1 loss
        # loss_path = self.beta * criterion(T_sg, T_sum)
        # T <= T_sw + T_wg
        loss_path = ineq_loss(self.beta, T_sg, T_sum, total=total_loss_sum)
        return_loss.append(loss_path)
        
        # T > Ts
        loss_path_T_Ts = ineq_loss(self.beta2, T_sw, T_sg, total=total_loss_sum)
        return_loss.append(loss_path_T_Ts)
        
        # T > Tg
        loss_path_T_Tg = ineq_loss(self.beta3, T_wg, T_sg, total=total_loss_sum)
        return_loss.append(loss_path_T_Tg)
        
        # T < T_coll_s
        T_sg = T[~no_coll_mask]
        loss_path_T_T_coll_s = ineq_loss(self.beta5, T_sg, T_coll_sw, total=total_loss_sum)
        return_loss.append(loss_path_T_T_coll_s)
        
        # T < T_coll_g
        loss_path_T_T_coll_g = ineq_loss(self.beta6, T_sg, T_coll_wg, total=total_loss_sum)
        return_loss.append(loss_path_T_T_coll_g)

        loss_metrics = []
        loss_metrics.append(no_coll_mask.sum())
        loss_metrics.append((~no_coll_mask).sum())
        loss_metrics.append(torch.sum(T_new <= T[both_sg_no_coll_mask]))
        loss_metrics.append(torch.sum(T_new > T[both_sg_no_coll_mask]))
        loss_metrics.append(torch.sum(T_sw > T[no_coll_mask]))
        loss_metrics.append(torch.sum(T_wg > T[no_coll_mask]))
        loss_metrics.append(torch.sum(T_sum < T[no_coll_mask]))
        return return_loss, loss_metrics
            
    def get_speed_loss(self, tau, Xp, Yobs):

        dtau = self.gradient(tau, Xp)
        
        D = Xp[:,self.dim:]-Xp[:,:self.dim] # y - x
        
        T0 = torch.einsum('ij,ij->i', D, D) # ||x - y||^2
        
        
        DT0    = dtau[:,:self.dim] # dtau_start
        DT1    = dtau[:,self.dim:] # dtau_goal

        T01    = T0*torch.einsum('ij,ij->i', DT0, DT0)         # ||x - y||^2 * ||dtau_start||^2
        T02    = -2*tau[:,0]*torch.einsum('ij,ij->i', DT0, D)  # -2 * tau * <y - x, dtau_start>

        T11    = T0*torch.einsum('ij,ij->i', DT1, DT1)         # ||x - y||^2 * ||dtau_goal||^2
        T12    = 2*tau[:,0]*torch.einsum('ij,ij->i', DT1, D)   # 2 * tau * <y - x, dtau_goal>
        
    
        T3    = tau[:,0]**2 # tau^2
        
        S0 = (T01-T02+T3)   # ||x - y||^2 * ||dtau_start||^2 - 2 * tau * <x - y, dtau_start> + tau^2
        S1 = (T11-T12+T3)   # ||x - y||^2 * ||dtau_goal||^2 - 2 * tau * <y - x, dtau_goal> + tau^2
        
        Ypred0 = 1/(torch.sqrt(S0)/T3)
        Ypred1 = 1/(torch.sqrt(S1)/T3)

        if self.args.speed_loss == 'sq':
            sq_Yobs0=torch.sqrt(Yobs[:,0]) # S*_x
            sq_Yobs1=torch.sqrt(Yobs[:,1]) # S*_y
            sq_Ypred0 = torch.sqrt(Ypred0)
            sq_Ypred1 = torch.sqrt(Ypred1)

            diff = abs(1-sq_Ypred0/sq_Yobs0)+abs(1-sq_Ypred1/sq_Yobs1)+\
                abs(1-sq_Yobs0/sq_Ypred0)+abs(1-sq_Yobs1/sq_Ypred1)
        
            loss_speed = torch.sum(diff)/Yobs.shape[0]
        
        elif self.args.speed_loss == 'hamilton':
            p = self.args.hamilton_p
            loss_speed = torch.abs(Yobs[:,0]**p / Ypred0**(p) - 1) + torch.abs(Yobs[:,1]**p / Ypred1**(p) - 1) \
                + torch.abs(Ypred0**(p) / Yobs[:,0]**p - 1) + torch.abs(Ypred1**(p) / Yobs[:,1]**p - 1)
            loss_speed = torch.sum(loss_speed) / Yobs.shape[0]
        else:
            raise NotImplementedError
        
        return loss_speed, dtau
    
    def get_dataset(self):
        if self.args.SE2:
            self.dataset = db.Room2DDataset(self.Params['DataPath'], self.args.use_no_coll_split)
        else:
            self.dataset = db.RoomDataset(self.Params['DataPath'], self.args.data_format, self.args.use_no_coll_split)
    
    def get_dataloader(self):
        self.get_dataset()
        
        dim = self.dim
        
        dists=torch.norm(self.dataset.data[:,:dim]-self.dataset.data[:,dim:2*dim],dim=1)
        weights = dists.max()-dists
        # weights = dists

        weights = torch.clamp(
                weights/weights.max(), self.Params['Training']['Resampling Bounds'][0], self.Params['Training']['Resampling Bounds'][1])
        
        if self.args.use_no_coll_split:
            num_workers = 1
            prefetch_factor = 1250
        elif self.args.enc_d < 4:
            num_workers = 8
            prefetch_factor = 1250
        else:
            num_workers = 1
            prefetch_factor = 1250
        
        print('num_workers: {}'.format(num_workers))
        if self.args.sampler == 'dynamic_weighted':
            self.post_weights, _ = torch.min(self.dataset.data[:, 2*dim:2*dim+2], dim=1)
            print(self.post_weights)
            self.post_weights = torch.clamp(self.post_weights/self.post_weights.max(), self.Params['Training']['Resampling Bounds'][0], self.Params['Training']['Resampling Bounds'][1])
        if self.args.sampler == 'weighted':
            if len(weights) > 2**24:
                raise ValueError('WeightedRandomSampler does not support weights larger than 2^24')
                train_sampler_wei = CustomWeightedRandomSampler(
                    weights, len(weights), replacement=True)
            else:
                train_sampler_wei = WeightedRandomSampler(
                        weights, len(weights), replacement=True)
            train_loader_wei = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=int(self.Params['Training']['Batch Size']),
                sampler=train_sampler_wei,
                pin_memory=True,
                num_workers = 1,
                prefetch_factor = 1250
            )
        elif self.args.sampler == 'batch_weighted':
            batch_sampler = BatchWeightedRandomSampler(
                int(self.Params['Training']['Batch Size']),
                weights, len(weights), replacement=True
            )
            train_loader_wei = FastDataLoader(
                self.dataset,
                batch_sampler=batch_sampler,
                pin_memory=False,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor
            )
        elif self.args.sampler == 'dynamic_weighted':
            self.dynamic_batch_sampler = DynamicWeightedRandomSampler(
                int(self.Params['Training']['Batch Size']),
                weights, len(weights), replacement=True
            )
            train_loader_wei = torch.utils.data.DataLoader(
                self.dataset,
                batch_sampler=self.dynamic_batch_sampler,
                pin_memory=True,
                num_workers = 1,
                prefetch_factor = 1250
            )
        else:
            train_loader_wei = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=int(self.Params['Training']['Batch Size']),
                shuffle=True,
                pin_memory=True,
                num_workers = 8,
                prefetch_factor = 1250
            )

        speed = self.dataset.data[:,2*dim:2*dim+2]
        self.logger.info('speed min: '+str(speed.min()))
        self.logger.info(f'speed max: {speed.max()}')
        return train_loader_wei
    
    def get_optimizer(self):
        return torch.optim.AdamW(
                self.network.parameters(),
                lr=self.Params['Training']['Learning Rate'],
                weight_decay=0.1)

    def train(self):
        self.Params['Training'] = {}
        self.Params['Training']['Batch Size'] = self.args.batch_size
        self.Params['Training']['Number of Epochs'] = self.args.epochs
        self.Params['Training']['Resampling Bounds'] = [0.2, 0.95]
        self.Params['Training']['Print Every * Epoch'] = 1
        self.Params['Training']['Save Every * Epoch'] = 10
        self.Params['Training']['Learning Rate'] = self.args.lr
        
        # Parameters to alter during training
        self.total_train_loss = []
        
        self.vis_dir = os.path.join(self.Params['ModelPath'], 'vis_train_plot')
        os.makedirs(self.vis_dir, exist_ok=True)
        self.vis_dir_xz = os.path.join(self.vis_dir, 'xz')
        self.vis_dir_xy = os.path.join(self.vis_dir, 'xy')
        os.makedirs(self.vis_dir_xz, exist_ok=True)
        os.makedirs(self.vis_dir_xy, exist_ok=True)
        self.ckpt_dir = os.path.join(self.Params['ModelPath'], 'ckpts')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tensorboard_dir = os.path.join(self.Params['ModelPath'], 'tensorboard')
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        
        """path loss"""
        self.beta = self.args.beta
        self.beta1 = self.args.beta1
        self.beta2 = self.args.beta2
        self.beta3 = self.args.beta3
        # self.beta4 = self.args.beta4
        self.beta5 = self.args.beta5
        self.beta6 = self.args.beta6

        if self.args.use_path_loss:
            if self.args.SE2:
                env_w, env_h, obstacles = read_hdf5(self.Params['DataPath'], 'env', data_names=['env_w', 'env_h', 'obstacles'])
                env_bound = shapely.geometry.Polygon([[-env_w/2, -env_h/2], [-env_w/2, env_h/2], [env_w/2, env_h/2], [env_w/2, -env_h/2]])
                obstacles = np.concatenate((obstacles, np.array(env_bound.exterior.xy)[np.newaxis,...]), axis=0) # add env boundary to obstacles

                self.obstacles = obstacles
                self.robot = torch.from_numpy(np.load(os.path.join(self.Params['DataPath'], 'robot.npy'))).float().to(self.Params['Device'])
                get_dists = get_dist_fn_SE2(self.args.dist_fn_type, self.dim, self.robot, self.obstacles, self.args.in_dim, self.args.h_dim, self.args.ckpt_path, 'cuda', self.args.solid)
                self.check_collision = lambda x: check_collision_SE2(x, get_dists, self.args.offset)
            else:
                file_path = glob.glob(os.path.join(self.Params['DataPath'], '*.off'))[0]
                v, f = igl.read_triangle_mesh(file_path)
                v = torch.from_numpy(v).float().to(self.Params['Device'])
                f = torch.from_numpy(f).long().to(self.Params['Device'])
                triangles = v[f].unsqueeze(0)
                get_dist = get_dist_fn(self.args.dist_fn_type, self.args.robot_name, self.Params['DataPath'], self.dim, triangles, self.Params['Device'], use_coll_robot=True,
                                       rotate_axis=self.args.rotate_axis, in_dim=self.args.in_dim, h_dim=self.args.h_dim,ckpt_path=self.args.ckpt_path, sim=self.args.sim)
                self.check_collision = lambda x: check_collision_SE3(get_dist, x, self.args.offset)
        
        self.bbox = None
        try:
            bbox = read_hdf5(self.Params['DataPath'], data_names=['bbox'])[0]
            self.bbox = bbox
            print('bbox:', bbox)
        except Exception as e:
            print('no bbox find!')
                
        self.network = NN(self.Params['Device'], self.dim, self.args)
        self.network.apply(self.network.init_weights)
        self.network.to(self.Params['Device'])

        self.optimizer = self.get_optimizer()
        
        if self.args.sched == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.Params['Training']['Number of Epochs'],
            )
            
        prev_diff = 1.0
        current_diff = 1.0
        
        start_epoch = 1
        if self.args.resume:
            model_path = glob.glob(os.path.join(self.ckpt_dir, 'Model_Epoch_{:0>5d}_*.pt'.format(self.args.resume_epoch)))
            if model_path:
                model_path = model_path[0]
                start_epoch = self.args.resume_epoch
                self.logger.info("=> loading checkpoint '{} (epoch {})".format(model_path, start_epoch))
                self.load(model_path)
                start_epoch += 1
                current_diff = self.resume_train_loss
            else:
                self.logger.info("=> no checkpoint found at '{}'".format(self.ckpt_dir))
                
        train_loader_wei = self.get_dataloader()

        if not self.args.SE2:
            grid = None

        current_state = deepcopy(self.network.state_dict())
        current_optimizer = deepcopy(self.optimizer.state_dict())
        prev_state_queue = []
        prev_optimizer_queue = []

        self.l0 = 500

        self.l1 = 500
        
        if self.args.record_loss:
            loss_recorder = LossRecoder(self.tensorboard_dir)
            loss_names = [
                'loss_train',
                'loss_speed_train',
            ]
            if self.args.use_path_loss:
                loss_names.extend([
                    'loss_path_T_Ts_Tg_train',
                    'loss_path_T_T_new_train',
                    'loss_path_T_Ts_train',
                    'loss_path_T_Tg_train',
                    'loss_path_T_T_coll_s_train',
                    'loss_path_T_T_coll_g_train',
                ])
            for loss_name in loss_names:
                loss_recorder.register_loss(loss_name)

        # use tqdm for progress bar
        for epoch in range(start_epoch, self.Params['Training']['Number of Epochs']+1):
            start = time.time()
            total_train_loss = 0

            total_diff=0
            path_loss_flag = self.args.use_path_loss and epoch > self.args.path_loss_start_epoch

            if path_loss_flag:
                total_no_coll = 0
                total_coll = 0
                total_T_T_new_meet = 0
                total_T_T_new_not_meet = 0
                total_T_T_sw_not_meet = 0
                total_T_T_wg_not_meet = 0
                total_T_T_sum_not_meet = 0
            
            self.lamb = min(1.0,max(0,(epoch-self.l0)/self.l1))
            
            prev_state_queue.append(current_state)
            prev_optimizer_queue.append(current_optimizer)
            if(len(prev_state_queue)>5):
                prev_state_queue.pop(0)
                prev_optimizer_queue.pop(0)
            
            current_state = deepcopy(self.network.state_dict())
            current_optimizer = deepcopy(self.optimizer.state_dict())

            if self.args.sched == 'custom':
                self.optimizer.param_groups[0]['lr']  = max(5e-4*(1-epoch/self.l0),1e-5)
            
            prev_diff = current_diff
            it=0
            
            while True:
                total_train_loss = 0
                total_diff = 0
                if self.args.record_loss:
                    loss_recorder.clear_loss()

                end = time.time()
                print('time before iter: {}'.format(end-start))

                start = time.time()
                for i, batch_data in tqdm(enumerate(train_loader_wei, 0), total=len(train_loader_wei), desc=f"Epoch {epoch}/{self.Params['Training']['Number of Epochs']}"):
                    data = batch_data

                    data = data.to(self.Params['Device'])

                    points = data[:,:2*self.dim]
                    speed = data[:,2*self.dim:2*self.dim+2]

                    if self.args.use_no_coll_split:
                        no_coll_mask = data[:, 2*self.dim+2].bool()
                    
                    tau, Xp = self.network.out(points)
                    
                    r = self.get_speed_loss(tau, Xp, speed)
                    loss_value, dtau = r[:2]
                    
                    loss_speed = loss_value.detach().clone()
                    
                    if path_loss_flag:
                        T = self.Time_from_tau(tau, Xp)

                    if path_loss_flag:
                        path_losses, path_loss_metrics = self.get_path_loss(Xp[no_coll_mask], dtau[no_coll_mask], tau[no_coll_mask], T[no_coll_mask], epoch)
                        
                        loss_path_1, loss_path, loss_path_T_Ts, loss_path_T_Tg, loss_path_T_T_coll_s, loss_path_T_T_coll_g = path_losses[:6]
                        total_no_coll += path_loss_metrics[0]
                        total_coll += path_loss_metrics[1]
                        total_T_T_new_meet += path_loss_metrics[2]
                        total_T_T_new_not_meet += path_loss_metrics[3]
                        total_T_T_sw_not_meet += path_loss_metrics[4]
                        total_T_T_wg_not_meet += path_loss_metrics[5]
                        total_T_T_sum_not_meet += path_loss_metrics[6]

                        loss_value += loss_path_1
                        loss_value += loss_path_T_Ts
                        loss_value += loss_path_T_Tg

                    loss_n = loss_value
                    
  
                    loss_value.backward()

                    # Update parameters
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # Record loss
                    if self.args.record_loss:
                        
                        loss_recorder.update_loss('loss_train', loss_value.item())
                        loss_recorder.update_loss('loss_speed_train', loss_speed.item())

                        if path_loss_flag:
                            loss_recorder.update_loss('loss_path_T_Ts_Tg_train', loss_path.item())
                            loss_recorder.update_loss('loss_path_T_T_new_train', loss_path_1.item())
                            loss_recorder.update_loss('loss_path_T_Ts_train', loss_path_T_Ts.item())
                            loss_recorder.update_loss('loss_path_T_Tg_train', loss_path_T_Tg.item())
                            loss_recorder.update_loss('loss_path_T_T_coll_s_train', loss_path_T_T_coll_s.item())
                            loss_recorder.update_loss('loss_path_T_T_coll_g_train', loss_path_T_T_coll_g.item())

                    total_train_loss += loss_value.detach().item()
                    total_diff += loss_n.detach().item()
                    
                    del points, speed, loss_value, loss_n

                total_train_loss /= len(train_loader_wei)
                total_diff /= len(train_loader_wei)

                if self.args.sched != 'custom':
                    self.scheduler.step()
                
                current_diff = total_diff
                diff_ratio = current_diff/prev_diff
                
                if epoch == 1:
                    ratio_max = self.args.start_epoch_loss_val
                elif self.args.use_path_loss and epoch == self.args.path_loss_start_epoch + 1:
                    ratio_max = 1.4
                    self.logger.info("Relax ratio check!(max_ratio: 1.4)\nPath loss start epoch = {} -- Loss = {:.4e} -- diff ratio={:.2f}: current loss {:.4f}\tprev loss {:.4f}".format(
                        epoch, total_diff, diff_ratio, current_diff, prev_diff))
                else:
                    ratio_max = self.args.loss_ratio
                    
                cond =  diff_ratio < ratio_max and diff_ratio > 0

                if (cond):#2.0
                    break
                else:
                    it += 1
                    with torch.no_grad():
                        if epoch == 1 and it > 5:
                            self.logger.info(f">>> Epoch 1 repeat too many times, reinitialize the network")
                            self.network.cpu()
                            self.network.apply(self.network.init_weights)
                            self.network.to(self.Params['Device'])
                            current_state = pickle.loads(pickle.dumps(self.network.state_dict()))
                            del self.optimizer
                            self.optimizer = self.get_optimizer()
                            current_optimizer = pickle.loads(pickle.dumps(self.optimizer.state_dict()))
                            prev_state_queue[0] = current_state
                            prev_optimizer_queue[0] = current_optimizer
                            it = 0
                        else:
                            q_size = len(prev_state_queue)
                            random_number = random.randint(0, min(q_size - 1, epoch-1))
                            self.network.load_state_dict(prev_state_queue[random_number], strict=True)
                            self.optimizer.load_state_dict(prev_optimizer_queue[random_number])
                    
                    self.logger.info("RepeatEpoch = {} -- Loss = {:.4e} -- diff ratio={:.2f}: current loss {:.4f}\tprev loss {:.4f}".format(
                        epoch, total_diff, diff_ratio, current_diff, prev_diff))
                
            start = time.time()
            self.total_train_loss.append(total_train_loss)
            
            
            if self.args.record_loss:
                loss_recorder.write_loss(epoch)
            
            if epoch % self.Params['Training']['Print Every * Epoch'] == 0:
                with torch.no_grad():
                    self.logger.info("Epoch = {} -- Loss = {:.4e}".format(
                        epoch, total_diff))

            if (epoch % self.Params['Training']['Save Every * Epoch'] == 0) or (epoch == self.Params['Training']['Number of Epochs']) or (epoch == 1):
                
                if self.args.SE2:
                    self.plot(epoch, total_diff, plane='xy')
                else:
                    self.plot(epoch, total_diff, grid, plane='xy')
                    self.plot(epoch, total_diff, grid, plane='xz')
                with torch.no_grad():
                    self.save(epoch=epoch, val_loss=total_diff)

            if self.args.sampler == 'dynamic_weighted' and epoch == self.args.dynmaic_epoch:
                print('change sampling strategy')
                self.dynamic_batch_sampler.set_weights(self.post_weights)
            end = time.time()
            print("Time after iter = {:.2f} s".format(end-start))
            if path_loss_flag:
                print('total no collision points: {}'.format(total_no_coll))
                print('total collision points: {}'.format(total_coll))
                print('total T >= T_new: {}'.format(total_T_T_new_meet))
                print('total T < T_new: {}'.format(total_T_T_new_not_meet))
                print('total T < T_sw: {}'.format(total_T_T_sw_not_meet))
                print('total T < T_wg: {}'.format(total_T_T_wg_not_meet))
                print('total T > T_sw + T_wg: {}'.format(total_T_T_sum_not_meet))
        loss_recorder.close()          

    def save(self, epoch='', val_loss=''):
        '''
            Saving a instance of the model
        '''
        torch.save({'epoch': epoch,
                    'model_state_dict': self.network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': self.total_train_loss,
                    'val_loss': self.total_train_loss}, os.path.join(self.ckpt_dir, 'Model_Epoch_{}_ValLoss_{:.6e}.pt'.format( str(epoch).zfill(5), val_loss)))

    def load(self, filepath, test=False):
        
        checkpoint = torch.load(
            filepath, map_location=torch.device(self.Params['Device']))
        
        if test:
            # TODO: check epoch
            self.network = NN(self.Params['Device'], self.dim, args=self.args)

        self.network.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        if test:
            self.network.to(torch.device(self.Params['Device']))
            self.network.float()
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if test:
            self.network.eval()
        else:
            self.network.train()
            
        self.resume_train_loss = checkpoint['train_loss'][-1]

    def init_blank_network(self):
        self.network = NN(self.Params['Device'], self.dim, args=self.args)

    def TravelTimes(self, Xp):
        Xp = Xp.to(torch.device(self.Params['Device']))
        
        tau, coords = self.network.out(Xp)
       
        D = Xp[:,self.dim:]-Xp[:,:self.dim]
        
        T0 = torch.einsum('ij,ij->i', D, D)

        TT = torch.sqrt(T0)/tau[:, 0]
            
        del Xp, tau, T0
        return TT
    
    def Time_from_tau(self, tau, Xp):
        
        D = Xp[:,self.dim:]-Xp[:,:self.dim]
        
        T0 = torch.einsum('ij,ij->i', D, D)

        TT = torch.sqrt(T0)/tau[:, 0]

        return TT
    
    def Tau(self, Xp):
        Xp = Xp.to(torch.device(self.Params['Device']))

        tau, coords = self.network.out(Xp)
        
        return tau

    def Speed(self, Xp):
        Xp = Xp.to(torch.device(self.Params['Device']))

        tau, Xp = self.network.out(Xp)
        dtau = self.gradient(tau, Xp)        
        
        with torch.no_grad():
            D = Xp[:,self.dim:]-Xp[:,:self.dim]
            T0 = torch.einsum('ij,ij->i', D, D)

            DT1 = dtau[:,self.dim:]

            T1    = T0*torch.einsum('ij,ij->i', DT1, DT1)
            T2    = 2*tau[:,0]*torch.einsum('ij,ij->i', DT1, D)

            T3    = tau[:,0]**2
            
            S = (T1-T2+T3)

            Ypred = T3 / torch.sqrt(S)
        
        del Xp, tau, dtau, T0, T1, T2, T3
        return Ypred
    
    def Speeds(self, Xp):
        tau, Xp = self.network.out(Xp)
        dtau = self.gradient(tau, Xp)        
        
        with torch.no_grad():
            D = Xp[:,self.dim:]-Xp[:,:self.dim]
            T0 = torch.einsum('ij,ij->i', D, D)

            DT1 = dtau[:,self.dim:]

            T1    = T0*torch.einsum('ij,ij->i', DT1, DT1)
            T2    = 2*tau[:,0]*torch.einsum('ij,ij->i', DT1, D)

            T3    = tau[:,0]**2
            
            S = (T1-T2+T3)

            Ypred1 = T3 / torch.sqrt(S)
            
            DT1 = dtau[:,:self.dim]
            T1 = T0*torch.einsum('ij,ij->i', DT1, DT1)
            T2 = 2*tau[:,0]*torch.einsum('ij,ij->i', DT1, -D)
            
            S = (T1-T2+T3)
            Ypred0 = T3 / torch.sqrt(S)
        return torch.stack((Ypred0, Ypred1), dim=1)
    
    def grad_from_dtau(self, Xp, dtau, tau):
        # x = start; y = goal
        D = Xp[:,self.dim:]-Xp[:,:self.dim]
        T0 = torch.sqrt(torch.einsum('ij,ij->i', D, D)).unsqueeze(1) # ||x - y||
        T3 = tau**2

        V0 = D # y - x
        V1 = dtau[:,self.dim:] # dtau_goal
        
        Y1 = 1/(T0*tau)*V0           # (y - x) / (||x - y|| * tau)
        Y2 = T0/T3*V1                # (dtau_goal * ||x - y||) / tau**2


        Ypred1 = -(Y1-Y2)            # -dT(x, y) / dy
        Spred1 = torch.norm(Ypred1, dim=1, keepdim=True)  # 1/ S(y)
        Ypred1 = 1/Spred1**2*Ypred1  # -S^2(y) * dT(x, y) / dy

        V0=-D  # x - y
        V1=dtau[:,:self.dim] # dtau_start
        
        Y1 = 1/(T0*tau)*V0           # (x - y) / (||x - y|| * tau)
        Y2 = T0/T3*V1                # (dtau_start * ||x - y||) / tau**2

        Ypred0 = -(Y1-Y2)            # -dT(x, y) / dx
        Spred0 = torch.norm(Ypred0, dim=1, keepdim=True)  # 1/ S(x)

        Ypred0 = 1/Spred0**2*Ypred0  # -S^2(x) * dT(x, y) / dx
        
        return torch.cat((Ypred0, Ypred1),dim=1)
    
    def Gradient(self, Xp):
        Xp = Xp.to(torch.device(self.Params['Device']))
       
        #Xp.requires_grad_()

        tau, Xp = self.network.out(Xp)
        dtau = self.gradient(tau, Xp)
        
        with torch.no_grad():
            # x = start; y = goal
            D = Xp[:,self.dim:]-Xp[:,:self.dim]
            T0 = torch.sqrt(torch.einsum('ij,ij->i', D, D)).unsqueeze(1) # ||x - y||
            T3 = tau**2

            V0 = D # y - x
            V1 = dtau[:,self.dim:] # dtau_goal
            
            Y1 = 1/(T0*tau)*V0           # (y - x) / (||x - y|| * tau)
            Y2 = T0/T3*V1                # (dtau_goal * ||x - y||) / tau**2


            Ypred1 = -(Y1-Y2)            # -dT(x, y) / dy
            Spred1 = torch.norm(Ypred1, dim=1, keepdim=True)  # 1/ S(y)
            Ypred1 = 1/Spred1**2*Ypred1  # -S^2(y) * dT(x, y) / dy

            V0=-D  # x - y
            V1=dtau[:,:self.dim] # dtau_start
            
            Y1 = 1/(T0*tau)*V0           # (x - y) / (||x - y|| * tau)
            Y2 = T0/T3*V1                # (dtau_start * ||x - y||) / tau**2

            Ypred0 = -(Y1-Y2)            # -dT(x, y) / dx
            Spred0 = torch.norm(Ypred0, dim=1, keepdim=True)  # 1/ S(x)

            Ypred0 = 1/Spred0**2*Ypred0  # -S^2(x) * dT(x, y) / dx
        
        return torch.cat((Ypred0, Ypred1),dim=1)
     
    def plot(self, epoch, total_train_loss, grid=None, plane='xy'):
        if self.args.SE2:
            xmin = [-self.args.env_w / 2, -self.args.env_h / 2]
            xmax = [self.args.env_w / 2, self.args.env_h / 2]
            spacing = 1 / 80
            X, Y = np.meshgrid(np.arange(xmin[0], xmax[0], spacing), np.arange(xmin[1], xmax[1], spacing))
        else:
            
            if self.bbox is not None:
                if plane == 'xy':
                    xmin = [self.bbox[0, 0], self.bbox[0, 1]]
                    xmax = [self.bbox[1, 0], self.bbox[1, 1]]
                elif plane == 'yz':
                    xmin = [self.bbox[0, 1], self.bbox[0, 2]]
                    xmax = [self.bbox[1, 1], self.bbox[1, 2]]
                elif plane == 'xz':
                    xmin = [self.bbox[0, 0], self.bbox[0, 2]]
                    xmax = [self.bbox[1, 0], self.bbox[1, 2]]
                else:
                    raise ValueError('plane must be xy, yz or xz')
                limit = np.max(self.bbox[1, :2] - self.bbox[0, :2]).item() / 2.
            else:
                limit = 0.5
                xmin     = [-limit,-limit]
                xmax     = [limit,limit]
            spacing=limit/40.0
            X,Y      = np.meshgrid(np.arange(xmin[0],xmax[0],spacing),np.arange(xmin[1],xmax[1],spacing))

        Xsrc = [0]*self.dim
        
        Xsrc[0] = self.pos[0]
        Xsrc[1] = self.pos[1]

        Xsrc = self.pos

        XP       = np.zeros((len(X.flatten()),2*self.dim))
        XP[:,:self.dim] = Xsrc
        if plane == 'xy':
            XP[:,self.dim+0]  = X.flatten()
            XP[:,self.dim+1]  = Y.flatten()
            # XP[:,self.dim+2]  = self.pos[2]
            if self.dim > 2:
                XP[:,self.dim+2]  = 0
            vis_dir = self.vis_dir_xy
        elif plane == 'yz':
            XP[:,self.dim+0]  = self.pos[0]
            XP[:,self.dim+1]  = Y.flatten()
            XP[:,self.dim+2]  = X.flatten()
        elif plane == 'xz':
            XP[:,self.dim+0]  = X.flatten()
            XP[:,self.dim+1]  = self.pos[1]
            XP[:,self.dim+2]  = Y.flatten()
            vis_dir = self.vis_dir_xz
        else:
            raise ValueError('plane must be xy, yz or xz')
        if self.args.dof > 3:
            XP[:,self.dim+3:] = self.pos[3:]
        XP = Variable(Tensor(XP)).to(self.Params['Device'])
            
        with torch.no_grad():
            tt = self.TravelTimes(XP)
            tau = self.Tau(XP)

        ss = self.Speed(XP)#*5
        
        TT = tt.to('cpu').data.numpy().reshape(X.shape)
        V  = ss.to('cpu').data.numpy().reshape(X.shape)
        TAU = tau.to('cpu').data.numpy().reshape(X.shape)

        fig = plt.figure()

        ax = fig.add_subplot(111)
        quad1 = ax.pcolormesh(X,Y,V,vmin=0,vmax=self.args.vmax)
        # ax.contour(X,Y,TT,np.arange(0,3,0.05), cmap='bone', linewidths=0.5)#0.25
        ax.contour(X,Y,TT,20, cmap='bone', linewidths=0.5)#0.25
        plt.colorbar(quad1,ax=ax, pad=0.1, label='Predicted Velocity')
        plt.savefig(os.path.join(vis_dir, "plots"+str(epoch)+f"_{plane}_"+str(round(total_train_loss,4))+"_0.jpg"),bbox_inches='tight')

        plt.close(fig)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        quad1 = ax.pcolormesh(X,Y,TAU,vmin=0,vmax=self.args.vmax)
        # ax.contour(X,Y,TT,np.arange(0,3,0.05), cmap='bone', linewidths=0.5)#0.25
        ax.contour(X,Y,TT,20, cmap='bone', linewidths=0.5)#0.25
        plt.colorbar(quad1,ax=ax, pad=0.1, label='Predicted Tau')
        plt.savefig(os.path.join(vis_dir, "tauplots"+str(epoch)+f"_{plane}_"+str(round(total_train_loss,4))+"_0.jpg"),bbox_inches='tight')

        plt.close(fig)
