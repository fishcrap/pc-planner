import torch

import torch.nn as nn
from obj_env_udf.dataset import GenUDFDataset, GenUDFBaselineDataset
from tqdm import tqdm
import os
from dataloader.fast_sampler import BatchRandomSampler
from dataloader.fast_dataloader import SimpleDataLoader
from dataloader.fast_dataloader import FastDataLoader
from utils.loss_utils.loss_recorder import LossRecoder
import glob
import random
from .encoders.conv_pointnet import ConvPointnet
from easydict import EasyDict as edict

class C_encoder_block(nn.Module):
    """
    residual block
    """
    def __init__(self, in_channel, hidden_size, out_channel):
        super().__init__()
        self.act = nn.ReLU()
        self.enc = nn.Sequential(
            nn.Linear(in_channel, hidden_size),
            self.act,
            nn.Linear(hidden_size, out_channel),
        )
    
    def forward(self, x):
        y = self.enc(x)
        y += x
        x = self.act(y)
        return x


class ObjectMinUDF(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=128, n_points=10, train=True):
        super().__init__()
        self.in_dim = in_dim
        self.n_points = n_points
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            *[C_encoder_block(hidden_dim, hidden_dim, hidden_dim) for _ in range(2)],
        )

        self.symm_aggregation = nn.MaxPool1d(kernel_size=n_points)
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            C_encoder_block(hidden_dim, hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        if train:

            #add distance supervision for each point
            self.fc3 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                C_encoder_block(hidden_dim, hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, x):
        x = x.reshape(-1, self.in_dim) # (n * n_points, in_dim)
        x = self.fc(x) # (n * n_points, hidden_dim)

        y = self.fc3(x) #(n*n_points, 1)
        y = torch.abs(y)

        x = x.reshape(-1, self.n_points, x.shape[-1]) # (n, n_points, hidden_dim)
        x = self.symm_aggregation(x.transpose(1, 2)).squeeze(dim=-1) # (n, hidden_dim)
        x = self.fc2(x) #(n, 1)
        x = torch.abs(x)
        return x,y
    
    def out(self, x):
        x = x.reshape(-1, self.in_dim) # (n * n_points, in_dim)
        x = self.fc(x) # (n * n_points, hidden_dim)

        # y = self.fc3(x) #(n*n_points, 1)
        # y = torch.abs(y)

        x = x.reshape(-1, self.n_points, x.shape[-1]) # (n, n_points, hidden_dim)
        x = self.symm_aggregation(x.transpose(1, 2)).squeeze(dim=-1) # (n, hidden_dim)
        x = self.fc2(x) #(n, 1)
        x = torch.abs(x)
        return x


class ObjectGenUDF(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=128):
        super().__init__()
        self.in_dim = in_dim
        
        encoder_args = {
            "c_dim": 256, # latent size 
            "dim": 3,
            "hidden_dim": 64,
            "plane_resolution": 64,
            "unet": True,
            "unet_kwargs": {"depth": 4, "merge_mode": "concat", "start_filts": 32},
        }
        latent_size = encoder_args['c_dim']
        self.shape_encoder = ConvPointnet(**encoder_args)
        
        self.pos_state_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            *[C_encoder_block(hidden_dim, hidden_dim, hidden_dim) for _ in range(2)],
        )
        
        self.pos_state_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            C_encoder_block(hidden_dim, hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
        
        self.pc_feat_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # C_encoder_block(hidden_dim, hidden_dim, hidden_dim)
        )
        
        self.fuser = nn.Sequential(
            nn.Linear(hidden_dim + latent_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.symm_aggregation = nn.AdaptiveAvgPool1d(1)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            C_encoder_block(hidden_dim, hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        self.latent_vec = None
        self.latent_vec_dict = {}
        self.freeze()
        
    def clear_latent_vec(self):
        self.latent_vec = None
    
    def __freeze_shape_encoder(self):
        for param in self.shape_encoder.parameters():
            param.requires_grad = False
    
    def freeze(self):
        self.__freeze_shape_encoder()
    
    
    def forward(self, sparse_pc, sparse_robot, dense_robot):
        """
        args:
            sparse_pc: (n[batch_size], n_points, 3)
            sparse_robot: (n_points, 3)
            dense_robot: (m_points, 3)
        return:
            pred_min_dists: (n, 1)
            pred_dists: (n, n_points, 1)
        """
        if self.latent_vec is None:
            self.latent_vec = self.shape_encoder(dense_robot.unsqueeze(0), sparse_robot.unsqueeze(0)).squeeze(0) # (n_points, latent_size)
            self.latent_vec = self.latent_vec.repeat(sparse_pc.shape[0], 1, 1) #(n, n_points, latent_size)
        x = sparse_pc.reshape(-1, self.in_dim) # (n * n_points, in_dim)
        x = self.pos_state_encoder(x) # (n * n_points, hidden_dim)

        y = self.pos_state_decoder(x) # (n * n_points, 1)
        y = torch.abs(y) # g(x,y,z, tht, phi, psi)
        n = sparse_pc.shape[0]
        y = y.reshape(n, -1, 1) # (n, n_points, 1)
        
        x = x.reshape(n, -1, x.shape[-1]) # (n, n_points, hidden_dim)
        # x = self.pc_feat_extractor(x) # (n, n_points, hidden_dim)
        
        x = torch.cat([x, self.latent_vec], dim=-1) # (n, n_points, hidden_dim + latent_size)
        x = self.fuser(x) # (n, n_points, hidden_dim)
        x = self.symm_aggregation(x.transpose(1, 2)).squeeze(dim=-1) # (n, hidden_dim)
        x = self.decoder(x) # (n, 1)
        
        x = torch.abs(x)
        return x, y
    
    def out_batch(self, sparce_pc, sparse_robot, dense_robot):
        """
        args:
            sparse_pc: (n, b, n_points, 3)
            sparse_robot: (b, n_points, 3)
            dense_robot: (b, m_points, 3)
        return:
            pred_min_dists: (n, b, 1)
        """
        if self.latent_vec is None:
            self.latent_vec = self.shape_encoder(dense_robot, sparse_robot) # (b, n_points, latent_size)

        x = self.pos_state_encoder(sparce_pc) # (n, b, n_points, hidden_dim)
        
        latent_vec = self.latent_vec.repeat(sparce_pc.shape[0], 1, 1, 1) # (n, b, n_points, latent_size)
        x = torch.cat([x, latent_vec], dim=-1) # (n, b, n_points, hidden_dim + latent_size)
        x = self.fuser(x) # (n, b, n_points, hidden_dim)
        n = x.shape[0]
        x = self.symm_aggregation(x.view(-1, x.shape[-2], x.shape[-1]).transpose(-2, -1)).squeeze(dim=-1) # (n * b, hidden_dim)
        x = x.view(n, -1, x.shape[-1])
        x = self.decoder(x) # (n, b, 1)
        
        x = torch.abs(x)
        return x
    
    def out_seq(self, sparse_pc, sparse_robot, dense_robot, idx):
        """
        args:
            sparse_pc: (n[batch_size], n_points, 3)
            sparse_robot: (n_points, 3)
            dense_robot: (m_points, 3)
        return:
            pred_min_dists: (n, 1)
        """
        if self.latent_vec_dict.get(idx, None) is None:
            self.latent_vec_dict[idx] = self.shape_encoder(dense_robot.unsqueeze(0), sparse_robot.unsqueeze(0)).squeeze(0) # (n_points, latent_size)
        # x = sparse_pc.reshape(-1, self.in_dim) # (n * n_points, in_dim)
        x = self.pos_state_encoder(sparse_pc) # (n, n_points, hidden_dim)
        
        # n = sparse_pc.shape[0]
        # x = x.reshape(n, -1, x.shape[-1]) # (n, n_points, hidden_dim)
        
        latent_vec = self.latent_vec_dict[idx].repeat(sparse_pc.shape[0], 1, 1) #(n, n_points, latent_size)
        x = torch.cat([x, latent_vec], dim=-1) # (n, n_points, hidden_dim + latent_size)
        x = self.fuser(x) # (n, n_points, hidden_dim)
        x = self.symm_aggregation(x.transpose(1, 2)).squeeze(dim=-1) # (n, hidden_dim)
        x = self.decoder(x) # (n, 1)
        
        x = torch.abs(x)
        return x

    def out(self, sparse_pc, sparse_robot, dense_robot):
        """
        args:
            sparse_pc: (n[batch_size], n_points, 3)
            sparse_robot: (n_points, 3)
            dense_robot: (m_points, 3)
        return:
            pred_min_dists: (n, 1)
        """
        if self.latent_vec is None:
            self.latent_vec = self.shape_encoder(dense_robot.unsqueeze(0), sparse_robot.unsqueeze(0)).squeeze(0) # (n_points, latent_size)
        x = sparse_pc.reshape(-1, self.in_dim) # (n * n_points, in_dim)
        x = self.pos_state_encoder(x) # (n * n_points, hidden_dim)
        
        n = sparse_pc.shape[0]
        x = x.reshape(n, -1, x.shape[-1]) # (n, n_points, hidden_dim)
        # x = self.pc_feat_extractor(x) # (n, n_points, hidden_dim)
        
        latent_vec = self.latent_vec.repeat(sparse_pc.shape[0], 1, 1) #(n, n_points, latent_size)
        x = torch.cat([x, latent_vec], dim=-1) # (n, n_points, hidden_dim + latent_size)
        x = self.fuser(x) # (n, n_points, hidden_dim)
        x = self.symm_aggregation(x.transpose(1, 2)).squeeze(dim=-1) # (n, hidden_dim)
        x = self.decoder(x) # (n, 1)
        
        x = torch.abs(x)
        return x

class ObjectGenUDF_Baseline(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=128):
        super().__init__()
        self.in_dim = in_dim
        
        encoder_args = {
            "c_dim": 256, # latent size 
            "dim": 3,
            "hidden_dim": 64,
            "plane_resolution": 64,
            "unet": True,
            "unet_kwargs": {"depth": 4, "merge_mode": "concat", "start_filts": 32},
        }
        latent_size = encoder_args['c_dim']
        self.shape_encoder = ConvPointnet(**encoder_args)
        
        self.pos_state_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            *[C_encoder_block(hidden_dim, hidden_dim, hidden_dim) for _ in range(2)],
        )

        
        self.fuser = nn.Sequential(
            nn.Linear(hidden_dim + latent_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.symm_aggregation = nn.AdaptiveAvgPool1d(1)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            C_encoder_block(hidden_dim, hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        self.latent_vec = None
        self.freeze()
        
    def clear_latent_vec(self):
        self.latent_vec = None
    
    def __freeze_shape_encoder(self):
        for param in self.shape_encoder.parameters():
            param.requires_grad = False
    
    def freeze(self):
        self.__freeze_shape_encoder()
    
    
    def forward(self, states, dense_robot):
        """
        args:
            states: (n[batch_size], 3)
            dense_robot: (m_points, 3)
        return:
            pred_min_dists: (n, 1)
            pred_dists: (n, n_points, 1)
        """
        if self.latent_vec is None:
            self.latent_vec = self.shape_encoder(dense_robot.unsqueeze(0), dense_robot.unsqueeze(0)).squeeze(0) # (m_points, latent_size)
            self.latent_vec = self.latent_vec.repeat(states.shape[0], 1, 1) #(n, m_points, latent_size)
        x = self.pos_state_encoder(states) # (n, hidden_dim)
        
        x = x.unsqueeze(1).repeat(1, dense_robot.shape[0], 1) # (n, m_points, hidden_dim)
        
        x = torch.cat([x, self.latent_vec], dim=-1) # (n, m_points, hidden_dim + latent_size)
        x = self.fuser(x) # (n, m_points, hidden_dim)
        x = self.symm_aggregation(x.transpose(1, 2)).squeeze(dim=-1) # (n, hidden_dim)
        x = self.decoder(x) # (n, 1)
        
        x = torch.abs(x)
        return x

    def out(self, states, dense_robot):
        """
        args:
            states: (n[batch_size], 3)
            dense_robot: (m_points, 3)
        return:
            pred_min_dists: (n, 1)
            pred_dists: (n, n_points, 1)
        """
        if self.latent_vec is None:
            self.latent_vec = self.shape_encoder(dense_robot.unsqueeze(0), dense_robot.unsqueeze(0)).squeeze(0) # (m_points, latent_size)
        x = self.pos_state_encoder(states) # (n, hidden_dim)
        
        x = x.unsqueeze(1).repeat(1, dense_robot.shape[0], 1) # (n, m_points, hidden_dim)

        latent_vec = self.latent_vec.repeat(states.shape[0], 1, 1) #(n, m_points, latent_size)
        x = torch.cat([x, latent_vec], dim=-1) # (n, m_points, hidden_dim + latent_size)
        x = self.fuser(x) # (n, m_points, hidden_dim)
        x = self.symm_aggregation(x.transpose(1, 2)).squeeze(dim=-1) # (n, hidden_dim)
        x = self.decoder(x) # (n, 1)
        
        x = torch.abs(x)
        return x

class trainer:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

    def gradient(self, y, x, create_graph=True):                                                               

        grad_y = torch.ones_like(y)                                                                 
        grad_x = torch.autograd.grad(y, x, grad_y, only_inputs=True, retain_graph=True, create_graph=create_graph)[0]
        
        return grad_x
    
    def get_eikonal_loss(self, points, pred_dists):
        grad = self.gradient(pred_dists, points)
        grad_norm = torch.linalg.norm(grad, dim=1)
        criterion = nn.MSELoss()
        loss = criterion(grad_norm, torch.ones_like(grad_norm))
        return loss


    def train(self):
        self.device = self.args.device
        self.dim = self.args.dim
        self.ckpt_dir = os.path.join(self.args.exp_dir, 'ckpts')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.save_freq = self.args.save_freq
        tensorboard_dir = os.path.join(self.args.exp_dir, 'tensorboard')
        os.makedirs(tensorboard_dir, exist_ok=True)

        if self.args.net_type == 'baseline':
            dataset = GenUDFBaselineDataset(self.args.data_dir)
        else:
            dataset = GenUDFDataset(self.args.data_dir)

        self.dataset = dataset
        # dataloader = SimpleDataLoader(dataset, BatchRandomSampler(self.args.batch_size, dataset))
        
        # dataloader = torch.torch.utils.data.DataLoader(
        #     dataset,
        #     batch_sampler=BatchRandomSampler(self.args.batch_size, dataset),
        #     pin_memory=True,
        #     num_workers=8,
        #     prefetch_factor=1250
        # )
        
        dataloader = FastDataLoader(
            dataset,
            batch_sampler=BatchRandomSampler(self.args.batch_size, dataset),
            pin_memory=True,
            num_workers=8,
            prefetch_factor=1250
        )

        
        def copy_parameters_ft(model, pretrained_dict, logger, verbose=True):
            # ref: https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3
            new_state_dict = {}
            for param_name in pretrained_dict:
                if 'encoder' in param_name:
                    newname = param_name.replace('encoder', 'shape_encoder')
                    new_state_dict[newname] = pretrained_dict[param_name]
                else:
                    new_state_dict[param_name] = pretrained_dict[param_name]

            pretrained_dict = new_state_dict
        
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                            k in model_dict and pretrained_dict[k].size() == model_dict[k].size()}

            # print(model_dict.keys())
            # print(pretrained_dict.keys)
            if verbose:
                logger.info('=' * 27)
                logger.info('Restored Params and Shapes:')
                for k, v in pretrained_dict.items():
                    logger.info(f'{k} : {v.size()}')
                logger.info('=' * 68)
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            return model
        
        start_epoch = 1
        if self.args.net_type == 'baseline':
            self.model = ObjectGenUDF_Baseline(in_dim=self.args.dim, hidden_dim=self.args.h_dim)
        else:
            self.model = ObjectGenUDF(in_dim=self.args.dim, hidden_dim=self.args.h_dim)
        self.model.train()
        
        if self.args.load_pretrain_enc:
            ckpt = torch.load(self.args.pretrain_path)
            pretrain_model_dict = ckpt['state_dict']
            copy_parameters_ft(self.model, pretrain_model_dict, self.logger)
    
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        if self.args.resume:
            model_path = glob.glob(os.path.join(self.ckpt_dir, 'Model_Epoch_{:0>5d}_*.pt'.format(self.args.resume_epoch)))
            if model_path:
                model_path = model_path[0]
                start_epoch = self.args.resume_epoch
                self.logger.info("=> loading checkpoint '{} (epoch {})".format(model_path, start_epoch))
                self.load(model_path)
                start_epoch += 1
            else:
                self.logger.info("=> no checkpoint found at '{}'".format(self.ckpt_dir))

        
        # criterion = nn.MSELoss()
        self.criterion = nn.L1Loss()

        loss_recorder = LossRecoder(tensorboard_dir)
        if self.args.net_type in ['baseline', 'genudf_no_dist']:
            loss_names = ['loss', 'min_dist_loss']
        else:
            loss_names = ['loss', 'dist_loss', 'min_dist_loss']
        loss_recorder.register_losses(loss_names)
        
        param = sum([p.nelement() for p in self.model.parameters()])
        print('Parameters: {} [M]'.format(param / 10. ** 6))
        
        # for name, param in self.model.named_parameters():
        #     print(f'{name}: requires_grad={param.requires_grad}')
        
        if self.args.net_type == 'genudf':
            selected_indices = torch.randint(0, dataset.sparse_robot.shape[0], (self.args.batch_size,)).to(self.device)
        
        epochs = self.args.epochs
        progress_bar = tqdm(range(start_epoch, epochs + 1), total=epochs+1-start_epoch, desc=f"Epoch {start_epoch}/{epochs}")

        for epoch in progress_bar:
            total_loss = 0
            dataset.next_robot()
            for b_idx, batch_data in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{epochs}", leave=False):
                if self.args.net_type == 'baseline':
                    loss, loss_dict = self.train_step_baseline(batch_data)
                elif self.args.net_type == 'genudf_no_dist':
                    loss, loss_dict = self.train_step_no_dist_loss(batch_data)
                else:
                    loss, loss_dict = self.train_step(batch_data, selected_indices)

                loss.backward()
                self.optimizer.step()
                
                loss_recorder.update_losses(loss_dict)
                idx = b_idx + epoch * len(dataloader)
                loss_recorder.write_batch_loss(idx)

                total_loss += loss.item()
            loss_avg = total_loss / len(dataloader)
            if epoch < epochs:
                progress_bar.set_description(f'Epoch {epoch+1}/{epochs}')
                progress_bar.set_postfix({'loss': f'{loss_avg:.9f}'})
            if epoch % self.save_freq == 0 or epoch == epochs or epoch == 1:
                self.save(epoch, loss)

            loss_recorder.write_loss(epoch, write_batch=False)
            loss_recorder.clear_loss()
            self.model.clear_latent_vec()
            
        loss_recorder.close()
    
    def train_step(self, batch_data, selected_indices):
        # robot_points, min_dists, robot_dists, sparse_robot, dense_robot = batch_data
        robot_points, min_dists, robot_dists = batch_data
        sparse_robot, dense_robot = self.dataset.get_robots()
        
        robot_points = robot_points.to(self.device)
        min_dists = min_dists.to(self.device)
        robot_dists = robot_dists.to(self.device)
        sparse_robot = sparse_robot.to(self.device)
        dense_robot = dense_robot.to(self.device)

        self.optimizer.zero_grad()

        pred_min_d, pred_robot_d = self.model(robot_points, sparse_robot, dense_robot)

        dist_loss = self.criterion(pred_robot_d.squeeze().gather(dim=-1, index=selected_indices.unsqueeze(1)).squeeze(),
                                robot_dists.gather(dim=-1, index=selected_indices.unsqueeze(1)).squeeze())

        min_dist_loss = self.criterion(pred_min_d.squeeze(), min_dists)

        loss = min_dist_loss + self.args.alpha * dist_loss
        
        loss_dict = {'loss': loss.item(), 'dist_loss': dist_loss.item(), 'min_dist_loss': min_dist_loss.item()}
        
        return loss, loss_dict
    
    def train_step_baseline(self, batch_data):
        robot_states, min_dists = batch_data
        dense_robot = self.dataset.get_robots()
        
        robot_states = robot_states.to(self.device)
        min_dists = min_dists.to(self.device)
        dense_robot = dense_robot.to(self.device)

        self.optimizer.zero_grad()

        pred_min_d = self.model(robot_states, dense_robot)

        min_dist_loss = self.criterion(pred_min_d.squeeze(), min_dists)

        loss = min_dist_loss
        
        loss_dict = {'loss': loss.item(), 'min_dist_loss': min_dist_loss.item()}
        
        return loss, loss_dict

    def train_step_no_dist_loss(self, batch_data):
        robot_points, min_dists, robot_dists = batch_data
        sparse_robot, dense_robot = self.dataset.get_robots()
        
        robot_points = robot_points.to(self.device)
        min_dists = min_dists.to(self.device)
        # robot_dists = robot_dists.to(self.device)
        sparse_robot = sparse_robot.to(self.device)
        dense_robot = dense_robot.to(self.device)

        self.optimizer.zero_grad()

        pred_min_d, pred_robot_d = self.model(robot_points, sparse_robot, dense_robot)

        min_dist_loss = self.criterion(pred_min_d.squeeze(), min_dists)

        loss = min_dist_loss
        
        loss_dict = {'loss': loss.item(), 'min_dist_loss': min_dist_loss.item()}

        return loss, loss_dict

    def save(self, epoch, loss):
        '''
            Saving a instance of the model
        '''
        torch.save({'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': loss},
                    os.path.join(self.ckpt_dir, 'Model_Epoch_{}_Loss_{:.6e}.pt'.format(str(epoch).zfill(5), loss)))

    def load(self, ckpt_path):
        
        checkpoint = torch.load(
            ckpt_path, map_location=torch.device(self.device))

        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        