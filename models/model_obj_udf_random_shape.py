import torch

import torch.nn as nn
from models.database import UDFDataset, SdfDataset, SdfDataset_
from tqdm import tqdm
import os
from dataloader.fast_sampler import BatchRandomSampler
from dataloader.fast_dataloader import SimpleDataLoader
from utils.loss_utils.loss_recorder import LossRecoder
import glob
import random
from .blocks import C_encoder_block

class ObjectUDF(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=128, n_points=10):
        super().__init__()
        self.in_dim = in_dim
        self.n_points = n_points
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            *[C_encoder_block(hidden_dim, hidden_dim, hidden_dim) for _ in range(3)],
        )
        self.symm_aggregation = nn.MaxPool1d(kernel_size=n_points)
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            C_encoder_block(hidden_dim, hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, x):
        x = x.reshape(-1, self.in_dim) # (n * n_points, in_dim)
        x = self.fc(x) # (n * n_points, hidden_dim)
        x = x.reshape(-1, self.n_points, x.shape[-1]) # (n, n_points, hidden_dim)
        x = self.symm_aggregation(x.transpose(1, 2)).squeeze(dim=-1) # (n, hidden_dim)
        x = self.fc2(x) #(n, 1)
        x = torch.abs(x)
        return x

class ObjectMinUDF(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=128, n_points=10):
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

        self.net = {
            'object_udf': ObjectUDF,
            'object_min_udf': ObjectMinUDF,
        } 

        # dataset = UDFFullDataset(self.args.data_dir)
        # dataloader = torch.torch.utils.data.DataLoader(
        #     dataset,
        #     batch_size=1,
        #     shuffle=False,
        #     pin_memory=True,
        #     num_workers=1,
        # )
        self.ObjectMinUDF_flag = (self.args.net_name == 'object_min_udf')
        dataset = SdfDataset_(self.args.data_dir,self.ObjectMinUDF_flag)
        dataloader = SimpleDataLoader(dataset, BatchRandomSampler(self.args.batch_size, dataset))
        
        start_epoch = 1
        self.model = self.net[self.args.net_name](in_dim=self.args.dim, hidden_dim=self.args.h_dim, n_points=10)
        self.model.train()
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
        criterion = nn.L1Loss()

        loss_recorder = LossRecoder(tensorboard_dir)
        # loss_names = ['loss', 'dist_loss', 'eik_loss']
        loss_names = ['loss', 'dist_loss']
        if self.ObjectMinUDF_flag:
            loss_names.append('point_dist_loss')
        loss_recorder.register_losses(loss_names)
        
        epochs = self.args.epochs
        progress_bar = tqdm(range(start_epoch, epochs + 1), total=epochs+1-start_epoch, desc=f"Epoch 1/{epochs}")

        gt_mask = torch.ones(self.args.batch_size*dataset.sample_points.shape[1], 1)
        #print(gt_mask.shape)
        all_indices = torch.where(gt_mask)[0]
        selected_indices =  random.sample(all_indices.tolist(), int(0.1*gt_mask.shape[0]))


        for epoch in progress_bar:
            total_loss = 0
            #for b_idx, (sample_points, dists) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{epochs}", leave=False):
            for b_idx, batch_dataset in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{epochs}", leave=False):
                if self.ObjectMinUDF_flag:
                    sample_points, dists, point_dists = batch_dataset
                    point_dists = point_dists.to(self.device)
                else:
                    sample_points, dists = batch_dataset

                sample_points = sample_points.to(self.device)
                dists = dists.to(self.device)
                # data = data.to(self.device)
                # points = data[:, :self.dim]
                # dists = data[:, -1]
                self.optimizer.zero_grad()

                # points.requires_grad_(True)
                pred_d = self.model(sample_points)

                if self.ObjectMinUDF_flag:
                    pred_min_d, pred_point_d= pred_d
                    #print('point_dists: ', point_dists.reshape(-1,1).shape)
                    point_dist_loss = criterion(pred_point_d[selected_indices],point_dists.reshape(-1,1)[selected_indices])

                else:
                    pred_min_d = pred_d
                # loss = criterion(pred_d.squeeze(), dists) + self.args.alpha * self.get_eikonal_loss(points, pred_d)
                dist_loss = criterion(pred_min_d.squeeze(), dists)

                loss = dist_loss

                if self.ObjectMinUDF_flag:
                    loss += point_dist_loss
                # eik_loss = self.args.alpha * self.get_eikonal_loss(points, pred_d)
                # loss = dist_loss + eik_loss
                
                
                loss.backward()
                self.optimizer.step()
                
                # loss_dict = {'loss': loss.item(), 'dist_loss': dist_loss.item(), 'eik_loss': eik_loss.item()}
                loss_dict = {'loss': loss.item(), 'dist_loss': dist_loss.item()}
                if self.ObjectMinUDF_flag:
                    loss_dict.update({'point_dist_loss':point_dist_loss.item()})
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
        
        loss_recorder.close()
    
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
        