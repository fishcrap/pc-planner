import torch

import torch.nn as nn
from models.database import UDFDataset
from tqdm import tqdm
import os
from dataloader.fast_sampler import BatchRandomSampler
from dataloader.fast_dataloader import SimpleDataLoader
from utils.loss_utils.loss_recorder import LossRecoder
from .blocks import C_encoder_block

class UDF(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            C_encoder_block(hidden_dim, hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return torch.abs(self.fc(x))


class trainer:
    def __init__(self, args):
        self.args = args

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

        dataset = UDFDataset(self.args.data_dir)
        dataloader = SimpleDataLoader(dataset, BatchRandomSampler(self.args.batch_size, dataset))
        self.model = UDF(in_dim=self.args.dim, hidden_dim=self.args.h_dim)
        self.model.train()
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        loss_recorder = LossRecoder(tensorboard_dir)
        loss_names = ['loss', 'dist_loss', 'eik_loss']
        loss_recorder.register_losses(loss_names)
        
        epochs = self.args.epochs
        progress_bar = tqdm(range(1, epochs + 1), total=epochs, desc=f"Epoch 1/{epochs}")
        for epoch in progress_bar:
            total_loss = 0
            for b_idx, data in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{epochs}", leave=False):
                data = data.to(self.device).squeeze()
                points = data[:, :self.dim]
                dists = data[:, -1]
                self.optimizer.zero_grad()

                points.requires_grad = True
                pred_d = self.model(points)


                # loss = criterion(pred_d.squeeze(), dists) + self.args.alpha * self.get_eikonal_loss(points, pred_d)
                dist_loss = criterion(pred_d.squeeze(), dists)
                eik_loss = self.args.alpha * self.get_eikonal_loss(points, pred_d)
                loss = dist_loss + eik_loss
                
                
                loss.backward()
                self.optimizer.step()
                
                loss_dict = {'loss': loss.item(), 'dist_loss': dist_loss.item(), 'eik_loss': eik_loss.item()}
                loss_recorder.update_losses(loss_dict)
                idx = b_idx + epoch * len(dataloader)
                loss_recorder.write_batch_loss(idx)

                total_loss += loss.item()
            loss_avg = total_loss / len(dataloader)
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


 
