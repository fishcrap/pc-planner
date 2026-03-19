import torch

import torch.nn as nn
from tqdm import tqdm
import os
from models.UDF.StEik import Network as StEikNet
from dataloader.fast_sampler import BatchRandomSampler
from dataloader.fast_dataloader import SimpleDataLoader
from utils.loss_utils.loss_recorder import LossRecoder
import torch.optim.lr_scheduler as lr_scheduler
from models.UDF.losses import Loss
from models.database import StEikDataset

class trainer:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

    def train(self):
        args = self.args
        self.device = args.device
        self.dim = args.dim
        self.ckpt_dir = os.path.join(args.exp_dir, 'ckpts')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.save_freq = args.save_freq
        
        tensorboard_dir = os.path.join(args.exp_dir, 'tensorboard')
        os.makedirs(tensorboard_dir, exist_ok=True)

        dataset = StEikDataset(args.data_dir)
        dataloader = SimpleDataLoader(dataset, BatchRandomSampler(args.batch_size, dataset))
        self.model = StEikNet(latent_size=args.latent_size, in_dim=3, decoder_hidden_dim=args.h_dim, nl=args.nl,
                  encoder_type=args.encoder_type, decoder_n_hidden_layers=args.decoder_n_hidden_layers,
                  init_type=args.init_type, neuron_type=args.neuron_type, sphere_init_params=args.sphere_init_params)
        self.model.train()
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=1.0) # Does nothing
        criterion = Loss(weights=args.loss_weights, loss_type=args.loss_type, div_decay=args.div_decay, div_type=args.div_type)
        
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        msg = f"Number of parameters in the current model: {n_params}"
        self.logger.info(msg)
        print(msg)
        
        loss_recorder = LossRecoder(tensorboard_dir)
        loss_recorder.register_losses(criterion.loss_names)
        
        epochs = args.epochs
        progress_bar = tqdm(range(1, epochs + 1), total=epochs, desc=f"Epoch 1/{epochs}")
        for epoch in progress_bar:
            total_loss = 0
            for b_idx, data in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{epochs}", leave=False):
                nonmnfld_points = data["nonmnfld_points"].to(self.device)
                dists = data["dists"].to(self.device)
                mnfld_points = data["mnfld_points"].to(self.device)
                self.optimizer.zero_grad()

                nonmnfld_points.requires_grad_(True)
                mnfld_points.requires_grad_(True)
                
                nonmnfld_points = nonmnfld_points.unsqueeze(0)
                mnfld_points = mnfld_points.unsqueeze(0)
                dists = dists.unsqueeze(0)

                pred = self.model(nonmnfld_points, mnfld_points)

                loss_dict = criterion(pred, mnfld_points, nonmnfld_points, dists)

                loss_dict['loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.)
                self.optimizer.step()
                total_loss += loss_dict['loss'].item()

                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        loss_dict[k] = v.item()
                loss_recorder.update_losses(loss_dict)
                idx = b_idx + epoch * len(dataloader)
                loss_recorder.write_batch_loss(idx)
                
                criterion.update_div_weight(b_idx + (epoch - 1) * len(dataloader), epochs * len(dataloader),
                                args.div_decay_params)  # assumes batch size of 1
            loss_avg = total_loss / len(dataloader)
            progress_bar.set_description(f'Epoch {epoch+1}/{epochs}')
            progress_bar.set_postfix({'loss': f'{loss_avg:.6f}'})
            if epoch % self.save_freq == 0 or epoch == epochs or epoch == 1:
                self.save(epoch, loss_avg)
            scheduler.step()
            loss_recorder.write_loss(epoch, write_batch=False)
            self.logger.info("Epoch = {} -- Loss = {:.4e}".format(epoch, loss_avg))

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
