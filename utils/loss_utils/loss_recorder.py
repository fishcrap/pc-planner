import numpy as np
from tensorboardX import SummaryWriter
import time
import os
import collections
from torch._six import string_classes

class LossMeter():
    def __init__(self, loss_name):
        self._loss_name = loss_name
        self._loss = []
        
    def clear_loss(self):
        self._loss = []
    
    def update_loss(self, loss_item):
        self._loss.append(loss_item)
    
    @property
    def loss_name(self):
        return self._loss_name
    
    @property
    def loss_mean(self):
        return np.mean(self._loss)
    
    @property
    def loss(self):
        return self._loss
    
    @property
    def last_loss(self):
        return self._loss[-1]
    
    @property
    def is_empty(self):
        return len(self._loss) == 0
    
    def __len__(self):
        return len(self._loss)

class LossRecoder():
    def __init__(self, dir):
        self.loss_names = []
        self.loss_dict = {}
        self.writer = SummaryWriter(os.path.join(dir, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))
        
    def register_loss(self, loss_name):
        self.loss_names.append(loss_name)
        loss_meter = LossMeter(loss_name)
        self.loss_dict[loss_name] = loss_meter
    
    def register_losses(self, loss_names):
        if isinstance(loss_names, collections.abc.Sequence) and not isinstance(loss_names, string_classes):
            for loss_name in loss_names:
                self.register_loss(loss_name)
        else:
            return TypeError("loss_names should be a sequence of strings, but got {}".format(type(loss_names)))
        
    def clear_loss(self):
        for loss_meter in self.loss_dict.values():
            loss_meter.clear_loss()
    
    def update_loss(self, loss_name, loss):
        self.loss_dict[loss_name].update_loss(loss)
        
    def update_losses(self, loss_dict):
        if isinstance(loss_dict, collections.abc.Mapping):
            for loss_name, loss in loss_dict.items():
                self.update_loss(loss_name, loss)
        else:
            return TypeError("loss_dict should be a mapping of loss_name to loss, but got {}".format(type(loss_dict)))
        
    def write_loss(self, epoch, write_batch=True):
        valid_loss_names = [loss_name for loss_name in self.loss_names if not self.loss_dict[loss_name].is_empty]
        for loss_name in valid_loss_names:
            self.writer.add_scalar(loss_name + '_epoch', self.loss_dict[loss_name].loss_mean, epoch)
        
        if write_batch:
            batch_size = len(self.loss_dict[valid_loss_names[0]])
            for i in range(batch_size):
                for loss_name in valid_loss_names:
                    self.writer.add_scalar(loss_name + '_batch', self.loss_dict[loss_name].loss[i], i + epoch * batch_size)
    
    def write_batch_loss(self, idx):
        valid_loss_names = [loss_name for loss_name in self.loss_names if not self.loss_dict[loss_name].is_empty]
        for loss_name in valid_loss_names:
            self.writer.add_scalar(loss_name + '_batch', self.loss_dict[loss_name].last_loss, idx)
    
    def write_spec_loss(self, epoch, loss_name, loss):
        self.writer.add_scalar(loss_name + '_epoch', loss, epoch)
    
    def close(self):
        self.writer.close()