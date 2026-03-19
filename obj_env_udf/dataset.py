import numpy as np
import torch

from utils.util import read_hdf5


class GenUDFDataset(torch.utils.data.Dataset):
    def __init__(self, path,):
        
        self.path = path
        self.n_robot = 167
        # self.n_robot = 2
        self.robot_perm_idx = 0
        
        dists, sparse_robot = read_hdf5(self.path, data_names=['dists', 'robot'], group_name='/train/sparse/0')
        self.numsamples = dists.shape[0]
        self.sparse_robot = sparse_robot

    def __getitem__(self, index):
        robot_points = self.robot_points[index]
        min_dists = self.min_dists[index]
        robot_dists = self.robot_dists[index]
        return robot_points, min_dists, robot_dists
    
    def get_robots(self):
        return self.sparse_robot, self.dense_robot
    
    def next_robot(self):
        if self.robot_perm_idx % self.n_robot == 0:
            self.robot_perm = np.random.permutation(self.n_robot)
            self.robot_perm_idx = 0
        idx = self.robot_perm[self.robot_perm_idx]
        self.robot_perm_idx += 1
        robot_points, robot_dists, sparse_robot = read_hdf5(self.path, data_names=['robot_points', 'robot_dists', 'robot'], group_name=f'/train/sparse/{idx}')
        dense_robot, min_dists = read_hdf5(self.path, data_names=['robot', 'dists'], group_name=f'/train/dense/{idx}')
        
        self.robot_points = torch.from_numpy(robot_points).float() # (n, sample_size, 3)
        self.min_dists = torch.from_numpy(min_dists).float() # (n, )
        self.robot_dists = torch.from_numpy(robot_dists).float() #(n, sample_size)
        self.sparse_robot = torch.from_numpy(sparse_robot).float() # (sample_size,)
        self.dense_robot = torch.from_numpy(dense_robot).float() # (sample_size,)
        
    
    def __getitems__(self, indices):
        robot_points = self.robot_points[indices]
        min_dists = self.min_dists[indices]
        robot_dists = self.robot_dists[indices]
        return robot_points, min_dists, robot_dists
    
    def __len__(self):
        return self.numsamples
    
class GenUDFBaselineDataset(torch.utils.data.Dataset):
    def __init__(self, path,):
        
        self.path = path
        self.n_robot = 167
        self.robot_perm_idx = 0
        
        dists = read_hdf5(self.path, data_names=['dists'], group_name='/train/sparse/0')[0]
        self.numsamples = dists.shape[0]

    def __getitem__(self, index):
        robot_states = self.robot_states[index]
        min_dists = self.min_dists[index]
        return robot_states, min_dists
    
    def get_robots(self):
        return self.dense_robot
    
    def next_robot(self):
        if self.robot_perm_idx % self.n_robot == 0:
            self.robot_perm = np.random.permutation(self.n_robot)
            self.robot_perm_idx = 0
        idx = self.robot_perm[self.robot_perm_idx]
        self.robot_perm_idx += 1
        robot_states = read_hdf5(self.path, data_names=['points'], group_name=f'/train/sparse/{idx}')[0]
        dense_robot, min_dists = read_hdf5(self.path, data_names=['robot', 'dists'], group_name=f'/train/dense/{idx}')
        
        self.robot_states = torch.from_numpy(robot_states).float() # (n, sample_size, 3)
        self.min_dists = torch.from_numpy(min_dists).float() # (n, )
        self.dense_robot = torch.from_numpy(dense_robot).float() # (sample_size,)
        
    
    def __getitems__(self, indices):
        robot_states = self.robot_states[indices]
        min_dists = self.min_dists[indices]
        return robot_states, min_dists
    
    def __len__(self):
        return self.numsamples 