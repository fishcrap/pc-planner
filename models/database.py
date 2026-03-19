import numpy as np
import torch
import os

from utils.util import read_hdf5

class RoomDataset(torch.utils.data.Dataset):
    def __init__(self, path, data_format, use_no_coll_split=False):
        if data_format == 'h5':
            start_points, goal_points, start_speed, goal_speed = read_hdf5(path, data_names=['start_points', 'goal_points', 'start_speed', 'goal_speed'])
            points = np.concatenate((start_points, goal_points), axis=1)
            speed = np.stack((start_speed, goal_speed), axis=1)
        else:
            points = np.load(os.path.join(path, 'sampled_points.npy'))
            speed = np.load(os.path.join(path, 'speed.npy'))

        if use_no_coll_split:
            no_coll_mask = read_hdf5(path, data_names=['no_coll_mask'])[0]
            self.no_coll_mask = torch.from_numpy(no_coll_mask)
            print('total no coll samples:', self.no_coll_mask.sum())

        print('point shape:', points.shape)
        print('speed shape:', speed.shape)
        
        points = torch.from_numpy(points).float()
        speed = torch.from_numpy(speed).float()
        self.data=torch.cat((points,speed),dim=1)
        
        if use_no_coll_split:
            self.data = torch.cat((self.data, self.no_coll_mask.unsqueeze(1)), dim=1)
    
    def __getitem__(self, index):
        data = self.data[index]
        return data
    
    def __getitems__(self, indices):
        data = self.data[indices]
        return data
    
    def __len__(self):
        return self.data.shape[0]

class Room2DDataset(torch.utils.data.Dataset):
    def __init__(self, path, use_no_coll_split=False):
        start_points, goal_points, start_speed, goal_speed = read_hdf5(path, data_names=['start_points', 'goal_points', 'start_speed', 'goal_speed'])
        points = np.concatenate((start_points, goal_points), axis=1)
        speed = np.stack((start_speed, goal_speed), axis=1)

        print('point shape:', points.shape)
        print('speed shape:', speed.shape)
        
        if use_no_coll_split:
            no_coll_mask = read_hdf5(path, data_names=['no_coll_mask'])[0]
            self.no_coll_mask = torch.from_numpy(no_coll_mask)
            print('total no coll samples:', self.no_coll_mask.sum())

        points = torch.from_numpy(points).float()
        speed = torch.from_numpy(speed).float()
        self.data=torch.cat((points,speed), dim=1)
        
        if use_no_coll_split:
            self.data = torch.cat((self.data, self.no_coll_mask.unsqueeze(1)), dim=1)

        self.dim = start_points.shape[1]
        dists = torch.linalg.norm(points[:, :self.dim]-points[:, self.dim:], dim=1)
        weights = dists.max() - dists
        weights = torch.clamp(weights / weights.max(), 0.2, 0.95)

        self.weights = weights
    
    def __getitem__(self, index):
        data = self.data[index]
        return data
    
    def __getitems__(self, indices):
        data = self.data[indices]
        return data
    
    def __len__(self):
        return self.data.shape[0]
    
    def get_init_weights(self):
        return self.weights
    
    def get_weights(self):
        self.weights += self.diff_step

        return self.weights

class UDFDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        points, dists = read_hdf5(path, data_names=['points', 'dists'], file_name='udf_data')

        print('point shape:', points.shape)
        print('dists shape:', dists.shape)

        points = torch.from_numpy(points).float()
        dists = torch.from_numpy(dists).float()
        self.data=torch.cat((points, dists.unsqueeze(1)),dim=1)

    def __getitem__(self, index):
        data = self.data[index]
        return data

    def __getitems__(self, indices):
        data = self.data[indices]
        return data

    def __len__(self):
        return self.data.shape[0]

class SdfDataset(torch.utils.data.Dataset):
    def __init__(self, path, data_format, use_query_points):
        if data_format == 'h5':
            sample_points, dists = read_hdf5(path, data_names=['sample_points', 'dists'])        
            if use_query_points:
                query_points = read_hdf5(path, data_names=['points'])[0]
                print(query_points.shape)
                query_points = torch.from_numpy(query_points).float()
        else:
            #TODO
            pass
        
        sample_points = torch.from_numpy(sample_points).float()
        dists = torch.from_numpy(dists).float()
        dists = dists.unsqueeze(-1)
        print('point shape:', sample_points.shape)
        print('speed shape:', dists.shape)
        data=torch.cat((sample_points,dists), dim=1)
        if use_query_points:
            data = torch.cat((data,query_points), dim =1)
        self.data = data
    
    def __getitem__(self, index):
        data = self.data[index]
        return data
    
    def __getitems__(self, indices):
        data = self.data[indices]
        return data
    
    def __len__(self):
        return self.data.shape[0]
    
class SdfDataset_(torch.utils.data.Dataset):
    def __init__(self, path, point_dists_flag):
        self.point_dists_flag = point_dists_flag
        sample_points, dists = read_hdf5(path, data_names=['sample_points', 'dists'])
        if point_dists_flag:
            point_dists = read_hdf5(path, data_names=['point_dists'])[0]
            self.point_dists = torch.from_numpy(point_dists).float() # (n, sample_size)
            print('point dists shape:', point_dists.shape)

        sample_points = torch.from_numpy(sample_points).float() # (n, sample_size, 2)
        dists = torch.from_numpy(dists).float() # (n, )

        print('point shape:', sample_points.shape)
        print('dists shape:', dists.shape)
        self.sample_points = sample_points
        self.dists = dists
    
    def __getitems__(self, indices):
        sample_points = self.sample_points[indices]
        dists = self.dists[indices]
        if self.point_dists_flag:
            point_dists = self.point_dists[indices]
            return sample_points, dists, point_dists
        return sample_points, dists
    
    def __len__(self):
        return self.dists.shape[0]

class StEikDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        points, dists, mnfld_points = read_hdf5(path, data_names=['points', 'dists', 'mnfld_points'], file_name='udf_data')

        print('nonmanifold point shape:', points.shape)
        print('dists shape:', dists.shape)
        print('manifold points shape', mnfld_points.shape)

        points = torch.from_numpy(points).float()
        dists = torch.from_numpy(dists).float()
        mnfld_points = torch.from_numpy(mnfld_points).float()
        
        self.points = points
        self.dists = dists
        self.mnfld_points = mnfld_points
        
        self.data = torch.cat((points, dists.unsqueeze(1)),dim=1)
        self.mnfld_batch_size = 10000
        self.n_mnfld = mnfld_points.shape[0]

    def __getitems__(self, indices):
        mnfld_indices = torch.randperm(self.n_mnfld)
        data = {"nonmnfld_points": self.points[indices], "dists": self.dists[indices],  "mnfld_points": self.mnfld_points[mnfld_indices]}
        return data

    def __len__(self):
        return self.data.shape[0]