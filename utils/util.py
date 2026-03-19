import logging
import sys
import torch


import os
import h5py

import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml


def get_logger(log_path=None, console=True):
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    # logger.setLevel(logging.DEBUG)
    # print(log_path, flush=True)
    assert(log_path or console)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if log_path:
        handler = logging.FileHandler(log_path, encoding="UTF-8")
        fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)

    if console:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.DEBUG)
    
        logger.addHandler(console)
    return logger

def write_hdf5(save_dir, data_name, data, file_name='data', group_name=None):
    with h5py.File(os.path.join(save_dir, f'{file_name}.h5'), 'w') as f:
        if group_name is not None:
            if group_name not in f:
                f.create_group(group_name)
            g = f[group_name]
        else:
            g = f
        if isinstance(data_name, str):
            g.create_dataset(data_name, data=data)
        elif isinstance(data_name, list):
            for i, name in enumerate(data_name):
                g.create_dataset(name, data=data[i])
        else:
            raise NotImplementedError

def append_hdf5(save_dir, data_name, data, file_name='data', group_name=None):
    with h5py.File(os.path.join(save_dir, f'{file_name}.h5'), 'a') as f:
        if group_name is not None:
            if group_name not in f:
                f.create_group(group_name)
            g = f[group_name]
        else:
            g = f
        if isinstance(data_name, str):
            g.create_dataset(data_name, data=data)
        elif isinstance(data_name, list):
            for i, name in enumerate(data_name):
                g.create_dataset(name, data=data[i])
        else:
            raise NotImplementedError
        
def modify_hdf5(save_dir, data_name, data, file_name='data'):
    with h5py.File(os.path.join(save_dir, f'{file_name}.h5'), 'a') as f:
        if isinstance(data_name, str):
            del f[data_name]
            f.create_dataset(data_name, data=data)
        elif isinstance(data_name, list):
            for i, name in enumerate(data_name):
                del f[name]
                f.create_dataset(name, data=data[i])
        else:
            raise NotImplementedError


def read_hdf5(save_dir, file_name='data', data_names=['start_points', 'goal_points', 'start_speed', 'goal_speed'], group_name=None):
    data = []
    if file_name.endswith('.h5'):
        file_name = file_name[:-3]
    with h5py.File(os.path.join(save_dir, f'{file_name}.h5'), 'r') as f:
        if group_name is not None:
            g = f[group_name]
        else:
            g = f
        for data_name in data_names:
            dataset = g[data_name]
            if dataset.shape == ():
                data.append(dataset[()])
            else:
                data.append(dataset[:])
    return data

def print_hdf5(save_dir, file_name='data', level=-1, print_full_name: bool = False, print_attrs: bool = True):
    if file_name.endswith('.h5'):
        file_name = file_name[:-3]
    with h5py.File(os.path.join(save_dir, f'{file_name}.h5'), 'r') as f:
        _print_hdf5(f, level, print_full_name, print_attrs)

def _print_hdf5(h5py_obj, level=-1, print_full_name: bool = False, print_attrs: bool = True) -> None:
    """ Prints the name and shape of datasets in a H5py HDF5 file.
    Parameters
    ----------
    h5py_obj: [h5py.File, h5py.Group]
        the h5py.File or h5py.Group object
    level: int
        What level of the file tree you are in
    print_full_name
        If True, the full tree will be printed as the name, e.g. /group0/group1/group2/dataset: ...
        If False, only the current node will be printed, e.g. dataset:
    print_attrs
        If True: print all attributes in the file
    Returns
    -------
    None
    """
    def is_group(f):
        return type(f) == h5py._hl.group.Group

    def is_dataset(f):
        return type(f) == h5py._hl.dataset.Dataset

    def print_level(level, n_spaces=5) -> str:
        if level == -1:
            return ''
        prepend = '|' + ' ' * (n_spaces - 1)
        prepend *= level
        tree = '|' + '-' * (n_spaces - 2) + ' '
        return prepend + tree

    for key in h5py_obj.keys():
        entry = h5py_obj[key]
        name = entry.name if print_full_name else os.path.basename(entry.name)
        if is_group(entry):
            print('{}{}'.format(print_level(level), name))
            _print_hdf5(entry, level + 1, print_full_name=print_full_name)
        elif is_dataset(entry):
            shape = entry.shape
            dtype = entry.dtype
            print('{}{}: {} {}'.format(print_level(level), name,
                                       shape, dtype))
    if level == -1:
        if print_attrs:
            print('attrs: ')
            for key, value in h5py_obj.attrs.items():
                print(' {}: {}'.format(key, value))


def seed_everything(manual_seed):
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    
def dump_config(cfg_dir, cfg_name:str, args):
    if not cfg_name.endswith('.yaml'):
        cfg_name += '.yaml'
    config_file = os.path.join(cfg_dir, f'{cfg_name}')
    with open(config_file, 'w') as f:
        yaml.dump(vars(args), f)
    print('==> Config file saved to {}'.format(config_file))
    
def dump_train_config(args):
    if args.resume:
        config_name = 'config_resume.yaml'
    else:
        config_name = 'config.yaml'
    dump_config(args.exp_dir, config_name, args)
    
def load_config(cfg_path):
    with open(cfg_path, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)