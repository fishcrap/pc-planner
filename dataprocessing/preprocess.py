import sys

sys.path.append('.')

from dataprocessing.convert_to_scaled_off import to_off
from dataprocessing.speed_sampling_gpu_refactor import sample_speed
from dataprocessing.gen_2d_map_and_samples_pytorch import gen_2d_samples
from glob import glob
import dataprocessing.config_loader as cfg_loader
import numpy as np
import os
import shutil
from gen_complex_robots import gen_robots
import termcolor as tc

from utils.util import seed_everything, dump_config

def _dump_config(cfg):
    dump_config(cfg.out_data_dir, 'config.yaml', cfg)

def main():

    cfg = cfg_loader.get_config()
    os.makedirs(cfg.data_dir, exist_ok=True)
    print('Copy data dir:', cfg.data_dir)
    print('Env:', cfg.input_data_glob)
    print('Out data dir: {}'.format(cfg.out_data_dir))

    if cfg.data_dir != cfg.out_data_dir:
        shutil.copytree(cfg.data_dir, cfg.out_data_dir, dirs_exist_ok=True)

    _dump_config(cfg)
    
    if cfg.fix_seed:
        seed_everything(1)
    if cfg.gen_2d:
        gen_robots(cfg.out_data_dir, cfg.robot, cfg.line_length, cfg.robot_samples, 'robot', dim=2, direction=cfg.robot_direction)
        gen_robots(cfg.out_data_dir, cfg.robot, cfg.line_length, cfg.robot_samples_cls, 'robot_for_cls', dim=2, direction=cfg.robot_direction)
        print('==> Generating 2d samples.')
        gen_2d_samples(cfg.out_data_dir, cfg.num_samples, cfg.num_dim, cfg.num_obs, cfg.env_width, cfg.env_height, cfg)
        _dump_config(cfg)
        return
    
    print('Finding raw files for preprocessing.')
    # paths = glob( "./"+cfg.out_data_dir + cfg.input_data_glob)
    if cfg.input_data_glob.startswith("/"):
        cfg.input_data_glob = cfg.input_data_glob[1:]
    paths = glob(os.path.join(".", cfg.out_data_dir, cfg.input_data_glob))
    print("Processed paths:", paths)
    paths = sorted(paths)

    print('Start scaling.')
    bboxs = []
    ori_bboxs = []
    for path in paths:
        bbox, ori_bbox = to_off(path, cfg.task_name, cfg.no_scale)
        bboxs.append(bbox)
        ori_bboxs.append(ori_bbox)
    
    bbox = bboxs[0]
    if cfg.task_name == 'arm':
        bbox = np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
    elif cfg.task_name == 'cabinet':
        bbox = np.array([[-1.2, -1.2, -1.2], [1.2, 1.2, 1.2]])
    
    ori_bbox = ori_bboxs[0]
    if cfg.filter_outer:
        ref_obj_path = glob(os.path.join('.', cfg.out_data_dir, cfg.ref_obj))[0]
        to_off(ref_obj_path, cfg.task_name, cfg.no_scale, bbox_spec=ori_bbox)
    
    if cfg.num_dim > 3:
        bbox = np.concatenate((bbox, np.tile(np.array([[-0.5], [0.5]]), (1, cfg.num_dim-3))), axis=1)
    cfg.bbox = bbox
    cfg.ori_bbox = ori_bbox
    
    print(tc.colored('bbox: {}'.format(bbox), 'red', attrs=['bold']))
    print('original bbox', ori_bbox)

    if cfg.robot is not None:
        robot_kwargs = {}
        if cfg.scale_robot:
            robot_kwargs['bbox_length'] = ori_bbox[1] - ori_bbox[0]

        for path in paths:
            gen_robots(os.path.dirname(path), cfg.robot, cfg.line_length, cfg.robot_samples, 'robot', direction=cfg.robot_direction, robot_path=cfg.robot_path, **robot_kwargs)

        for path in paths:
            gen_robots(os.path.dirname(path), cfg.robot, cfg.line_length, cfg.robot_samples_cls, 'robot_for_cls', direction=cfg.robot_direction, robot_path=cfg.robot_path, **robot_kwargs)

        if cfg.robot_sparse_path is not None:
            for path in paths:
                gen_robots(os.path.dirname(path), cfg.robot, cfg.line_length, cfg.robot_samples_cls, 'robot_sparse', direction=cfg.robot_direction, robot_path=cfg.robot_sparse_path, **robot_kwargs)

    for path in paths:
        print('Start speed sampling.')
        sample_speed(path, cfg.task_name, cfg.num_samples, cfg.num_dim, cfg)

    _dump_config(cfg)

if __name__ == '__main__':
    main()
