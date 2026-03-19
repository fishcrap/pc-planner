import os
os.environ["MKL_NUM_THREADS"] = "4"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"
import sys
sys.path.append('.')

from obj_env_udf.models.model_gen_udf import ObjectGenUDF
import os
import glob
import torch
from utils.util import get_logger
from vis.visualize import visual_dists
from dataprocessing.gen_2d_map_and_samples_pytorch import visual_speed as visual_dists_SE2, robot_transform as robot_transform_SE2
import time
import math
import bvh_distance_queries
from dataprocessing.speed_sampling_gpu_refactor import bvh_query, robot_transform
import numpy as np
import igl
from utils.util import read_hdf5
from tqdm import tqdm
import open3d as o3d
from config_parser.base_config_parser import genudf_eval_parse_args
from dataprocessing.speed_sampling_gpu_refactor import get_dist_fn

def _init(args):
    global logger
    logger = get_logger()
    global output_dir
    output = 'output'
    if len(args.exp_name.split('/')) > 1:
        exp_dir = args.exp_name
    else:
        exp_dir = os.path.join(output, args.exp_name)
    args.exp_dir = exp_dir


def env_prepare(data_dir):
    env = glob.glob(os.path.join(data_dir, '*.off'))[0]

    device = 'cuda'
    v, f = igl.read_triangle_mesh(env)

    v = torch.tensor(v).float().to(device)
    f = torch.tensor(f).long().to(device)
    triangles = v[f].unsqueeze(0)
    
    return triangles


def main():
    
    args = genudf_eval_parse_args()
    
    _init(args)

    epoch = '{:0>5d}'.format(args.epoch)
    
    if args.ckpt_path is not None:
        ckpt_path = args.ckpt_path
    else:
        ckpt_path = glob.glob(os.path.join(args.exp_dir, 'ckpts', 'Model_Epoch_{}_*.pt'.format(epoch)))[0]
    print('ckpt_path:', ckpt_path)
    triangles = env_prepare(args.test_data_dir)
    get_dist_gen_udf = get_dist_fn('obj_gen_udf', args.robot_name, args.test_data_dir, args.dof, triangles, args.device, use_coll_robot=True, rotate_axis=args.rotate_axis, in_dim=args.dim, h_dim=args.h_dim, ckpt_path=ckpt_path, sim=args.sim)
    get_dist_gen_udf_1 = get_dist_fn('obj_gen_udf', args.robot_name, args.test_data_dir, args.dof, triangles, args.device, use_coll_robot=True, rotate_axis=args.rotate_axis, in_dim=args.dim, h_dim=args.h_dim, ckpt_path=ckpt_path, sim=args.sim)
    get_dist_bvh = get_dist_fn('acc', args.robot_name, args.test_data_dir, args.dof, triangles, args.device, use_coll_robot=True, rotate_axis=args.rotate_axis, in_dim=args.dim, h_dim=args.h_dim, ckpt_path=ckpt_path, sim=args.sim)

    if args.load_test:
        states, dists = read_hdf5(args.test_data_dir, data_names=['goal_points', 'goal_dists'])
        states = torch.from_numpy(states).float().cuda()
    else:
        raise NotImplementedError
    
    fix_dim, fix_dim_v = None, None
    try:
        fix_dim, fix_dim_v = read_hdf5(args.test_data_dir, data_names=['fix_dim', 'fix_dim_v'])
        print('fix_dim:', fix_dim)
        print('fix_dim_v:', fix_dim_v)
    except Exception:
        print('no fix dim')
    def add_dim_val(x, dim, val):
        """
        args:
            x: [..., dof]
            dim: int
            v: float
        return:
            x: [..., dof+1]
        """
        v = torch.full((*x.shape[:-1], 1), val, dtype=x.dtype).to(x.device)
        x = torch.cat((x[..., :dim], v, x[..., dim:]), dim=-1)
        return x
    if fix_dim is not None and fix_dim_v is not None:
        states = add_dim_val(states, fix_dim[0], fix_dim_v[0])

    batch_size = 10 ** 5
    if states.shape[0] > batch_size:
        bvh_min_d = []
        for bid in range(0, math.ceil(states.shape[0] / batch_size)):
            dist = get_dist_bvh(states[bid*batch_size:(bid+1)*batch_size])
            bvh_min_d.append(dist)
        bvh_min_d = torch.cat(bvh_min_d, dim=0)  # (n,)
    else:
        bvh_min_d = get_dist_bvh(states)

    bvh_min_d = bvh_min_d.squeeze()
    bvh_min_d = bvh_min_d.cpu().numpy()
    
    pred_min_d = get_dist_gen_udf(states)
    pred_min_d = pred_min_d.squeeze()  
    pred_min_d = pred_min_d.cpu().numpy()  
    pred_min_d_1 = get_dist_gen_udf_1(states)
    pred_min_d_1 = pred_min_d_1.squeeze()
    pred_min_d_1 = pred_min_d_1.cpu().numpy()
    states = states.cpu().numpy()
    
    print(dists)
    print(pred_min_d)
    mean_error = np.mean(np.abs(pred_min_d-dists))
    print('pred dists max', pred_min_d.max())
    print('pred dists min', pred_min_d.min())
    print('gt dists max', dists.max())
    print('gt dists min', dists.min())
    print('mean error', mean_error)

    mean_error = np.mean(np.abs(bvh_min_d-pred_min_d))
    print('bvh dists max', bvh_min_d.max())
    print('bvh dists min', bvh_min_d.min())
    print('bvh pred mean error', mean_error)

    print(f'visualize...')
    if args.load_test:
        args.udf_save_dir = os.path.join(args.exp_dir, 'udf', f'test_{args.test_data_dir}', f'Epoch_{epoch}')
    os.makedirs(args.udf_save_dir, exist_ok=True)

    visual_dists(args.udf_save_dir, states, pred_min_d, 'udf', [dists.min().item(), dists.max().item()])
    visual_dists(args.udf_save_dir, states, dists, 'gt')
    # file_path = os.path.join(args.mesh_save_dir, 'sdf_mesh.ply')
    # implicit2mesh(model, None, file_path, 512, args.device, translate=[0.5, 0.5, 0.5])

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception(f'main exception: {str(e)}')
