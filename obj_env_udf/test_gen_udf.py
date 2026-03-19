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
from config_parser.base_config_parser import genudf_test_parse_args

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

@torch.no_grad()
def get_dists(model, points):
    batch_size = 10 ** 7
    if points.shape[0] > batch_size:
        dists = []
        for bid in range(0, math.ceil(points.shape[0] / batch_size)):
            dist = model(points[bid * batch_size: (bid + 1) * batch_size]).squeeze()
            dists.append(dist.detach().cpu().numpy())
        dists = np.concatenate(dists, axis=0)
    else:
        dists = model(points).squeeze().detach().cpu().numpy()
    return dists

@torch.no_grad()
def infer(model, *args):
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # start = time.time()
    # starter.record()
    time_cost =0
    with torch.no_grad():
        dists = model.out(*args)
        dists = dists.squeeze()
        # dists = get_dists(model, points)
    # ender.record()
    # torch.cuda.synchronize()
    # time_cost = starter.elapsed_time(ender)
    # print('model infer time {:.6f} s'.format(time_cost / 1000.))
    # print('time cost: {:.6f} s'.format(time.time() - start))
    return dists, time_cost

def bvh_prepare(data_dir):
    env = glob.glob(os.path.join(data_dir, '*.off'))[0]

    device = 'cuda'
    v, f = igl.read_triangle_mesh(env)

    v = torch.tensor(v).float().to(device)
    f = torch.tensor(f).long().to(device)
    triangles = v[f].unsqueeze(0)
    bvh = bvh_distance_queries.BVH()
    
    return triangles, bvh
    

def test_bvh_query(bvh, triangles, points, robot_size=None):
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # starter.record()
    time_cost =0
    dists = bvh_query(bvh, triangles, points.unsqueeze(0), pointsize=robot_size)
    # ender.record()
    
    # torch.cuda.synchronize()
    # time_cost = starter.elapsed_time(ender)
    # print('bvh time {:.6f} s'.format(time_cost / 1000.))
    
    return dists, time_cost

def calc_time(model, points, data_dir, dense_points, *extra_model_args):
    total_time = 0
    TEST_ITERS = 1000
    print('points shape:', points.shape)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    model_args = (points,) + extra_model_args
    infer(model, *model_args)
    starter.record()
    for i in range(TEST_ITERS):
        dists, time_cost = infer(model, *model_args)
        total_time += time_cost
        
    ender.record()
    torch.cuda.synchronize()
    time_cost = starter.elapsed_time(ender)
    
    print('model average time cost: {:.6f} s'.format(time_cost / TEST_ITERS / 1000.))
    # print('model average time cost: {:.6f} s'.format(total_time / TEST_ITERS / 1000.))
    
    triangles, bvh = bvh_prepare(data_dir)
    points = points.reshape(-1, 3)
    points = dense_points.reshape(-1, 3)
    print('bvh points shape', points.shape)
    test_bvh_query(bvh, triangles, points)
    starter.record()
    for i in range(TEST_ITERS):
        dists, time_cost = test_bvh_query(bvh, triangles, points)
        total_time += time_cost
    
    ender.record()
    torch.cuda.synchronize()
    time_cost = starter.elapsed_time(ender)
    
    print('bvh average time cost: {:.6f} s'.format(time_cost / TEST_ITERS / 1000.))
    # print('bvh average time cost: {:.6f} s'.format(total_time / TEST_ITERS / 1000.))

def main():
    
    args = genudf_test_parse_args()
    
    _init(args)

    epoch = '{:0>5d}'.format(args.epoch)
    ckpt_path = glob.glob(os.path.join(args.exp_dir, 'ckpts', 'Model_Epoch_{}_*.pt'.format(epoch)))[0]
    ckpt = torch.load(ckpt_path, map_location=torch.device(args.device))

    model = ObjectGenUDF(in_dim=args.dim, hidden_dim=args.h_dim)
    
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    model.to(args.device)
    
    if args.load_test:
        # ! get sparse robot
        sparse_robot = torch.tensor(np.load(os.path.join(args.test_data_dir, 'robot_sparse.npy')), dtype=torch.float32).to('cuda')
        # 'obj_env_udf/processed_shapenet/piano_normalize_0.2/42911b5824d49af774cbf1cd482c6ca0/test_case_128/model_normalized.ply'
        pcd = o3d.io.read_point_cloud(args.load_test_path)
        sparse_robot = torch.tensor(np.asarray(pcd.points), dtype=torch.float32).to('cuda')
        print('sparse_robot shape', sparse_robot.shape)
        
        # ! get dense robot
        dense_robot = torch.tensor(np.load(os.path.join(args.test_data_dir, 'robot.npy')), dtype=torch.float32).to('cuda')
        print('dense robot shape', dense_robot.shape)
        
        states, dists = read_hdf5(args.test_data_dir, data_names=['goal_points', 'goal_dists'])
        states = torch.from_numpy(states).float().cuda()
        
        # ! sparse points
        robot_points = robot_transform(states, 6, sparse_robot, scale=np.pi * 2)
        
        # fix_dim, fix_dim_v = None, None
        # try:
        #     fix_dim, fix_dim_v = read_hdf5(args.test_data_dir, data_names=['fix_dim', 'fix_dim_v'])
        #     print('fix_dim:', fix_dim)
        #     print('fix_dim_v:', fix_dim_v)
        # except Exception:
        #     print('no fix dim')
        # states = add_dim_val(states, fix_dim[0], fix_dim_v[0])
        # robot_points = robot_transform(states, 4, sparse_robot, scale=np.pi * 2, rotate_axis='z')
        
        # ! dense points
        dense_robot_points_list = []
        batch_size = 1000
        for bid in range(0, math.ceil(states.shape[0] / batch_size)):
            dense_robot_points = robot_transform(states[bid * batch_size: (bid + 1) * batch_size], 6, dense_robot, scale=np.pi * 2)
            # dense_robot_points = robot_transform(states[bid * batch_size: (bid + 1) * batch_size], 4, dense_robot, scale=np.pi * 2, rotate_axis='z')
            dense_robot_points_list.append(dense_robot_points.detach().cpu())
        dense_robot_points = torch.cat(dense_robot_points_list, dim=0)
        dense_robot_points = dense_robot_points.cuda()
        states = states.cpu().numpy()
    else:
        test_idx = 3
        split = 'test'
        sparse_robot, robot_points, states, sparse_dists = read_hdf5(args.data_dir, data_names=['robot', 'robot_points', 'points', 'dists'], group_name=f'/{split}/sparse/{test_idx}')
        dense_robot, dists = read_hdf5(args.data_dir, data_names=['robot', 'dists'], group_name=f'/{split}/dense/{test_idx}')
        
        robot_points = torch.from_numpy(robot_points).float().cuda()
        sparse_robot = torch.from_numpy(sparse_robot).float().cuda()
        dense_robot = torch.from_numpy(dense_robot).float().cuda()

    print('==> test numbers: {}'.format(dists.shape[0]))
    
    if args.test_time:
        
        # triangles, bvh = bvh_prepare(args.test_data_dir)
        # points = robot_points.reshape(-1, 3)
        # print('bvh points shape', points.shape)

        # sparse_dists, _ = test_bvh_query(bvh, triangles, points, sparse_robot.shape[0])
        # sparse_dists = sparse_dists.squeeze().detach().cpu().numpy()
        
        # print('bvh dense points shape', dense_robot_points.shape)
        # dense_dists_list = []
        # batch_size = 10000
        # for bid in range(0, math.ceil(dense_robot_points.shape[0] / batch_size)):
        #     dense_dists, _ = test_bvh_query(bvh, triangles, dense_robot_points[bid * batch_size: (bid + 1) * batch_size].reshape(-1, 3).contiguous(), dense_robot.shape[0])
        #     dense_dists_list.append(dense_dists.squeeze().detach().cpu().numpy())
        
        # dense_dists = np.concatenate(dense_dists_list, axis=0)
        
        # # dense_dists, dense_time_cost = test_bvh_query(bvh, triangles, dense_points, dense_robot.shape[0])
        # # dense_dists = dense_dists.squeeze().detach().cpu().numpy()
        
        # sd_mean_error = np.mean(np.abs(sparse_dists - dense_dists))
        # print('sparse dense mean error', sd_mean_error)
        # print(f'sparse max: {sparse_dists.max()}, min: {sparse_dists.min()}')
        # print(f'dense max: {dense_dists.max()}, min: {dense_dists.min()}')

        test_size = 10000
        calc_time(model, robot_points[:test_size].contiguous(), args.test_data_dir, dense_robot_points[:test_size].contiguous(), sparse_robot, dense_robot)
        return
    
    batch_size = 10 ** 4
    if dists.shape[0] > batch_size:
        pred_min_d_list = []
        for bid in tqdm(range(0, math.ceil(dists.shape[0] / batch_size))):
            with torch.no_grad():
                pred_min_d, _ = model(robot_points[bid * batch_size: (bid + 1) * batch_size], sparse_robot, dense_robot)
                pred_min_d_list.append(pred_min_d.detach().cpu().numpy())
        pred_min_d = np.concatenate(pred_min_d_list, axis=0)
    else:
        with torch.no_grad():
            pred_min_d, _ = model(robot_points, sparse_robot, dense_robot)
            pred_min_d = pred_min_d.detach().cpu().numpy()

    pred_min_d = pred_min_d.squeeze()  
    mean_error = np.mean(np.abs(pred_min_d-dists))
    print('pred dists max', pred_min_d.max())
    print('pred dists min', pred_min_d.min())
    print('gt dists max', dists.max())
    print('gt dists min', dists.min())
    print('mean error', mean_error)
    if not args.load_test:
        sd_mean_error = np.mean(np.abs(dists - sparse_dists))
        print('sparse dense mean error', sd_mean_error)
        print('sparse pred mean error', np.mean(np.abs(sparse_dists - pred_min_d)))
    print(f'visualize...')
    if args.load_test:
        args.udf_save_dir = os.path.join(args.exp_dir, 'udf', f'test_{args.test_data_dir}', f'Epoch_{epoch}')
    else:
        args.udf_save_dir = os.path.join(args.exp_dir, 'udf', f'test_{test_idx}', f'Epoch_{epoch}')
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
