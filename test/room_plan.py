import sys
sys.path.append('.')
from models import model_3d_room_refactor as md

import torch
import os 
import numpy as np
import math
import torch
from torch import Tensor
from torch.autograd import Variable

from timeit import default_timer as timer
import math

from utils.util import get_logger
import os
import time
import glob
from utils.util import read_hdf5, dump_config
import shapely
from config_parser.base_config_parser import test_parse_args

from utils.plan_utils import calc_path_length, SE2_render, save_3d, visual_eik_time, visual_speed, get_bbox, project_to_2d, get_ori_bbox
from room_succ import get_check_coll_fn, check_path_collision, get_check_coll_fn_for_adaptive
from dataprocessing.gen_2d_map_and_samples_pytorch import  robot_transform as robot_transform_SE2, get_dist_fn as get_dist_fn_SE2, check_collision as check_collision_SE2
from dataprocessing.speed_sampling_gpu_refactor import robot_transform as robot_transform_SE3, check_collision as check_collision_SE3, get_dist_fn, arm_transform, preprepare_arm
from room_succ import adaptive_planning

def _init(args):
    global output_dir
    output = 'output'
    if len(args.exp_name.split('/')) > 1:
        exp_dir = args.exp_name
    else:
        exp_dir = os.path.join(output, args.exp_name)
    args.exp_dir = exp_dir
    args.log_dir = os.path.join(exp_dir, 'logs')
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, 'test-{}.log'.format(time.strftime("%Y%m%d_%H%M%S", time.localtime())))
    
    global logger
    logger = get_logger(log_path)
    
def main():
    args = test_parse_args()
    _init(args)

    womodel = md.Model(args.exp_dir, args.data_dir, args.dof, [0,0], logger=logger, device=args.device, args=args)

    epoch = '{:0>5d}'.format(args.epoch)
    if args.no_load_ckpt:
        womodel.init_blank_network()
    else:
        model_path = glob.glob(os.path.join(args.exp_dir, 'ckpts', 'Model_Epoch_{}_*.pt'.format(epoch)))[0]
        womodel.load(model_path, test=True)

    if args.SE2:
        if args.dof == 2:
            start_goal = np.array([[-0.4, -0.4, 0.3, 0.2]])
        else:
            start_goal = np.array([[0.0, 0.0, 0.0, 0.2, -0.4, 0.25]])
    else:
        if args.dof == 3:
            start_goal = np.array([[-0.25, -0.2, 0.3, 0.05, -0.3, 0.0]])
        elif args.dof == 4:
            start_goal = np.array([[-0.25, -0.2, 0.3, 0, 0.25, -0.2, 0.3, 0]])
        else:
            start_goal = np.array([[-0.3, -0.2, 0.3, 0.25, 0, 0, 0.4, -0.2, 0.3, 0.25, 0, 0]]) # arm 6dof
            # start_goal = np.array([[0.404, -0.324, -0.254, -0.349, -0.282, -0.219, 0.402, -0.444, -0.253, -0.153, -0.233, -0.127]]) # ur5

    XP=start_goal
    XP = Variable(Tensor(XP)).to(args.device)

    dis=torch.norm(XP[:,args.dof:]-XP[:,:args.dof])
    start = timer()

    point0=[]
    point1=[]

    point0.append(XP[:,:args.dof])
    point1.append(XP[:,args.dof:])

    if args.vis_eik_time:
        travel_times = []
        start_speeds = []
        goal_speeds = []

    iters = 0
    while dis > args.dist_tol:
        gradient = womodel.Gradient(XP.clone())
        print(f"gradient: {gradient.detach().cpu().numpy().flatten()}")
        
        if args.vis_eik_time:
            if args.no_enc_env:
                T = womodel.TravelTimes(XP.clone())
                travel_times.append(T.item())
                start_speeds.append(torch.linalg.norm(gradient[:, :args.dof], dim=1).item())
                goal_speeds.append(torch.linalg.norm(gradient[:, args.dof:], dim=1).item())
            else:
                raise NotImplementedError
        XP = XP + args.step / args.vmax * gradient
        dis=torch.norm(XP[:,args.dof:]-XP[:,:args.dof])
        point0.append(XP[:,:args.dof])
        point1.append(XP[:,args.dof:])
        iters += 1
        if(iters > args.max_iters):
            break

    end = timer()

    print("time",end-start)
    point1.reverse()
    point=point0+point1
    points = torch.cat(point).detach()
    
    fix_dim, fix_dim_v = None, None
    dof_wo_fix_dim = args.dof
    try:
        fix_dim, fix_dim_v = read_hdf5(args.data_dir, data_names=['fix_dim', 'fix_dim_v'])
        print('fix_dim:', fix_dim)
        print('fix_dim_v:', fix_dim_v)
        fix_dim = fix_dim[0]
        fix_dim_v = fix_dim_v[0]
        args.dof += 1
    except Exception:
        print('no fix dim')
    
    if fix_dim is not None:
        # TODO: for more dims
        points_fix = torch.full((points.shape[0], 1), fix_dim_v).to(args.device)
        points = torch.cat((points[:, :fix_dim], points_fix, points[:, fix_dim:]), dim=1)

    torch.set_printoptions(precision=4, sci_mode=False, linewidth=200)
    print(torch.cat((torch.cat(point0), torch.cat(point1[::-1])), dim=1))
    torch.set_printoptions(profile='default')
    
    bbox = get_bbox(args.data_dir)

    #perform adaptive planning
    if args.SE2:
        obstacles, env_w, env_h = read_hdf5(args.data_dir, 'env', ['obstacles', 'env_w', 'env_h'])
        half_env_w = env_w / 2.0
        half_env_h = env_h / 2.0
        origin = (-half_env_w, -half_env_h)
        boundary = shapely.geometry.Polygon([origin, (origin[0] + env_w, origin[1]), (origin[0] + env_w, origin[1] + env_h), (origin[0], origin[1] + env_h)])
        env_max_len = np.sqrt(env_w*2 + env_h*2)
    else:
        env_scaled = glob.glob(os.path.join(args.data_dir, '*.off'))[0]
        if bbox is not None:
            env_max_len = np.sqrt(np.sum((bbox[1] - bbox[0])**2))
        else:
            env_max_len = np.sqrt(3) # TODO: should be scaled
    
    robot_dir = args.data_dir if args.coll_robot_dir is None else args.coll_robot_dir
    check_coll_fn = get_check_coll_fn(args.data_dir, robot_dir, args.dof, args.SE2, args.offset, args.solid, args.robot_name, args.rotate_axis, args.dist_fn_type, args.in_dim, args.h_dim, args.ckpt_path, device=args.device, sim=args.sim)

    start_goal_coll = ~check_coll_fn(torch.stack((points[0,:], points[-1,:]), dim=0))

    print('start or goal is coll: {}'.format(start_goal_coll))
    
    if args.adaptive_planning:
        check_coll_fn_for_adp = get_check_coll_fn_for_adaptive(args.data_dir, robot_dir, args.dof, args.SE2, args.offset, args.solid, args.robot_name, args.rotate_axis, args.dist_fn_type, args.in_dim, args.h_dim, args.ckpt_path, sim=args.sim)
        new_path, is_succ= adaptive_planning(dof_wo_fix_dim, args.vmax, args.spacing, points, womodel, check_coll_fn_for_adp, env_max_len, fix_dim, fix_dim_v)
    
        if is_succ:
            print("adaptive planning success!")
            points = new_path
            print(points)

    is_coll = check_path_collision(points, check_coll_fn, env_max_len, args.spacing)

    path_length = calc_path_length(points).detach().cpu().numpy()
    
    if not args.point and args.robot_name != 'arm':
        robot = np.load(os.path.join(args.data_dir, 'robot.npy'))
        robot = torch.from_numpy(robot).float().to(args.device)
    
    if args.scale_env:
        ori_bbox = get_ori_bbox(args.data_dir)
        if ori_bbox is not None:
            scale = ori_bbox[1] - ori_bbox[0]
            scale = torch.from_numpy(scale).float().to(args.device)
            robot = robot * scale
            if args.SE2:
                points[:, :2] = points[:, :2] * scale
            else:
                points[:, :3] = points[:, :3] * scale
            scale = scale.cpu().numpy()
        else:
            from termcolor import colored
            print(colored("##### no ori bbox #####", 'red', attrs=['bold']))
    else:
        scale = None
    
    if not args.point:
        if args.SE2:
            robot_transform = lambda x: robot_transform_SE2(x, args.dof, robot)
        else:
            if args.robot_name == 'arm':
                import pytorch_kinematics as pk
                chain, v_list = preprepare_arm(args.data_dir, sim=args.sim)
                robot_transform = lambda x: arm_transform(x, chain, np.pi * 2 * 0.9, v_list, sim=args.sim)
            else:
                robot_transform = lambda x: robot_transform_SE3(x, args.dof, robot, np.pi * 2, args.rotate_axis)

    if args.point:
        xyz = points.to('cpu').data.numpy()
        centers = None
    else:
        coords = robot_transform(points)

        xyz = coords.to('cpu').data.numpy()
        
        centers = points[:, :3].to('cpu').data.numpy()

    print(points)
    pos = start_goal

    pos_enc = '_'.join(list(map(lambda x: f"{x:.3f}", pos.flatten()))) + f'_step_{args.step}' + f'_path_length_{path_length:.3f}'
    pos_enc += '_fail' if is_coll else '_succ'
    if args.scale_env and ori_bbox is not None:
        pos_enc += '_scale_back'
    vis_dir = os.path.join(args.exp_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    
    args.track_dir = os.path.join(vis_dir, "path")
    os.makedirs(args.track_dir, exist_ok=True)
    np.savetxt(os.path.join(args.track_dir, f'path_{pos_enc}.txt'), points.detach().cpu().numpy(), fmt='%.4f', delimiter=' ')
    track_path = os.path.join(args.track_dir, f'path_{pos_enc}.npy')
    np.save(track_path, points.detach().cpu().numpy())
    print('==> Track saved to {}'.format(track_path))

    if args.SE2 or fix_dim is not None:
        fig_vis_dir = os.path.join(vis_dir, "figs")
        os.makedirs(fig_vis_dir, exist_ok=True)
        video_vis_dir = os.path.join(vis_dir, "videos")
        if args.vis_track:
            os.makedirs(video_vis_dir, exist_ok=True)
    
    if not args.SE2:
        m3d_vis_dir = os.path.join(vis_dir, "3d")
        os.makedirs(m3d_vis_dir, exist_ok=True)

    print(xyz.shape)
    
    print('==> Path length: {:.3f}'.format(path_length.item()))

    print("Try to display!")
    
    if args.vis_eik_time:
        time_vis_dir = os.path.join(vis_dir, "eik_times")
        os.makedirs(time_vis_dir, exist_ok=True)
        visual_eik_time(time_vis_dir, travel_times, epoch, pos_enc)
        speeds = start_speeds + goal_speeds[::-1]
        speed_vis_dir = os.path.join(vis_dir, "speeds")
        os.makedirs(speed_vis_dir, exist_ok=True)
        visual_speed(speed_vis_dir, speeds, epoch, pos_enc)

    if args.SE2:
        SE2_render(obstacles, boundary, xyz, args.point, args.vis_track, epoch, pos_enc, fig_vis_dir, video_vis_dir, args.vis_frames, points.detach().cpu().numpy())
    else:
        save_3d(xyz, centers, env_scaled, args.point, m3d_vis_dir, epoch, pos_enc, scale)
        if fix_dim is not None:
            project_to_2d(xyz, env_scaled, fix_dim_v, args.point, args.vis_track, epoch, pos_enc, fig_vis_dir, video_vis_dir, args.vis_frames, points.detach().cpu().numpy(), scale)

    cfg_dir = os.path.join(vis_dir, 'cfg', pos_enc)

    del args.exp_dir
    del args.exp_name
    del args.log_dir
    del args.vis_eik_time
    del args.vis_track
    del args.vis_frames
    del args.track_dir
    if fix_dim is not None:
        args.dof = dof_wo_fix_dim
    os.makedirs(cfg_dir, exist_ok=True)
    dump_config(cfg_dir, 'plan_cfg', args)

    print('==> Collision: {}!'.format('yes' if is_coll else 'no'))

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception(f'main exception: {str(e)}')






