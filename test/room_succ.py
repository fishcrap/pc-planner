import sys
sys.path.append('.')
import os
os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from config_parser.base_config_parser import succ_parse_args
from dataprocessing.gen_2d_map_and_samples_pytorch import sample_and_filter_2d_points, check_collision as check_collision_SE2, robot_transform as robot_transform_SE2, get_dist_fn as get_dist_fn_SE2
from dataprocessing.speed_sampling_gpu_refactor import check_collision as check_collision_SE3, get_dist_fn, sample_and_filter_points, robot_transform as robot_transform_SE3, preprepare_arm, arm_transform
from utils.plan_utils import SE2_render, visual_eik_time, calc_path_length, visual_speed, save_3d, add_dim_val, add_dim_val_np, get_bbox, project_to_2d, remove_dim_val
from utils.util import read_hdf5, write_hdf5, get_logger, dump_config
import torch
import numpy as np
import matplotlib.path as mplPath
import os
import glob
from models import model_3d_room_refactor as md
import time
from tqdm import tqdm
import shapely
import igl
import kaolin as kal
import random

def adaptive_planning(dof, vmax, spacing, coll_track, tester, check_coll_fn, env_max_len, fix_dim, fix_dim_v):
    where_d = check_coll_fn(coll_track)
    coll_track_return = coll_track.clone()
    if fix_dim is not None:
        coll_track = remove_dim_val(coll_track, fix_dim)

    is_success = False
    if torch.any(~where_d):
        indices = torch.nonzero(~where_d, as_tuple=False)#
        start_check = torch.any(indices == 0 ).item()
        goal_check = torch.any(indices == coll_track.shape[0]-1).item()
        min_index = indices.min(dim=0)[0]
        max_index = indices.max(dim=0)[0]
        random_bool = random.choice([True, False])
        if random_bool:
            selected_index =torch.tensor([0]) if min_index.item()==0 else torch.randint(low=0, high=min_index.item(), size=(1,), device='cuda').long() #[0, min_index-1]
            selected_point = coll_track[selected_index] #[0, min_index-1]
            
        else:
            upper_bound = coll_track.shape[0]-1
            selected_index = torch.tensor([upper_bound]) if max_index.item() == upper_bound else torch.randint(low=max_index.item()+1, high=upper_bound+1, size=(1,),device='cuda').long()
            selected_point = coll_track[selected_index]
    else:
        selected_point = random.choice([coll_track[0], coll_track[-1]])
        selected_point = selected_point.unsqueeze(0)

    jump_iter = 0
    start_point = coll_track[0]
    goal_point  = coll_track[-1]
    jump_points_n = [5, 10, 15] #progressive jump points
    while (jump_iter<3):
        direction = torch.rand((jump_points_n[jump_iter], selected_point.shape[1]), dtype=torch.float32, device='cuda') 
        raidus = torch.rand((jump_points_n[jump_iter], 1), dtype=torch.float32, device='cuda') *(0.05+(0.1-0.05)/2 * jump_iter)#0.05-0.1
        jump_points = selected_point.repeat(jump_points_n[jump_iter],1) + torch.nn.functional.normalize(direction, dim=1) * raidus #shape(jump_points_n, 3)
        
        start2jump_p= torch.cat([start_point.repeat(jump_points_n[jump_iter], 1), jump_points], dim=1)
        jump2goal_p = torch.cat([jump_points, goal_point.repeat(jump_points_n[jump_iter], 1)], dim=1)

        time_start = time.time()
        start2jump_path_mask, start2jump_path, start2jump_times = check_path_valid(dof, vmax, spacing, start2jump_p, tester, check_coll_fn, env_max_len, fix_dim, fix_dim_v)
        jump2goal_path_mask, jump2start_path, jump2goal_times = check_path_valid(dof, vmax, spacing, jump2goal_p, tester, check_coll_fn, env_max_len, fix_dim, fix_dim_v)
        time_end = time.time()
        jump_point_valid = start2jump_path_mask & jump2goal_path_mask
        if torch.any(jump_point_valid):
            vaild_id = torch.nonzero(jump_point_valid, as_tuple=False) #
            sum_time = start2jump_times + jump2goal_times
            sum_time_vaild = sum_time[vaild_id]
            min_time_id = vaild_id[torch.argmin(sum_time_vaild)]

            start2jump_vaild_path = start2jump_path[min_time_id]
            jump2start_vaild_path = jump2start_path[min_time_id]
            start2goal_path = torch.cat([start2jump_vaild_path, jump2start_vaild_path], dim=1).squeeze(0)
            coll_track_return = start2goal_path
            is_success = True
            break
        else:
            jump_iter +=1
    
    return coll_track_return, is_success

def check_path_valid(dof, vmax, spacing, points, tester, check_coll_fn, env_max_len, fix_dim, fix_dim_v):
    XP = points.clone()
    points_mask = torch.ones(points.shape[0], dtype=torch.int32, device=points.device)    
    dist = torch.linalg.norm(XP[:,dof:]-XP[:,:dof], dim=1)

    start_points = []
    goal_points = []

    start_points.append(XP[:, :dof].detach().cpu().numpy())
    goal_points.append(XP[:, dof:].detach().cpu().numpy())
    start = time.time()
    with torch.no_grad():
        T = tester.TravelTimes(XP)
        eik_times = T

    iters = torch.zeros(XP.shape[0], dtype=torch.int32, device=XP.device)

    not_complete = dist > 0.06
    
    while torch.any(not_complete):
        XP_not_compl = XP[not_complete]
        gradient = tester.Gradient(XP_not_compl)
        XP_not_compl = XP_not_compl + 0.03 / vmax * gradient

        XP[not_complete] = XP_not_compl
        
        dist = torch.linalg.norm(XP[:,dof:]-XP[:,:dof], dim=1)
        start_points.append(XP[:, :dof].detach().cpu().numpy())
        goal_points.append(XP[:, dof:].detach().cpu().numpy())
        
        iters += not_complete.int() # only add the iters for the points that are not complete, record the half of the track length
        
        exceed_limit = iters > 500
        if torch.any(exceed_limit):
            break
        not_complete &= dist > 0.06 # update the not_complete points

    goal_points.reverse()
    track_points = start_points + goal_points
    track_points = np.stack(track_points, axis=1) # (n, 2*(m+1), dof)  

    print("net_time:", time.time()-start)  
    if fix_dim is not None:
        track_points = add_dim_val_np(track_points, fix_dim, fix_dim_v)

    track_points_list= []
    n, point_size, dof = track_points.shape
    check_points = torch.from_numpy(track_points).cuda()

    def check_path_collision_batch(points, check_coll_fn, env_max_len, spacing):
        """
        args:
            points: [batch, n, dof]
        return: 
            collision: [batch, n]
        """

        dist = torch.linalg.norm(points[:, 1:] - points[:,:-1], dim=1)
        sub_points = torch.linspace(0, 1, int(torch.minimum(torch.max(dist), torch.tensor([env_max_len]).to(dist.device)) / spacing) + 1, device=points.device) # (m,)
        t = sub_points.repeat(points.shape[0], points.shape[1] - 1, 1).unsqueeze(-1) # (B, n-1, m, 1)
        p1 = points[:, :-1].unsqueeze(2) # (B, n-1, 1, dof)
        p2 = points[:, 1:].unsqueeze(2) # (B, n-1, 1, dof)
        check_points = p1 + t * (p2 - p1) # (B,n-1, m, dof)
        check_points = check_points.reshape(-1, points.shape[-1]) # (B*(n-1)*m, dof)

        no_collision = check_coll_fn(check_points) # (B*(n-1)*m)
        no_collision = no_collision.reshape(points.shape[0], points.shape[1] - 1, -1) # (B, n-1, m)
        return no_collision

    coll = check_path_collision_batch(check_points, check_coll_fn, env_max_len, spacing)
    iter_time = time.time()
    no_coll_mask = torch.any(torch.any(coll, dim=1), dim=1)
    end_time = time.time()     
    return no_coll_mask, check_points, eik_times


def filter_points_outside_obstacles_2d(points, obstacles):
    """
    points: [numsamples, dof]
    obstacles: [numobs, numverts, 2]
    """
    out_obs = np.ones(points.shape[0], dtype=bool)
    for i in range(len(obstacles) - 1): # the final one is the boundary, not need to check
        poly_path = mplPath.Path(obstacles[i].transpose(1,0))
        in_poly = poly_path.contains_points(points.detach().cpu().numpy())
        out_obs &= ~in_poly
        
    return out_obs

def filter_points_outside_obstacles_3d(points, v, f):
    """
    points: [numsamples, dof]
    """
    return kal.ops.mesh.check_sign(v, f, points.unsqueeze(0)).squeeze()

def sample_test_points(numsamples, data_dir, check_coll_fn, sample_fn, filter_fn, pseudu_dim, fix_dim=None, fix_dim_v=None):
    remain_num = numsamples
    total_num = 0
    i = 0
    
    points_list = []
    while remain_num > 0:
        
        start_points, goal_points = sample_fn(numsamples)
        if(start_points.shape[0] <= 1):
            print('start points generate size is too small!')
            continue
        
        print('{} points are generated'.format(start_points.shape[0]))
        
        if fix_dim is not None:
            start_points = add_dim_val(start_points, fix_dim, fix_dim_v)
            goal_points = add_dim_val(goal_points, fix_dim, fix_dim_v)
        
        if filter_fn:
            out_obs = filter_fn(start_points[:, :pseudu_dim])
            out_obs &= filter_fn(goal_points[:, :pseudu_dim])
            start_points = start_points[out_obs]
            goal_points = goal_points[out_obs]
        
        print(f'>>> then {start_points.shape[0]} points are in the free space')
        
        no_collision = check_coll_fn(start_points)
        no_collision &= check_coll_fn(goal_points)
        start_points = start_points[no_collision]
        goal_points = goal_points[no_collision]
        
        print(f'>>> finally, {start_points.shape[0]} points are in the free space and no collision')
        
        if fix_dim is not None:
            start_points = remove_dim_val(start_points, fix_dim)
            goal_points = remove_dim_val(goal_points, fix_dim)
        
        points = torch.cat((start_points, goal_points), dim=1)
    
        points_list.append(points.detach().cpu().numpy())
        
        remain_num -= points.shape[0]
        total_num += points.shape[0]
        print(f'>>> iter {i}, processed {points.shape[0]} points, remain {max(remain_num, 0)} points')
        if total_num > numsamples:
            break
        i += 1
    
    points = np.concatenate(points_list, axis=0)[:numsamples]
    print(f'>>> total {points.shape[0]} points are generated')
    
    write_hdf5(data_dir, 'points', points, f'test_points_{points.shape[0]}')
    print(f'test points saved to {os.path.join(data_dir, f"test_points_{points.shape[0]}.h5")}')

    return points

def check_path_collision(points, check_coll_fn, env_max_len, spacing):
    """
    args:
        points: [n, dof]
    return:
        True if collision
        False if no collision
    """
    dist = torch.linalg.norm(points[1:] - points[:-1], dim=1)
    sub_points = torch.linspace(0, 1, int(torch.minimum(torch.max(dist), torch.tensor([env_max_len]).to(dist.device)) / spacing) + 1, device=points.device) # (m,)
    t = sub_points.repeat(points.shape[0] - 1, 1).unsqueeze(2) # (n-1, m, 1)
    p1 = points[:-1].unsqueeze(1) # (n-1, 1, dof)
    p2 = points[1:].unsqueeze(1) # (n-1, 1, dof)
    check_points = p1 + t * (p2 - p1) # (n-1, m, dof)
    check_points = check_points.reshape(-1, points.shape[1]) # ((n-1)*m, dof)

    no_collision = check_coll_fn(check_points)
    return torch.any(~no_collision)
    
def _init(args):
    global output_dir
    output = 'output'
    if len(args.exp_name.split('/')) > 1:
        exp_dir = args.exp_name
    else:
        exp_dir = os.path.join(output, args.exp_name)
    args.exp_dir = exp_dir
    
    args.sr_dir = os.path.join(args.exp_dir, 'succ_rate')
    os.makedirs(args.sr_dir, exist_ok=True)
    
    global logger
    logger = get_logger()
    
def check_and_find_collision_points(args, points, tester, check_coll_fn, env_max_len):
    """
    points: [n, 2*dof]
    """
    coll_points = []
    coll_tracks = []
    coll_points_id = []
    collision = 0
    for i in tqdm(range(points.shape[0]), desc='testing'):
        start_goal = points[i,:].unsqueeze(0)

        XP = start_goal

        dis=torch.norm(XP[:,args.dof:]-XP[:,:args.dof])
        start = time.time()

        start_points=[]
        goal_points=[]

        start_points.append(XP[:,:args.dof])
        goal_points.append(XP[:,args.dof:])

        iters = 0
        while dis > 0.06:
            gradient = tester.Gradient(XP.clone())

            XP = XP + 0.03 / args.vmax * gradient
            dis=torch.norm(XP[:,args.dof:]-XP[:,:args.dof])
            start_points.append(XP[:,:args.dof])
            goal_points.append(XP[:,args.dof:])
            iters += 1
            if(iters > 500):
                break

        end = time.time()

        goal_points.reverse()
        track_points = start_points + goal_points

        track_points = torch.cat(track_points)
        
        start = time.time()
        if iters > 500 or check_path_collision(track_points, check_coll_fn, env_max_len, args.spacing):
            if iters > 500:
                print(f'>>> iter {i}, the path is too long')
            coll_points.append(points[i].detach().cpu().numpy())
            coll_tracks.append(track_points.detach().cpu().numpy())
            collision += 1
        end = time.time()
        # print("check time {:.6f} s".format(end-start))
    return collision, coll_points, coll_tracks, coll_points_id
    
def check_and_find_collision_points_speedup(args, points, tester, check_coll_fn, check_coll_for_ada_fn, env_max_len, fix_dim, fix_dim_v):
    """
    points: [n, 2*dof]
    """
    XP = points.clone()
    
    dist = torch.linalg.norm(XP[:,args.dof:]-XP[:,:args.dof], dim=1)
    
    start_points = []
    goal_points = []
    
    start_points.append(XP[:, :args.dof].detach().cpu().numpy())
    goal_points.append(XP[:, args.dof:].detach().cpu().numpy())
    
    if args.vis_eik_time:
        if args.no_enc_env:
            eik_times = []
            with torch.no_grad():
                T = tester.TravelTimes(XP)
            eik_times.append(T.detach().cpu().numpy())
            start_speeds = []
            goal_speeds = []
            # T_grad = torch.zeros(XP.shape[0], args.dof * 2, device=XP.device)
            S = tester.Speeds(XP)
            start_speeds.append(S[:, 0].detach().cpu().numpy())
            goal_speeds.append(S[:, 1].detach().cpu().numpy())
        else:
            raise NotImplementedError
    
    iters = torch.zeros(XP.shape[0], dtype=torch.int32, device=XP.device)
    
    not_complete = dist > 0.06
    start = time.time()
    while torch.any(not_complete):
        XP_not_compl = XP[not_complete]
        gradient = tester.Gradient(XP_not_compl)
        XP_not_compl = XP_not_compl + 0.03 / args.vmax * gradient
        XP[not_complete] = XP_not_compl
        if args.vis_eik_time:
            with torch.no_grad():
                T_not_compl = tester.TravelTimes(XP_not_compl)
            T[not_complete] = T_not_compl
            eik_times.append(T.detach().cpu().numpy())
            S_not_compl = tester.Speeds(XP_not_compl)
            S[not_complete] = S_not_compl
            start_speeds.append(S[:, 0].detach().cpu().numpy())
            goal_speeds.append(S[:, 1].detach().cpu().numpy())

        dist = torch.linalg.norm(XP[:,args.dof:]-XP[:,:args.dof], dim=1)
        start_points.append(XP[:, :args.dof].detach().cpu().numpy())
        goal_points.append(XP[:, args.dof:].detach().cpu().numpy())
        
        iters += not_complete.int() # only add the iters for the points that are not complete, record the half of the track length
        
        exceed_limit = iters > 500
        if torch.any(exceed_limit):
            break
        not_complete &= dist > 0.06 # update the not_complete points
    
    end = time.time()
    net_time = end-start
    time_print = "network time {:.2f} s".format(end-start)
    print(time_print)
    goal_points.reverse()
    
    track_points = start_points + goal_points
    track_points = np.stack(track_points, axis=1) # (n, 2*(m+1), dof)
    
    if fix_dim is not None:
        track_points = add_dim_val_np(track_points, fix_dim, fix_dim_v)
        points = add_dim_val(points, fix_dim, fix_dim_v)
        points = add_dim_val(points, fix_dim + args.dof_with_fix_dim, fix_dim_v)
        print(points.shape)
    
    if args.vis_eik_time:
        eik_times = np.stack(eik_times, axis=1) # (n, m+1)
        coll_eik_times = []
        speeds = start_speeds + goal_speeds[::-1]
        # TODO: calc the last points' speed
        speeds = np.stack(speeds, axis=1) #(n, 2*(m+1))
        coll_speeds = []
        succ_eik_times =[]
        succ_speeds = []
    succ_points =[]
    succ_points_id = []
    succ_tracks =[]
    coll_points = []
    coll_points_id = []
    coll_tracks = []
    collision = 0
    path_lengths = []
    
    """dapative planning save"""
    coll_tracks_not_replan = []
    succ_tracks_replan = []
    perform_adaptive_points = []
    time_cost = []
    count=0
    for i in tqdm(range(points.shape[0]), desc='testing'):
        check_points_np = np.concatenate((track_points[i, :iters[i]+1, :], track_points[i, -(iters[i]+1):, :]), axis=0)  # +1 for the start point and the goal point
        check_points = torch.from_numpy(check_points_np).cuda()
        if iters[i] > 500 or check_path_collision(check_points, check_coll_fn, env_max_len, args.spacing):
            if iters[i] > 500:
                print(f'>>> iter {i}, the path is too long')
                reserve_sg=[0,-1]
                check_points = check_points[reserve_sg]
                print('check_points',check_points)
            if args.adaptive_planning:
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                starter.record()
                new_path, is_succ = adaptive_planning(args.dof, args.vmax, args.spacing, check_points, tester, check_coll_for_ada_fn, env_max_len, fix_dim, fix_dim_v)
                ender.record()
                torch.cuda.synchronize()
                time_cost.append(starter.elapsed_time(ender))
                count+=1
                print("newpath",new_path.shape)
                if new_path.shape[0]>500:
                    print(f'>>> new path is too long')
                    
                elif not check_path_collision(new_path, check_coll_fn, env_max_len, args.spacing):
                    print(f'adaptive planning success!, points id: {i}')
                    succ_points.append(points[i].detach().cpu().numpy())
                    succ_points_id.append(i)
                    succ_tracks.append(new_path.detach().cpu().numpy())
                    path_lengths.append(calc_path_length(new_path).detach().cpu().numpy())

                    #make a visual comparison
                    succ_tracks_replan.append(new_path.detach().cpu().numpy())
                    coll_tracks_not_replan.append(check_points_np)
                    perform_adaptive_points.append(points[i].detach().cpu().numpy())

                    if args.vis_eik_time:
                        succ_eik_times.append(eik_times[i, :iters[i]+1])
                        succ_speeds.append(np.concatenate((speeds[i, :iters[i]+1], speeds[i, -(iters[i]+1):]), axis=0))

                    continue
            coll_points.append(points[i].detach().cpu().numpy())
            coll_points_id.append(i)
            coll_tracks.append(check_points_np)
            collision += 1
            if args.vis_eik_time:
                coll_eik_times.append(eik_times[i, :iters[i]+1])
                coll_speeds.append(np.concatenate((speeds[i, :iters[i]+1], speeds[i, -(iters[i]+1):]), axis=0))
        else: # add succ path length
            path_lengths.append(calc_path_length(check_points).detach().cpu().numpy())

            succ_points.append(points[i].detach().cpu().numpy())
            succ_points_id.append(i)
            succ_tracks.append(check_points_np)
            if args.vis_eik_time:
                succ_eik_times.append(eik_times[i, :iters[i]+1])
                succ_speeds.append(np.concatenate((speeds[i, :iters[i]+1], speeds[i, -(iters[i]+1):]), axis=0))

        # torch.cuda.empty_cache()
    if args.adaptive_planning:
        print(f'==> adaptive planning time cost: {np.array(time_cost).sum()/count} ms')
        avg = (net_time+np.array(time_cost).sum()/1000)/points.shape[0]
        time_print = f'\n avg time cost: {avg} + {net_time/points.shape[0]+np.array(time_cost).max()/1000-avg} s'
        print(time_print)
    return_vals = [collision, coll_points, coll_tracks, path_lengths, coll_points_id, succ_points, succ_tracks, succ_points_id, time_print]
    if args.vis_eik_time:
        return_vals.append(coll_eik_times)
        return_vals.append(coll_speeds)
        return_vals.append(succ_eik_times)
        return_vals.append(succ_speeds)
    if args.vis_adaptive:
        return_vals.append(coll_tracks_not_replan)
        return_vals.append(succ_tracks_replan)
        return_vals.append(perform_adaptive_points)
    return return_vals

def get_check_coll_fn(data_dir, robot_dir, dof, SE2, offset, solid, robot_name, rotate_axis='x', dist_fn_type='acc', in_dim=3, h_dim=128, ckpt_path=None, scale=None, device='cuda', sim=False):
    if SE2:
        robot_for_coll = np.load(os.path.join(data_dir, 'robot_for_cls.npy'))
        robot_for_coll = torch.from_numpy(robot_for_coll).float().to(device)
        
        env_width, env_height, obstacles = read_hdf5(data_dir, 'env', data_names=['env_w', 'env_h', 'obstacles'])
        env_bound = shapely.geometry.Polygon([[-env_width/2, -env_height/2], [-env_width/2, env_height/2], [env_width/2, env_height/2], [env_width/2, -env_height/2]])
        obstacles = np.concatenate((obstacles, np.array(env_bound.exterior.xy)[np.newaxis,...]), axis=0) # add env boundary to obstacles
        get_dists = get_dist_fn_SE2(dist_fn_type, dof, robot_for_coll, obstacles, in_dim, h_dim, ckpt_path, device, solid)
        check_collision = lambda x: check_collision_SE2(x, get_dists, offset)

    else:
        file_path = glob.glob(os.path.join(data_dir, '*.off'))[0]
        v_np, f_np = igl.read_triangle_mesh(file_path)
        if scale is not None:
            v_np = v_np * scale
        v = torch.from_numpy(v_np).float().cuda()
        f = torch.from_numpy(f_np).long().cuda()
        triangles = v[f].unsqueeze(0)
        get_dist = get_dist_fn(dist_fn_type, robot_name, robot_dir, dof, triangles, 'cuda', use_coll_robot=True, rotate_axis=rotate_axis,
                               in_dim=in_dim, h_dim=h_dim, ckpt_path=ckpt_path, sim=sim)
        check_collision = lambda x: check_collision_SE3(get_dist, x, offset)

    return check_collision


def get_check_coll_fn_for_adaptive(data_dir, robot_dir, dof, SE2, offset, solid, robot_name, rotate_axis='x', dist_fn_type='acc', in_dim=3, h_dim=128, ckpt_path=None, scale=None, sim=False):
    if SE2:
        robot_for_coll = np.load(os.path.join(data_dir, 'robot_for_cls.npy'))
        robot_for_coll = torch.from_numpy(robot_for_coll).float().cuda()
        
        env_width, env_height, obstacles = read_hdf5(data_dir, 'env', data_names=['env_w', 'env_h', 'obstacles'])
        env_bound = shapely.geometry.Polygon([[-env_width/2, -env_height/2], [-env_width/2, env_height/2], [env_width/2, env_height/2], [env_width/2, -env_height/2]])
        obstacles = np.concatenate((obstacles, np.array(env_bound.exterior.xy)[np.newaxis,...]), axis=0) # add env boundary to obstacles
        get_dists = get_dist_fn_SE2(dist_fn_type, dof, robot_for_coll, obstacles, in_dim, h_dim, ckpt_path, 'cuda', solid)
        check_collision = lambda x: check_collision_SE2(x, get_dists, offset)

    else:
        file_path = glob.glob(os.path.join(data_dir, '*.off'))[0]
        v_np, f_np = igl.read_triangle_mesh(file_path)
        if scale is not None:
            v_np = v_np * scale
        v = torch.from_numpy(v_np).float().cuda()
        f = torch.from_numpy(f_np).long().cuda()
        triangles = v[f].unsqueeze(0)
        get_dist = get_dist_fn(dist_fn_type, robot_name, robot_dir, dof, triangles, 'cuda', use_coll_robot=True, rotate_axis=rotate_axis,
                               in_dim=in_dim, h_dim=h_dim, ckpt_path=ckpt_path, sim=sim)
        check_collision = lambda x: check_collision_SE3(get_dist, x, offset)

    return check_collision

def main():
    args = succ_parse_args()
    _init(args)
    
    bbox = get_bbox(args.data_dir)
    fix_dim, fix_dim_v = None, None
    try:
        fix_dim, fix_dim_v = read_hdf5(args.data_dir, data_names=['fix_dim', 'fix_dim_v'])
        print('fix_dim:', fix_dim)
        print('fix_dim_v:', fix_dim_v)
        fix_dim, fix_dim_v = fix_dim[0], fix_dim_v[0]
    except Exception:
        print('no fix dim')
    dof_with_fix_dim = args.dof + 1 if fix_dim is not None else args.dof
    args.dof_with_fix_dim = dof_with_fix_dim
    if args.SE2:
        if args.coll_robot_dir is not None:
            robot_for_coll = np.load(args.coll_robot_dir, 'robot_for_cls.npy')
        else:
            robot_for_coll = np.load(os.path.join(args.data_dir, 'robot_for_cls.npy'))
        robot_for_coll = torch.from_numpy(robot_for_coll).float().cuda()
        
        env_width, env_height, obstacles = read_hdf5(args.data_dir, 'env', data_names=['env_w', 'env_h', 'obstacles'])
        env_bound = shapely.geometry.Polygon([[-env_width/2, -env_height/2], [-env_width/2, env_height/2], [env_width/2, env_height/2], [env_width/2, -env_height/2]])
        obstacles = np.concatenate((obstacles, np.array(env_bound.exterior.xy)[np.newaxis,...]), axis=0) # add env boundary to obstacles
        get_dists = get_dist_fn_SE2(args.dist_fn_type, args.dof, robot_for_coll, obstacles, args.in_dim, args.h_dim, args.ckpt_path, 'cuda', args.solid)
        check_collision = lambda x: check_collision_SE2(x, get_dists, args.offset)

        get_dists_for_adaptive = get_dist_fn_SE2(args.dist_fn_type_for_adaptive, args.dof, robot_for_coll, obstacles, args.in_dim, args.h_dim, args.ckpt_path, 'cuda', args.solid)
        check_collision_for_adaptive  = lambda x: check_collision_SE2(x, get_dists_for_adaptive, args.offset)

        env_max_len = np.sqrt(env_width*2 + env_height*2)
        sample_and_filter_points_fn = lambda x: sample_and_filter_2d_points(x, args.dof, env_width, env_height, random_loc=False)
        filter_fn = lambda x: filter_points_outside_obstacles_2d(x, obstacles)
        pseudu_dim = 2
    else:
        file_path = glob.glob(os.path.join(args.data_dir, '*.off'))[0]
        v_np, f_np = igl.read_triangle_mesh(file_path)
        v = torch.from_numpy(v_np).float().cuda()
        f = torch.from_numpy(f_np).long().cuda()
        triangles = v[f].unsqueeze(0)
        
        if args.ref_data_dir is not None:
            file_path = glob.glob(os.path.join(args.ref_data_dir, '*.off'))[0]
        else:
            print('############## please specify the ref data dir if clipped ##############')
            print("############## the most recently used is datasets/gibson_clip_ref ##############") 
        mesh = kal.io.off.import_mesh(file_path)
        kal_v = mesh.vertices.unsqueeze(0).cuda()
        kal_f = mesh.faces.cuda()
        
        if args.coll_robot_dir is not None:
            robot_dir = args.coll_robot_dir
        else:
            robot_dir = args.data_dir
        get_dist = get_dist_fn(args.dist_fn_type, args.robot_name, robot_dir, dof_with_fix_dim, triangles, 'cuda', use_coll_robot=True, rotate_axis=args.rotate_axis,
                               in_dim=args.in_dim, h_dim=args.h_dim, ckpt_path=args.ckpt_path, sim=args.sim)
        check_collision = lambda x: check_collision_SE3(get_dist, x, args.offset)

        get_dists_for_adaptive = get_dist_fn(args.dist_fn_type_for_adaptive, args.robot_name, args.data_dir, dof_with_fix_dim, triangles, 'cuda', use_coll_robot=True, rotate_axis=args.rotate_axis,
                               in_dim=args.in_dim, h_dim=args.h_dim, ckpt_path=args.ckpt_path, sim=args.sim)
        check_collision_for_adaptive = lambda x: check_collision_SE3(get_dists_for_adaptive, x, args.offset)
        if bbox is not None:
            env_max_len = np.sqrt(np.sum((bbox[1] - bbox[0])**2))
        else:
            env_max_len = np.sqrt(3) # TODO: should be scaled

        # env_max_len /= 20
        sample_and_filter_points_fn = lambda x: sample_and_filter_points(x, args.dof, 'cuda', bbox=bbox)
        # filter_fn = lambda x: filter_points_outside_obstacles_3d(x, v_np, f_np)
        if args.robot_name == 'arm':
            filter_fn = None
        else:
            filter_fn = lambda x: filter_points_outside_obstacles_3d(x, kal_v, kal_f)
        pseudu_dim = 3

    if args.load_test_points:
        if args.test_points_path:
            file_name = os.path.basename(args.test_points_path)
            test_points_dir = os.path.dirname(args.test_points_path)
            points = read_hdf5(test_points_dir, file_name, data_names=['points'])[0]
        elif args.test_points_dir:
            file_paths = glob.glob(os.path.join(args.test_points_dir, 'test_points*.h5'))
            file_paths.sort()
            print(f'>>> {len(file_paths)} test points files are found')
            print(f'>>> we use the first one: {file_paths[0]}')
            file_path = file_paths[0]
            file_name = os.path.basename(file_path)
            points = read_hdf5(args.test_points_dir, file_name, data_names=['points'])[0]
        else:
            print('>>> please specify the test points path or the test points dir')
            print(f'>>> or we will use the default test_points.h5 in the {args.data_dir}')
            points = read_hdf5(args.data_dir, 'test_points', data_names=['points'])[0]
        args.test_points_n = points.shape[0]
        print(f'>>> {points.shape[0]} points are loaded')
    else:
        points = sample_test_points(args.test_points_n, args.data_dir, check_collision, sample_and_filter_points_fn, filter_fn, pseudu_dim, fix_dim, fix_dim_v)
    
    epoch = '{:0>5d}'.format(args.epoch)
    args.fail_dir = os.path.join(args.sr_dir, 'epoch_{}_points_{}_faliure_case_{}'.format(epoch, args.test_points_n, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))
    os.makedirs(args.fail_dir, exist_ok=True)
    dump_config(args.fail_dir, 'succ_config', args)
    
    tester = md.Model(args.exp_dir, args.data_dir, args.dof, [0,0], logger=logger, device='cuda', args=args)

    model_path = glob.glob(os.path.join(args.exp_dir, 'ckpts', 'Model_Epoch_{}_*.pt'.format(epoch)))[0]
    tester.load(model_path, test=True)

    points = torch.from_numpy(points).float().cuda()

    start = time.time()
    if args.more_acc_check:
        if args.vis_eik_time:
            raise NotImplementedError
        collision, coll_points, coll_tracks, coll_points_id = check_and_find_collision_points(args, points, tester, check_collision, env_max_len)
    else:
        r = check_and_find_collision_points_speedup(args, points, tester, check_collision, check_collision_for_adaptive, env_max_len, fix_dim, fix_dim_v)
        collision, coll_points, coll_tracks, path_lengths, coll_points_id, succ_points, succ_tracks, succ_points_id, time_print = r[:9]
        if args.vis_eik_time:
            coll_eik_times, coll_speeds, succ_eik_times, succ_speeds = r[9:13]
            if args.vis_adaptive:
                coll_tracks_not_replan, succ_tracks_replan, perform_adpative_points = r[13:]
        else:
            if args.vis_adaptive:
                coll_tracks_not_replan, succ_tracks_replan, perform_adpative_points = r[9:]
    succ_rate = 1 - collision / points.shape[0]
    coll_file_path = os.path.join(args.fail_dir, f'coll_points_{args.test_points_n}_succ_rate_{succ_rate*100:.2f}.txt')
    coll_file_id_path = os.path.join(args.fail_dir, f'coll_points_id.txt')
    if collision > 0:
        coll_points = np.stack(coll_points)
        np.savetxt(coll_file_path, coll_points, fmt='%.6f')
        coll_points_id = np.array(coll_points_id)
        np.savetxt(coll_file_id_path, coll_points_id, fmt='%d')
    else:
        with open(coll_file_path, 'w') as f:
            pass
    if len(succ_points)>0:
        path_lengths = np.stack(path_lengths)
        succ_points_id = np.array(succ_points_id)
        np.savetxt(os.path.join(args.fail_dir, f'path_length.txt'), path_lengths, fmt='%.6f')
        np.savetxt(os.path.join(args.fail_dir, f'succ_points_id.txt'), succ_points_id, fmt='%d')

    mean_path_length = np.mean(path_lengths)
    with open(os.path.join(args.fail_dir, f'metric_epoch_{epoch}_points_{args.test_points_n}.txt'), 'w') as f:
        f.write(f'Mean path length: \t{mean_path_length:.3f}\n')
        f.write(f'Success rate: \t{succ_rate*100:.2f}%')
        f.write(f'Time cost: \t{time_print}')
    
    end = time.time()
    print("=> total time {:.3f} s".format(end-start))
    print(f'>>> success rate: {succ_rate*100:.2f}%')
    print(f'>>> mean path length: {mean_path_length:.3f}')
    
    if args.vis_failure or args.vis_succ or args.vis_adaptive:
        if args.robot_name != 'arm':
            robot = np.load(os.path.join(args.data_dir, 'robot.npy'))
            robot = torch.from_numpy(robot).float().cuda()
        if args.SE2:
            robot_transform = lambda x: robot_transform_SE2(x, args.dof, robot)
        else:
            if args.robot_name == 'arm':
                import pytorch_kinematics as pk
                chain, v_list = preprepare_arm(args.data_dir, sim=args.sim)
                robot_transform = lambda x: arm_transform(x, chain, np.pi * 2 * 0.9, v_list, sim=args.sim)
            else:
                robot_transform = lambda x: robot_transform_SE3(x, dof_with_fix_dim, robot, np.pi * 2, args.rotate_axis)
            env_scaled = glob.glob(os.path.join(args.data_dir, '*.off'))[0]
        if args.vis_failure:
            for i in tqdm(range(len(coll_tracks)), desc='visualizing failure'):
                vis_dir = os.path.join(args.fail_dir, 'failure_case', f'{i}')
                os.makedirs(vis_dir, exist_ok=True)
                pos_enc = '_'.join(list(map(lambda x: '{:.3f}'.format(x), coll_points[i].flatten()))) 
                track_points = torch.from_numpy(coll_tracks[i]).float().cuda()
                # if fix_dim is not None:
                #     track_points = add_dim_val(track_points, fix_dim[0], fix_dim_v[0]) # TODO: for more dims
                track_points = robot_transform(track_points)
                track_points = track_points.detach().cpu().numpy()
                if args.SE2:
                    SE2_render(obstacles[:-1], env_bound, track_points, args.point, False, epoch, pos_enc, vis_dir)
                else:
                    centers = coll_tracks[i][:, :3]
                    save_3d(track_points, centers, env_scaled, args.point, vis_dir, epoch, pos_enc)
                    if fix_dim is not None:
                        project_to_2d(track_points, env_scaled, fix_dim_v, args.point, False, epoch, pos_enc, vis_dir)
                if args.vis_eik_time:
                    visual_eik_time(vis_dir, coll_eik_times[i], epoch, pos_enc)
                    visual_speed(vis_dir, coll_speeds[i], epoch, pos_enc)
        if args.vis_succ:
            n = min(args.succ_n, len(succ_tracks))
            # randomly select n succ cases
            succ_tracks = np.random.choice(succ_tracks, n, replace=False)
            for i in tqdm(range(n), desc='visualizing succ'):
                vis_dir = os.path.join(args.fail_dir, 'succ_case', f'{i}')
                os.makedirs(vis_dir, exist_ok=True)
                pos_enc = '_'.join(list(map(lambda x: '{:.3f}'.format(x), succ_points[i].flatten()))) 
                track_points = torch.from_numpy(succ_tracks[i]).float().cuda()
                track_points = robot_transform(track_points)
                track_points = track_points.detach().cpu().numpy()
                if args.SE2:
                    SE2_render(obstacles[:-1], env_bound, track_points, args.point, False, epoch, pos_enc, vis_dir)
                    # SE2_render(obstacles[:-1], env_bound, track_points, args.point, True, epoch, pos_enc, vis_dir, vis_dir)
                else:
                    save_3d(track_points, succ_tracks[i][:, :3], env_scaled, args.point, vis_dir, epoch, pos_enc)
                    if fix_dim is not None:
                        project_to_2d(track_points, env_scaled, fix_dim_v, args.point, False, epoch, pos_enc, vis_dir)
                if args.vis_eik_time:
                    visual_eik_time(vis_dir, succ_eik_times[i], epoch, pos_enc)
                    visual_speed(vis_dir, succ_speeds[i], epoch, pos_enc)
        
        if args.vis_adaptive:
            for i in tqdm(range(len(succ_tracks_replan)), desc='visualizing comparison after adaptive planning'):
                vis_dir = os.path.join(args.fail_dir, 'succ_case_replan', f'{i}')
                os.makedirs(vis_dir, exist_ok=True)
                pos_enc = '_'.join(list(map(lambda x: '{:.3f}'.format(x), perform_adpative_points[i].flatten()))) 
                succ_enc = pos_enc+'_succ' 
                succ_track_points = torch.from_numpy(succ_tracks_replan[i]).float().cuda()
                succ_track_points = robot_transform(succ_track_points)
                succ_track_points = succ_track_points.detach().cpu().numpy()

                coll_track_points = torch.from_numpy(coll_tracks_not_replan[i]).float().cuda()
                coll_track_points = robot_transform(coll_track_points)
                coll_track_points = coll_track_points.detach().cpu().numpy()

                if args.SE2:
                    SE2_render(obstacles[:-1], env_bound, succ_track_points, args.point, False, epoch, succ_enc, vis_dir)
                    SE2_render(obstacles[:-1], env_bound, coll_track_points, args.point, False, epoch, pos_enc, vis_dir)
                else:
                    save_3d(succ_track_points, succ_tracks_replan[i][:, :3], env_scaled, args.point, vis_dir, epoch, succ_enc)
                    save_3d(coll_track_points, coll_tracks_not_replan[i][:, :3], env_scaled, args.point, vis_dir, epoch, pos_enc)
                    if fix_dim is not None:
                        project_to_2d(succ_track_points, env_scaled, fix_dim_v, args.point, False, epoch, succ_enc, vis_dir)
                        project_to_2d(coll_track_points, env_scaled, fix_dim_v, args.point, False, epoch, pos_enc, vis_dir)

        print("=> total time {:.3f} s".format(time.time()-start))
        print(f'>>> success rate: {succ_rate*100:.2f}%')

    print('>>> collision points are saved to {}'.format(coll_file_path))
    
if __name__ == '__main__':
    main()