import os 
import numpy as np
from timeit import default_timer as timer
import igl
import traceback
import math
import torch
import pytorch_kinematics as pk

import bvh_distance_queries
import math
import random
import glob
from easydict import EasyDict as edict
from utils.util import write_hdf5, append_hdf5
import open3d as o3d
from vis.visualize import visual_space_labels, visual_point_speed
# import trimesh
import kaolin as kal

def bvh_query(bvh, triangles, query_points, pointsize=None, batch_size=10**7):
    """
    query_points: (b, n, 3); # TODO: b == 1, change!!!!
    return unsigned distances
    """
    # batch_size = 5 * 10**6
    # batch_size = 10**7
    batch_size = 1 * 10**7
    if query_points.shape[1] > batch_size:
        # print('query points size:', query_points.shape[1])
        distances = []
        for bid in range(math.ceil(query_points.shape[1] / batch_size)):
            # print(f'partial query: {query_points[:, bid*batch_size:(bid+1)*batch_size, :].shape}')
            partial_distances, closest_points, closest_faces, closest_bcs= bvh(triangles, query_points[:, bid*batch_size:(bid+1)*batch_size, :])
            # torch.cuda.synchronize()
            distances.append(partial_distances)
        del partial_distances
        distances = torch.cat(distances, dim=1)
    else:
        distances, closest_points, closest_faces, closest_bcs= bvh(triangles, query_points)
        # torch.cuda.synchronize()

    distances = distances.squeeze(dim=0)
    
    if pointsize:
        distances, _ = torch.min(torch.reshape(distances, (-1, pointsize)), dim=1)
    
    distances = torch.sqrt(distances)
    
    return distances

def flip_false_to_true(matrix, flip_percentage):
    # Calculate the number of False values to flip based on the flip_percentage
    total_false_count = torch.sum(~matrix).item()  # ~ flips True to False and False to True
    flip_count = int(total_false_count * flip_percentage)

    # Get the indices of False values in the matrix
    false_indices = torch.where(~matrix)[0]

    # Randomly select `flip_count` indices to flip from False to True
    # TODO: change it into pytorch-style
    selected_indices = random.sample(false_indices.tolist(), flip_count)

    # Flip the selected indices to True
    matrix[selected_indices] = True

    return matrix

def check_collision(get_dist, points, offset, prob=0, filter_outer_fn=None, bbox=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]):
    """
    get_dist: fn for getting distances
    points: (n, dof), generated robot states [positions,[rotation]]
    offset: minimum distance between the robot and the room
    bbox: bounding box

    return a boolean tensor of shape (n, ) which indicates whether the robot collides with the room
    True: no collision
    False: collision
    """
    # torch.set_printoptions(precision=8, sci_mode=False)

    unsigned_distance = get_dist(points)
    if filter_outer_fn is not None:
        out_obs = filter_outer_fn(points[..., :3].unsqueeze(0)).squeeze()
        unsigned_distance[out_obs] = 0.
    
    where_d = unsigned_distance >= offset #TODO: check =

    if prob > 0:
        where_d = flip_false_to_true(where_d, prob)
    return where_d

def sample_and_filter_points(numsamples, dim, device, sample_enlarge_ratio=2, bbox=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], fix_dim_dict=None, filter_outer_fn=None):        
    """
    return:
        start, goal: [numsamples, dim]
    """
    bbox = torch.tensor(bbox, dtype=torch.float32, device=device)
    P  = torch.rand((int(sample_enlarge_ratio * numsamples), dim), dtype=torch.float32, device=device) * (bbox[1] - bbox[0]) + bbox[0]
    dP = torch.rand((int(sample_enlarge_ratio * numsamples), dim), dtype=torch.float32, device=device)-0.5
    rL = (torch.rand((int(sample_enlarge_ratio * numsamples), 1), dtype=torch.float32, device=device))*torch.sqrt(torch.sum((bbox[1] - bbox[0])**2))
    nP = P + torch.nn.functional.normalize(dP, dim=1)*rL

    PointsInside = torch.all((nP <= bbox[1]),dim=1) & torch.all((nP >= bbox[0]),dim=1)
    x0 = P[PointsInside, :]
    x1 = nP[PointsInside, :]
    
    if filter_outer_fn is not None:
        in_obs = filter_outer_fn(x0.detach().cpu().numpy()[..., :3])
        in_obs &= filter_outer_fn(x1.detach().cpu().numpy()[..., :3])
        
        x0 = x0[in_obs]
        x1 = x1[in_obs]
    
    #TODO: check if the projected points still subject to sampling distribution
    if fix_dim_dict is not None:
        for dim, v in fix_dim_dict.items():
            x0[:, dim] = v
            x1[:, dim] = v
    return x0, x1


def arm_transform(states, chain: pk.SerialChain, scale, sample_points_list, keep_links=False, batch_size=80000, sim=False):
    """
    args:
        states: (n, dof)
    return:
        points: (n, sample_size, 3) [keep_links=False] or (n, n_link, link_sample_size, 3) [keep_links=True]
    """
    theta =  scale * states
    query_points_list = []
    for bid in range(math.ceil(states.shape[0] / batch_size)):
        tg_batch = chain.forward_kinematics(
            theta[bid * batch_size: (bid + 1) * batch_size], end_only=False)
        p_list = []
        it = 0
        pointsize = 0
        for tg in tg_batch:
            if it > 1:
                # v = np.load(os.path.join(out_dir, 'meshes/collision', tg + '.npy))
                v = sample_points_list[it - 2]
                # nv = np.ones((v.shape[0], 4))
                pointsize = pointsize + v.shape[0]

                t = torch.ones((v.shape[0], 4)).float().cuda()
                t[:, :3] = v[:, :3]
                m = tg_batch[tg].get_matrix() # (batch_size, 4, 4)
                p = torch.matmul(m[:], t.T)
                p = torch.permute(p, (0, 2, 1)).contiguous()
                p_list.append(p)
                del m, p, t, v
            it += 1
        
        if keep_links:
            p = torch.stack(p_list, dim = 1) # (batch_size, n_link, link_sample_size, 4)
        else:
            p = torch.cat(p_list, dim=1) # (batch_size, sample_size, 4), sample_size = total points of all the links
        # p = torch.reshape(p, (p.shape[0] * p.shape[1], p.shape[2])).contiguous()
        query_points = p[..., 0:3].contiguous() # (batch_size, sample_size, 3) or (batch_size, n_link, link_sample_size, 4)
        if sim:
            query_points = query_points * 0.4
        query_points_list.append(query_points)
    
    query_points = torch.cat(query_points_list, dim=0) # scale the arm
    return query_points

def arm_get_distances(x, bvh, triangles, chain: pk.SerialChain, scale, sample_points_list, sim=False):
    query_points = arm_transform(x, chain, scale, sample_points_list=sample_points_list, sim=sim)
    pointsize = query_points.shape[1]
    query_points = query_points.reshape(1, -1, query_points.shape[2]).contiguous()
    unsigned_distance = bvh_query(bvh, triangles, query_points, pointsize=pointsize)

    return unsigned_distance

def point_get_distances(x, bvh, triangles):
    query_points = x
    query_points = query_points.unsqueeze(dim=0)

    unsigned_distance = bvh_query(bvh, triangles, query_points)
    
    return unsigned_distance

def robot_transform(points, dof, robot, scale, rotate_axis='x'):
    """
    args:
        points: (n, dof) [x, y, z, [theta1, theta2, theta3]]
        dof: int
        robot: (robot_size, 3)
    return:
        coords: [x, y, z], shape: (n, sample_size, 3)
    """
    # 4dof for complex robot
    if dof > 3:
        theta = points[:, 3:]
    
    n = points.size(0)
    if len(robot.shape) == 3:
        coords = robot
        print("load random shape robot!")
    else:
        coords = robot.repeat(n, 1, 1) # (n, sample_size, 3)   sample points of the robot in different start positions
    # coords = robot.clone()
    
    if dof > 3:
        theta = theta * scale
        device = theta.device    
        if dof == 4:
            if rotate_axis == 'x':
                Rx = torch.tensor([1, 0, 0]).cuda().repeat(n, 1, 1)
                Rx = torch.cat([Rx, torch.stack([torch.zeros(n, dtype=torch.float32, device=device), torch.cos(theta[:,0]), -torch.sin(theta[:,0])], dim=-1).unsqueeze(1)], dim=-2)
                Rx = torch.cat([Rx, torch.stack([torch.zeros(n, dtype=torch.float32, device=device), torch.sin(theta[:,0]), torch.cos(theta[:,0])], dim=-1).unsqueeze(1)], dim=-2)
                R = Rx
            elif rotate_axis == 'y':
                Ry = torch.stack([torch.cos(theta[:,0]), torch.zeros(n, dtype=torch.float32, device=device), torch.sin(theta[:,0])], dim=-1).unsqueeze(1)
                Ry = torch.cat([Ry, torch.tensor([0, 1, 0]).cuda().repeat(n, 1, 1)], dim=-2)
                Ry = torch.cat([Ry, torch.stack([-torch.sin(theta[:,0]), torch.zeros(n, dtype=torch.float32, device=device), torch.cos(theta[:,0])], dim=-1).unsqueeze(1)], dim=-2)
                R = Ry
            else:
                Rz = torch.stack([torch.cos(theta[:,0]), -torch.sin(theta[:,0]), torch.zeros(n, dtype=torch.float32, device=device),
                                  torch.sin(theta[:,0]),  torch.cos(theta[:,0]), torch.zeros(n, dtype=torch.float32, device=device),
                                  torch.zeros(n, device=device).float(), torch.zeros(n, device=device).float(), torch.ones(n, device=device).float()], dim=1).reshape(n, 3, 3)
                # Rz = torch.stack([torch.cos(theta[:,0]), -torch.sin(theta[:,0]), torch.zeros(n, dtype=torch.float32, device=device)], dim=-1).unsqueeze(1)
                # Rz = torch.cat([Rz, torch.stack([torch.sin(theta[:,0]), torch.cos(theta[:,0]), torch.zeros(n, dtype=torch.float32, device=device)], dim=-1).unsqueeze(1)], dim=-2)
                # Rz = torch.cat([Rz, torch.tensor([0, 0, 1]).cuda().repeat(n, 1, 1)], dim=-2)
                R = Rz
        elif dof == 6:
            Rx = torch.tensor([1, 0, 0]).cuda().repeat(n, 1, 1)
            Rx = torch.cat([Rx, torch.stack([torch.zeros(n, dtype=torch.float32, device=device), torch.cos(theta[:,0]), -torch.sin(theta[:,0])], dim=-1).unsqueeze(1)], dim=-2)
            Rx = torch.cat([Rx, torch.stack([torch.zeros(n, dtype=torch.float32, device=device), torch.sin(theta[:,0]), torch.cos(theta[:,0])], dim=-1).unsqueeze(1)], dim=-2)

            Ry = torch.stack([torch.cos(theta[:,1]), torch.zeros(n, dtype=torch.float32, device=theta.device), torch.sin(theta[:,1])], dim=-1).unsqueeze(1)
            Ry = torch.cat([Ry, torch.tensor([0, 1, 0]).cuda().repeat(n, 1, 1)], dim=-2)
            Ry = torch.cat([Ry, torch.stack([-torch.sin(theta[:,1]), torch.zeros(n, dtype=torch.float32, device=theta.device), torch.cos(theta[:,1])], dim=-1).unsqueeze(1)], dim=-2)

            Rz = torch.stack([torch.cos(theta[:,2]), -torch.sin(theta[:,2]), torch.zeros(n, dtype=torch.float32, device=theta.device)], dim=-1).unsqueeze(1)
            Rz = torch.cat([Rz, torch.stack([torch.sin(theta[:,2]), torch.cos(theta[:,2]), torch.zeros(n, dtype=torch.float32, device=theta.device)], dim=-1).unsqueeze(1)], dim=-2)
            Rz = torch.cat([Rz, torch.tensor([0, 0, 1]).cuda().repeat(n, 1, 1)], dim=-2)
            
            
            R = torch.bmm(Rz, torch.bmm(Ry, Rx)) # (n, 3, 3)
        else:
            raise NotImplementedError
        
        coords = torch.bmm(R, coords.permute(0, 2, 1)).permute(0, 2, 1) # (n, sample_size, 3)
        # coords = torch.einsum("ijk, lk -> ilj", R, coords)

    # translate the robot
    xyz = points[:, :3]
    coords = coords + xyz.unsqueeze(dim=1) # (n, sample_size, 3)
    return coords

def robot_get_distances(x, bvh, triangles, robot, scale, rotate_axis='x', dim=3):
    
    robot_size = robot.size(0)
    coords = robot_transform(x, dim, robot, scale, rotate_axis)
    coords = coords.reshape(-1, 3).contiguous()
    query_points = coords.unsqueeze(dim=0)
    
    unsigned_distance = bvh_query(bvh, triangles, query_points, robot_size)
    
    return unsigned_distance

def prepare_model(dim, sample_num, h_dim, ckpt_path, device='cuda'):
    """
    args:
        ckpt_path: path to the model checkpoint
    return:
        model: the UDF model
    """
    from models.model_obj_udf_random_shape import ObjectMinUDF
    print(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=torch.device(device))
    model = ObjectMinUDF(in_dim=dim, hidden_dim=h_dim, n_points=sample_num)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    model.to(device)
    print(">>> Object udf model initialization completed!")
    return model

def mlp_dists(points, model, robot, scale, rotate_axis='x', dim=3):
    """
    args:
        points: [x, y [, theta]], shape: (N, 2 or 3)
    return:
        dists: shape: (N, )
    """
    p_size = points.shape[0]

    coords = robot_transform(points, dim, robot, scale, rotate_axis)
    
    batch_size = int(10 ** 7/coords.shape[1])
    with torch.no_grad():
        if coords.shape[0] > batch_size:
            dists = []
            for bid in range(0, math.ceil(coords.shape[0] / batch_size)):
                dist = model.out(coords[bid * batch_size: (bid + 1) * batch_size]).squeeze()
                dists.append(dist)
            dists = torch.cat(dists, dim=0) #(n,)
        else:
            dists = model.out(coords).squeeze()
    
    return dists

def prepare_gen_udf(dim, h_dim, ckpt_path, device='cuda'):
    """
    args:
        dim: 3, [x, y, z], SE3 transformed position, so dim=3
        ckpt_path: path to the model checkpoint
    return:
        model: the UDF model
    """
    from obj_env_udf.models.model_gen_udf import ObjectGenUDF
    ckpt = torch.load(ckpt_path, map_location=torch.device(device))
    model = ObjectGenUDF(in_dim=dim, hidden_dim=h_dim)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    model.to(device)
    print(">>> Object Gen udf model initialization completed!")
    return model

@torch.no_grad()
def gen_udf_dists(states, model, sparse_robot, dense_robot, scale, rotate_axis='x', dim=4):
    """
    args:
        states: [x, y, z, [, theta]], shape: (N, dim)
    return:
        dists: shape: (N, )
    """
    
    coords = robot_transform(states, dim, sparse_robot, scale, rotate_axis)
    batch_size = int(10 ** 6 / coords.shape[1])
    if coords.shape[0] > batch_size:
        dists = []
        for bid in range(0, math.ceil(coords.shape[0] / batch_size)):
            dist = model.out(coords[bid * batch_size: (bid + 1) * batch_size], sparse_robot, dense_robot).squeeze(dim=1)
            dists.append(dist)
        dists = torch.cat(dists, dim=0)  # (n,)
    else:
        dists = model.out(coords, sparse_robot, dense_robot).squeeze(dim=1)
    
    return dists

def prepare_gen_udf_baseline(dim, h_dim, ckpt_path, device='cuda'):
    """
    args:
        dim: 6, [x, y, z, th1, th2, th3]
        ckpt_path: path to the model checkpoint
    return:
        model: the UDF model
    """
    from obj_env_udf.models.model_gen_udf import ObjectGenUDF_Baseline
    ckpt = torch.load(ckpt_path, map_location=torch.device(device))
    model = ObjectGenUDF_Baseline(in_dim=dim, hidden_dim=h_dim)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    model.to(device)
    print(">>> Object Gen udf Baseline model initialization completed!")
    return model

@torch.no_grad()
def gen_udf_baseline_dists(states, model, dense_robot):
    """
    args:
        states: [x, y, z, [, theta]], shape: (N, dim)
    return:
        dists: shape: (N, )
    """
    
    batch_size = int(10 ** 4 / states.shape[1])
    if states.shape[0] > batch_size:
        dists = []
        for bid in range(0, math.ceil(states.shape[0] / batch_size)):
            dist = model.out(states[bid * batch_size: (bid + 1) * batch_size], dense_robot).squeeze(dim=1)
            dists.append(dist)
        dists = torch.cat(dists, dim=0)  # (n,)
    else:
        dists = model.out(states, dense_robot).squeeze(dim=1)
    
    return dists

@torch.no_grad()
def arm_gen_udf_dists(states, model, sparse_links, dense_links, chain, scale):
    """
    args:
        states: [th1, th2, ..., thn], shape: (N, dim)
    return:
        dists: shape: (N, )
    """
    links_coords = arm_transform(states, chain, scale, sparse_links, keep_links=True) # (n, n_links, link_sample_size, 3)
    batch_size = int(10 ** 6 / (links_coords.shape[2]))
    sparse_links = torch.stack(sparse_links, dim=0)
    
    for i in range(len(sparse_links)):
        coords = links_coords[:, i, ...] # (n, link_sample_size, 3)
        sparse_robot = sparse_links[i]
        dense_robot = dense_links[i]
        if coords.shape[0] > batch_size:
            dists = []
            for bid in range(0, math.ceil(coords.shape[0] / batch_size)):
                dist = model.out_seq(coords[bid * batch_size: (bid + 1) * batch_size], sparse_robot, dense_robot, i).squeeze(dim=1)
                # dist = model.out(coords[bid * batch_size: (bid + 1) * batch_size], sparse_robot, dense_robot).squeeze(dim=1)
                dists.append(dist)
            dists = torch.cat(dists, dim=0)  # (n,)
        else:
            dists = model.out_seq(coords, sparse_robot, dense_robot, i).squeeze(dim=1)
        if i == 0:
            min_dists = dists
        else:
            min_dists = torch.min(min_dists, dists)
    return min_dists

def preprepare_arm(robot_dir, dtype=torch.float32, device='cuda', sparse=False, sim=False):
    with open(os.path.join(robot_dir, 'dim')) as f:
        iter = 0
        for line in f:
            data = line.split()
            if iter==0:
                dim = int(data[0])
            else:
                end_effect = data[0]
            iter=iter+1
    chain = pk.build_serial_chain_from_urdf(
        open(glob.glob(os.path.join(robot_dir, '*.urdf'))[0]).read(), end_effect)
    chain = chain.to(dtype=dtype, device=device)
    frame_names = chain.get_frame_names()
    def to_mesh_name(frame_name: str) -> str:
        # 只在确实有后缀时才去掉
        for suffix in ['_frame', '/frame', '.frame']:
            if frame_name.endswith(suffix):
                return frame_name[:-len(suffix)]
        return frame_name
    link_names = [to_mesh_name(name) for name in frame_names]
    if sim:
        link_names = link_names[1:]
    arm_dir = os.path.join(robot_dir, 'meshes/collision')
    print("link name:", link_names)
    if len(glob.glob(os.path.join(arm_dir, '*.ply'))) > 0:
        v_list = [
            torch.tensor(o3d.io.read_point_cloud(os.path.join(arm_dir, link_names[i] + '.ply')).points, dtype=dtype, device=device)
            for i in range(len(link_names))
        ]
        if sparse:
            sparse_v_list = [
                torch.tensor(o3d.io.read_point_cloud(os.path.join(arm_dir, link_names[i] + '_sparse' + '.ply')).points, dtype=dtype, device=device)
                for i in range(len(link_names))
            ]
            v_list = torch.stack(v_list, dim=0)
    elif len(glob.glob(os.path.join(arm_dir, '*.obj'))) > 0:
        v_list = [
            torch.tensor(igl.read_triangle_mesh(os.path.join(arm_dir, link_names[i] + '.obj'))[0], dtype=dtype, device=device)
            for i in range(len(link_names))
        ]
    else:
        v_list = [
            torch.from_numpy(np.load(os.path.join(arm_dir, link_names[i] + '.npy'))).type(dtype).to(device)
            for i in range(len(link_names))
        ]
    if sparse:
        return chain, sparse_v_list, v_list
    return chain, v_list

def get_dist_fn(dist_fn_type, robot_name, robot_dir, dim, triangles, device, rotate_axis='x', use_coll_robot=False, in_dim=3, h_dim=None, ckpt_path=None, **arm_kwargs):
    dtype = torch.float32
    bvh = bvh_distance_queries.BVH()
    
    if robot_name == 'arm':
        scale = 0.9 * math.pi / 0.5
        if dist_fn_type == 'acc':
            chain, v_list = preprepare_arm(robot_dir, dtype=dtype, device=device, **arm_kwargs)
            get_distances = {
                'func': arm_get_distances,
                'args': [bvh, triangles, chain, scale, v_list],
                'kwargs': arm_kwargs
            }
        else:
            chain, sparse_v_list, dense_v_list = preprepare_arm(robot_dir, dtype=dtype, device=device, sparse=True, **arm_kwargs)
            model = prepare_gen_udf(in_dim, h_dim, ckpt_path, device='cuda')
            get_distances = {
                'func': arm_gen_udf_dists,
                'args': [model, sparse_v_list, dense_v_list, chain, scale],
            }

    elif robot_name == 'point':
        get_distances = {
            'func': point_get_distances,
            'args': [bvh, triangles],
        }
    elif robot_name == 'robot':
        if use_coll_robot:
            robot = np.load(os.path.join(robot_dir, 'robot_for_cls.npy'))
            robot = torch.tensor(robot, dtype=torch.float32, device='cuda')
        else:
            robot = np.load(os.path.join(robot_dir, 'robot.npy'))
            robot = torch.tensor(robot, dtype=torch.float32, device='cuda')
        
        scale = math.pi * 2
        robot_size = robot.size(0)
        print(f'robot size: {robot_size}')
        if dist_fn_type =='acc':
            get_distances = {
                'func': robot_get_distances,
                'args': [bvh, triangles, robot, scale, rotate_axis, dim],
            }
        elif dist_fn_type == 'obj_udf':
            model = prepare_model(in_dim, robot.shape[0], h_dim, ckpt_path, device='cuda')
            get_distances = {
                'func': mlp_dists,
                'args': [model, robot, scale, rotate_axis, dim],
            }
        elif dist_fn_type == 'obj_gen_udf':
            sparse_robot = np.load(os.path.join(robot_dir, 'robot_sparse.npy'))
            sparse_robot = torch.tensor(sparse_robot, dtype=torch.float32, device='cuda')
            sparse_robot_size = sparse_robot.size(0)
            print(f'sparse robot size: {sparse_robot_size}')
            model = prepare_gen_udf(in_dim, h_dim, ckpt_path, device='cuda')
            get_distances = {
                'func': gen_udf_dists,
                'args': [model, sparse_robot, robot, scale, rotate_axis, dim],
            }
        elif dist_fn_type == 'obj_gen_udf_baseline':
            print(f'dense robot size: {robot_size}')
            model = prepare_gen_udf_baseline(in_dim, h_dim, ckpt_path, device='cuda')
            get_distances = {
                'func': gen_udf_baseline_dists,
                'args': [model, robot],
            }
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    if get_distances.get('kwargs', None) is not None:
        return lambda x: get_distances['func'](x, *get_distances['args'], **get_distances['kwargs'])
    return lambda x: get_distances['func'](x, *get_distances['args'])


def sample_points_and_distance(numsamples, dim, task_name, offset, margin, device, triangles, out_dir, cfg, sample_enlarge_ratio=2, colls_prob=0):
    get_dists = get_dist_fn(cfg.dist_fn_type, task_name, out_dir, dim, triangles, device, cfg.rotate_axis, in_dim=cfg.in_dim, h_dim=cfg.h_dim, ckpt_path=cfg.ckpt_path, sim=cfg.sim)
    get_dists_coll = get_dist_fn(cfg.dist_fn_type, task_name, out_dir, dim, triangles, device, cfg.rotate_axis,
                                cfg.use_no_coll_split or cfg.check_colls, in_dim=cfg.in_dim, h_dim=cfg.h_dim, ckpt_path=cfg.ckpt_path, sim=cfg.sim)


    X_list = []
    Y_list = []
    OutsideSize = numsamples + 10
    WholeSize = 0
    i = 0
    
    if cfg.use_no_coll_split:
        no_coll_mask_list = []

    while OutsideSize > 0:
        fix_dim_dict = {}
        if cfg.fix_dim is not None and cfg.fix_dim_v is not None:
            for j in range(len(cfg.fix_dim)):
                fix_dim_dict[cfg.fix_dim[j]] = cfg.fix_dim_v[j]
        x0, x1 = sample_and_filter_points(numsamples, dim, device, sample_enlarge_ratio, bbox=cfg.bbox, fix_dim_dict=fix_dim_dict)
        
        if(x0.shape[0]<=1):
            print('x0 generate size is too small')
            continue

        unsigned_distance = get_dists(x0)
        
        y0 = unsigned_distance
        if cfg.filter_outer:
            out_obs = cfg.filter_outer_fn(x0[..., :3].unsqueeze(0)).squeeze()
            y0[out_obs] = 0.

        if cfg.filter_start:
            where_d = (unsigned_distance <=  margin) & \
                        (unsigned_distance >=  offset)
            x0 = x0[where_d]
            x1 = x1[where_d]
            y0 = y0[where_d]

        if cfg.check_colls:
            n_colls_points = check_collision(get_dists_coll, x0, offset, prob=colls_prob)
            x0 = x0[n_colls_points]
            x1 = x1[n_colls_points]
            y0 = y0[n_colls_points]
        
        if(x1.shape[0]<=1):
            print('x1 generate size is too small')
            continue
        

        unsigned_distance = get_dists(x1)
        
        y1 = unsigned_distance
        
        if cfg.filter_outer:
            # out_obs = cfg.filter_outer_fn(x1.detach().cpu().numpy()[..., :3])
            out_obs = cfg.filter_outer_fn(x1[..., :3].unsqueeze(0)).squeeze()
            y1[out_obs] = 0.
        
        if cfg.check_colls:
            n_colls_points = check_collision(get_dists_coll, x1, offset, prob=colls_prob)
            x0 = x0[n_colls_points]
            x1 = x1[n_colls_points]
            y0 = y0[n_colls_points]
            y1 = y1[n_colls_points]
        
        if cfg.use_no_coll_split:
            start_no_coll = check_collision(get_dists_coll, x0, offset, filter_outer_fn=cfg.filter_outer_fn)
            goal_no_coll = check_collision(get_dists_coll, x1, offset, filter_outer_fn=cfg.filter_outer_fn)
            no_coll_mask = start_no_coll & goal_no_coll
            pair_dists = torch.linalg.norm(x0 - x1, dim=1)
            no_coll_mask &= (pair_dists > 0.12)
            no_coll_mask_list.append(no_coll_mask.detach().cpu().numpy())
            
        x = torch.cat((x0,x1),1)
        y = torch.cat((y0.unsqueeze(1),y1.unsqueeze(1)),1)

        X_list.append(x.detach().cpu().numpy())
        Y_list.append(y.detach().cpu().numpy())
        OutsideSize = OutsideSize - x.shape[0]
        WholeSize = WholeSize + x.shape[0]
        print(f'>>> iter {i}, processed {WholeSize} points')
        if(WholeSize > numsamples):
            break
        del x0, x1, y0, y1, x, y, unsigned_distance
        if cfg.filter_start:
            del where_d
        torch.cuda.empty_cache()
        i += 1
    
    sampled_points = np.concatenate(X_list, axis=0)[:numsamples]
    distance = np.concatenate(Y_list, axis=0)[:numsamples]

    return_vals = [sampled_points, distance]
    
    if cfg.use_no_coll_split:
        no_coll_mask = np.concatenate(no_coll_mask_list, axis=0)[:numsamples]
        return_vals.append(no_coll_mask)

    return return_vals

def sample_bound_points_and_speed(numsamples, dim, vertices, faces, velocity_max, offset, margin, out_dir, task_name, cfg):
    
    numsamples = int(numsamples)
    
    total_samples = numsamples
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    vertices = torch.tensor(vertices, dtype=torch.float32, device='cuda')
    faces = torch.tensor(faces, dtype=torch.long, device='cuda')
    
    triangles = vertices[faces].unsqueeze(dim=0)
    
    print('vertices shape', vertices.shape)
    print('faces shape', faces.shape)
    print('triangles shape', triangles.shape)
    
    r = sample_points_and_distance(numsamples, dim, task_name, offset, margin, device, triangles, out_dir, cfg, colls_prob=cfg.colls_prob)
    sampled_points, distance = r[:2]

    if cfg.use_no_coll_split:
        no_coll_mask = r[2]

    if cfg.speed_model == 'exp':
        speed0 = np.exp(np.log(velocity_max + 1) * np.clip(distance[:, 0], a_min=offset, a_max=margin)/margin)
        speed1 = np.exp(np.log(velocity_max + 1) * np.clip(distance[:, 1], a_min=offset, a_max=margin)/margin)
    elif cfg.speed_model == 'log':
        speed0 = velocity_max / np.log(margin + 1) * np.log(np.clip(distance[:, 0], a_min=offset, a_max=margin) + 1)
        speed1 = velocity_max / np.log(margin + 1) * np.log(np.clip(distance[:, 1], a_min=offset, a_max=margin) + 1)
    elif cfg.speed_model == 'exp_inv': # exp(-d^2/(lambda*epsilon^2))
        epsilon = margin/2
        speed0 = np.exp(-np.square(np.minimum(distance[:, 0], epsilon))/(0.25*epsilon**2))
        speed1 = np.exp(-np.square(np.minimum(distance[:, 1], epsilon))/(0.25*epsilon**2))
    else:
        speed0 = velocity_max*np.clip(distance[:,0] , a_min = offset, a_max = margin)/margin
        speed1 = velocity_max*np.clip(distance[:,1] , a_min = offset, a_max = margin)/margin
    
    speed = np.stack((speed0, speed1), axis=1)
    
    return_vals = [sampled_points, speed]
    if cfg.use_no_coll_split:
        return_vals.append(no_coll_mask)

    return return_vals

def sample_speed(path, task_name, numsamples, dim, args):
    
    try:
        out_dir = os.path.dirname(path)
        file_name = os.path.splitext(os.path.basename(path))[0]
        input_file = os.path.join(out_dir, file_name + '_scaled.off')
        print('Env for sample:', input_file)
        
        out_file = os.path.join(out_dir, 'data.h5')
        if os.path.exists(out_file):
            print(f'Exists: {out_file}')

        if args.no_scale:
            limit = np.min(args.bbox[1, :2] - args.bbox[0, :2]).item()
        else:
            limit = 0.5
        
        print('limit:', limit)
        velocity_max = args.vmax
        
        if task_name=='c3d' or task_name=='test':
            margin = limit/5.0
            offset = margin/10.0 
        elif task_name=='gibson' or task_name == 'gibson_complex':
            margin = limit/20.0
            offset = margin/10.0
        elif task_name=='arm' or task_name == 'cabinet':
            margin = limit/10.0
            offset = margin/15.0
        elif task_name == 'room':
            margin = limit/10.0
            offset = margin/7.0
        elif task_name in ['room_low_speed', 'room_fmm', 'room_thick', 'room_split', 'room_point']:
            margin = limit/5.0
            offset = margin/10.0
        elif task_name == 'room_low_low_speed':
            margin = limit/10.0
            offset = margin/30.0
        else:
            raise NotImplementedError
        
        args.margin = margin
        args.offset = offset
        
        print('margin:', margin)
        print('offset:', offset)
        
        v, f = igl.read_triangle_mesh(input_file)

        extra_cfg = edict()
        extra_cfg.check_colls = args.check_colls
        extra_cfg.colls_prob = args.colls_prob
        extra_cfg.speed_model = args.speed_model
        extra_cfg.rotate_axis = args.rotate_axis
        extra_cfg.use_no_coll_split = args.use_no_coll_split
        extra_cfg.bbox = args.bbox
        extra_cfg.fix_dim = args.fix_dim
        extra_cfg.fix_dim_v = args.fix_dim_v
        extra_cfg.filter_outer = args.filter_outer
        extra_cfg.sim = args.sim
        """udf model args"""
        extra_cfg.dist_fn_type = args.dist_fn_type
        extra_cfg.in_dim = args.in_dim
        extra_cfg.h_dim  = args.h_dim
        extra_cfg.ckpt_path  = args.ckpt_path
        extra_cfg.filter_start = not args.no_filter_start
        extra_cfg.robot = args.robot
        
        if args.filter_outer:
            ref_obj_scaled_path = os.path.join(out_dir, os.path.splitext(args.ref_obj)[0] + '_scaled.off')
            # ref_obj_v, ref_obj_f = igl.read_triangle_mesh(ref_obj_scaled_path)
            # extra_cfg.ref_obj_v = ref_obj_v
            # extra_cfg.ref_obj_f = ref_obj_f
            # ref_obj_path = os.path.join(out_dir, args.ref_obj)
            
            # mesh = trimesh.load(ref_obj_scaled_path)
            mesh = kal.io.off.import_mesh(ref_obj_scaled_path)

            # os.remove(ref_obj_scaled_path)
            # os.remove(ref_obj_path)
            print(f'>>> already read vertices and faces of the ref obj {args.ref_obj}')
            # print(f'delete {ref_obj_scaled_path}')
            # print(f'delete {ref_obj_path}')

            # cfg.filter_outer_fn = lambda x: igl.winding_number(cfg.ref_obj_v, cfg.ref_obj_f, x) >= 0
            # extra_cfg.filter_outer_fn = lambda x: igl.signed_distance(x, ref_obj_v, ref_obj_f)[0] < 0
            # extra_cfg.filter_outer_fn = lambda x: ~mesh.contains(x)
            kal_v = mesh.vertices.unsqueeze(0).cuda()
            kal_f = mesh.faces.cuda()
            extra_cfg.filter_outer_fn = lambda x: ~kal.ops.mesh.check_sign(kal_v, kal_f, x)
        else:
            extra_cfg.filter_outer_fn = None

        start = timer()

        if task_name in ['arm', 'gibson', 'room_point', 'room', 'room_low_speed', 'room_thick', 'room_low_low_speed', 'room_split', 'gibson_complex', 'cabinet']: # ! consider rigid body
            if task_name == 'arm':
                op = 'arm'
            elif task_name in ['room_point', 'gibson']:
                op = 'point'
            else:
                op = 'robot'
            r = sample_bound_points_and_speed(numsamples, dim, v, f, velocity_max, offset, margin, out_dir, op, extra_cfg)
            if extra_cfg.use_no_coll_split:
                sampled_points, speed, no_coll_mask = r
            else:
                sampled_points, speed = r

            print('sampled points shape', sampled_points.shape)
            print('speed shape', speed.shape)
            start_points, goal_points = sampled_points[:, :dim], sampled_points[:, dim:]
            start_speed, goal_speed = speed[:, 0], speed[:, 1]
        else:
            raise NotImplementedError

        end = timer()

        print("time: {:.6f} s".format(end-start))

        print('speed max', np.max(speed))
        print('speed min', np.min(speed))
        
        """visualize"""
        if args.use_no_coll_split:
            visual_space_labels(out_dir, start_points, no_coll_mask.astype(int), 'start')
            visual_space_labels(out_dir, goal_points, no_coll_mask.astype(int), 'goal')
        
        print('Start visualizing.')
        visual_point_speed(out_dir, start_points, start_speed, 'start')
        visual_point_speed(out_dir, goal_points, goal_speed, 'goal')
        
        if args.fix_dim is not None and args.fix_dim_v is not None:
            dims = list(range(args.num_dim))
            for i in range(len(args.fix_dim)):
                dims.remove(args.fix_dim[i])
            print("reserved dims:", dims)
            start_points = start_points[:, dims]
            goal_points = goal_points[:, dims]
            args.bbox = args.bbox[:, dims]
        
        write_hdf5(out_dir, ['start_points', 'goal_points', 'start_speed', 'goal_speed'], [start_points, goal_points, start_speed, goal_speed])
        
        append_hdf5(out_dir, ['bbox'], [args.bbox])
        append_hdf5(out_dir, ['ori_bbox'], [args.ori_bbox])
        if args.fix_dim is not None and args.fix_dim_v is not None:
            append_hdf5(out_dir, ['fix_dim', 'fix_dim_v'], [np.array(args.fix_dim), np.array(args.fix_dim_v)])
        if args.use_no_coll_split:
            append_hdf5(out_dir, 'no_coll_mask', no_coll_mask)
        args.bbox = args.bbox.tolist()
        args.ori_bbox = args.ori_bbox.tolist()
    except Exception as err:
        print('Error with {}: {}'.format(path, traceback.format_exc()))
    
