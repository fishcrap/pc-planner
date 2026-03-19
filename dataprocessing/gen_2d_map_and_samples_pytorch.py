import matplotlib.pyplot as plt
import shapely
import random
import numpy as np
import torch
import time
import math
import os
from utils.util import write_hdf5, read_hdf5, append_hdf5

def gen_2d_map(num_obstacles, env_width, env_height):
    # Create random rectangular obstacles without overlapping
    obstacles = []
    
    max_obstacle_width = env_width / 5
    max_obstacle_height = env_height / 5
    min_obstacle_width = env_width / 20
    min_obstacle_height = env_height / 20
    
    half_env_width = env_width / 2
    half_env_height = env_height / 2

    def is_collision(new_obstacle, existing_obstacles):
        for obstacle in existing_obstacles:
            if new_obstacle.intersects(obstacle):
                return True
        return False

    def is_collision_hard(new_obstacle, existing_obstacles):
        min_distance = math.inf
        for obstacle in existing_obstacles:
            if new_obstacle.distance(obstacle) < min_distance:
                min_distance = new_obstacle.distance(obstacle)
        if min_distance < 0.1:
            return True    
        return False

    for _ in range(num_obstacles):
        while True:
            # Randomly generate obstacle position and size
            x1 = random.uniform(-half_env_width, half_env_width)
            y1 = random.uniform(-half_env_height, half_env_height)
            width = random.uniform(min_obstacle_width, max_obstacle_width)
            height = random.uniform(min_obstacle_height, max_obstacle_height)
            
            x2 = min(x1 + width, half_env_width)
            y2 = min(y1 + height, half_env_height)
            
            # Create a Polygon object for the obstacle and add it to the list
            obstacle = shapely.geometry.Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

            # Check for collisions with existing obstacles
            if not is_collision_hard(obstacle, obstacles):
                obstacles.append(obstacle)
                break
    return obstacles

def sample_and_filter_2d_points(numsamples, dof, env_width, env_height, random_loc, sample_enlarge_ratio=2):
    # Randomly generate points
    half_env_width = env_width / 2
    half_env_height = env_height / 2
    scaled_num_samples = int(sample_enlarge_ratio * numsamples)
    start_x = (torch.rand(scaled_num_samples, dtype=torch.float32, device='cuda') - 0.5) * env_width
    start_y = (torch.rand(scaled_num_samples, dtype=torch.float32, device='cuda') - 0.5) * env_height
    if dof == 3:
        start_theta = torch.rand(scaled_num_samples, dtype=torch.float32, device='cuda') - 0.5
        start_points = torch.stack((start_x, start_y, start_theta), dim=1)
        stretch_len = math.sqrt(env_width ** 2 + env_height ** 2 + 1) # 1 for theta
    else:
        start_points = torch.stack((start_x, start_y), dim=1)
        stretch_len = math.sqrt(env_width ** 2 + env_height ** 2)
    if random_loc:
        goal_x = (torch.rand(scaled_num_samples, dtype=torch.float32, device='cuda') - 0.5) * env_width
        goal_y = (torch.rand(scaled_num_samples, dtype=torch.float32, device='cuda') - 0.5) * env_height
        if dof == 3:
            goal_theta = torch.rand(scaled_num_samples, dtype=torch.float32, device='cuda') - 0.5
            goal_points = torch.stack((goal_x, goal_y, goal_theta), dim=1)
        else:
            goal_points = torch.stack((goal_x, goal_y), dim=1)
    else:
        direction = torch.rand((scaled_num_samples, dof), dtype=torch.float32, device='cuda') - 0.5
        raidus = torch.rand((scaled_num_samples, 1), dtype=torch.float32, device='cuda') * stretch_len

        goal_points = start_points + torch.nn.functional.normalize(direction, dim=1) * raidus
        
        PointInside = (goal_points[:, 0] >= -half_env_width) & (goal_points[:, 0] <= half_env_width) & (goal_points[:, 1] >= -half_env_height) & (goal_points[:, 1] <= half_env_height)
        if dof == 3:
            PointInside &= (goal_points[:, 2] >= -0.5) & (goal_points[:, 2] <= 0.5)
        
        start_points = start_points[PointInside]
        goal_points = goal_points[PointInside]
    
    return start_points, goal_points

def robot_transform(points, dof, robot):
    """
    args:
        points: [x, y [, theta]], shape: (n, 2 or 3)
    return:
        coords: [x, y], shape: (n, sample_size, 2)
    """
    p_size = points.shape[0]

    if dof > 2:
        scale = np.pi * 2
        theta = scale * points[:, 2]

        R = torch.stack((torch.cos(theta), -torch.sin(theta), torch.sin(theta), torch.cos(theta)), dim=1).reshape(p_size, 2, 2)

    coords = robot.repeat(p_size, 1, 1) # (n, sample_size, 2)   sample points of the robot in different positions

    if dof > 2:
        # rotate the robot
        coords = torch.einsum("ijk, ilk -> ilj", R, coords)

    # translate the robot
    coords = coords + points[:, :2].unsqueeze(dim=1) # (n, sample_size, 2)
    
    return coords

def prepare_model(dim, h_dim, ckpt_path, device='cuda'):
    """
    args:
        ckpt_path: path to the model checkpoint
    return:
        model: the UDF model
    """
    from models.model_udf import UDF
    ckpt = torch.load(ckpt_path, map_location=torch.device(device))
    model = UDF(in_dim=dim, hidden_dim=h_dim)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    model.to(device)
    return model

def mlp_dists(points, dof, model, robot=None):
    """
    args:
        points: [x, y [, theta]], shape: (N, 2 or 3)
    return:
        dists: shape: (N, )
    """
    p_size = points.shape[0]
    if robot is not None:
        coords = robot_transform(points, dof, robot)
        
        coords = coords.reshape(-1, 2).contiguous() # (n * sample_size, 2)
    
    else:
        coords = points # (n, 2)
    
    batch_size = 10 ** 7
    with torch.no_grad():
        if coords.shape[0] > batch_size:
            dists = []
            for bid in range(0, math.ceil(coords.shape[0] / batch_size)):
                dist = model(coords[bid * batch_size: (bid + 1) * batch_size]).squeeze()
                dists.append(dist)
            dists = torch.cat(dists, dim=0)
        else:
            dists = model(coords).squeeze()
    
    if robot is not None:
        # Find the minimum distance for each sampled robot's position
        dists, _ = torch.min(dists.reshape(p_size, -1), dim=1)
    
    return dists

def get_distances(points, obstacles, dof, robot=None, solid=False, task_name = None, device='cuda'):
    """
    points: [x, y [, theta]], shape: (N, 2 or 3)
    obstacles: (m, 2, k), m obstacles, each obstacle has k points
    robot: (u, 2), u points of the robot
    """
    p_size = points.shape[0]

    if robot is not None:
        coords = robot_transform(points, dof, robot)#true or false determines whether to deform or not 
        coords = coords.reshape(-1, 2).contiguous() # (n * sample_size, 2)
    else:
        coords = points # (n, 2)


    query_size = coords.shape[0]
    # Initialize the shortest distance list
    shortest_distances = torch.full((query_size,), float('inf'), dtype=torch.float32, device=device)
    
    for i, obstacle in enumerate(obstacles):
        x_obs, y_obs = obstacle[0], obstacle[1]
        x1, x2, y1, y2 = min(x_obs), max(x_obs), min(y_obs), max(y_obs)

        # Convert obstacle coordinates to Pytorch arrays
        x1, y1, x2, y2 = torch.tensor(x1, dtype=torch.float32, device=device), torch.tensor(y1, dtype=torch.float32, device=device), torch.tensor(x2, dtype=torch.float32, device=device), torch.tensor(y2, dtype=torch.float32, device=device)

        # Convert coordinates to Pytorch arrays
        x, y = coords[:, 0], coords[:, 1]

        # Convert 0 to a Pytorch array
        zero_gpu = torch.zeros(query_size, dtype=torch.float32, device=device)
        
        # Calculate the distance from inner points to rectangular obstacles
        d_left = x - x1
        d_right = x2 - x
        d_top = y2 - y
        d_bottom = y - y1
        distance_in = torch.minimum(torch.minimum(d_left, d_right), torch.minimum(d_top, d_bottom))
        
        # Calculate the distance from outer points to rectangular obstacles
        distance_x = torch.maximum(torch.maximum(-d_left, -d_right), zero_gpu)
        distance_y = torch.maximum(torch.maximum(-d_bottom, -d_top), zero_gpu)
        distance_out = torch.sqrt(distance_x ** 2 + distance_y ** 2)

        
        in_obstacle = (d_left >= 0) & (d_right >= 0) & (d_top >= 0) & (d_bottom >= 0)
        
        if solid:
            if i == len(obstacles) - 1:    
                distance = torch.where(in_obstacle, distance_in, distance_out)
            else:
                distance = distance_out
        else:
            distance = torch.where(in_obstacle, distance_in, distance_out)
            # distance = distance_out
        
        # update the shortest distance
        shortest_distances = torch.minimum(shortest_distances, distance)
    
    if robot is not None:
        # Find the minimum distance for each sampled robot's position
        shortest_distances, _ = torch.min(shortest_distances.reshape(p_size, -1), dim=1)

    return shortest_distances

def check_collision(points, get_dists, offset):
    """
    points: (n, 3), generated robot position and rotation
    get_dists: fn for getting distances
    offset: minimum distance between the robot and the room

    return a boolean tensor of shape (n, ) which indicates whether the robot collides with the room
    True: no collision
    False: collision
    """

    unsigned_distance = get_dists(points)
    where_d = unsigned_distance >= offset

    return where_d

def get_dist_fn(dist_fn_type, dof, robot, obstacles, in_dim, h_dim, ckpt_path, device, solid=False):
    if dist_fn_type == 'acc':
        dist_fn = get_distances
        # fn_args = [obstacles, dof, robot, solid]
        fn_kwargs = {'obstacles': obstacles, 'dof': dof, 'robot': robot, 'solid': solid, 'device': device}
    elif dist_fn_type == 'udf':
        model = prepare_model(in_dim, h_dim, ckpt_path, device)
        dist_fn = mlp_dists
        # fn_args = [dof, model, robot]
        fn_kwargs = {'dof': dof, 'model': model, 'robot': robot}
    else:
        raise NotImplementedError
    # return lambda x: dist_fn(x, *fn_args)
    return lambda x: dist_fn(x, **fn_kwargs)

def sample_2d_points_and_speed(numsamples, dof, obstacles, env_width, env_height, margin, offset, velocity_max, out_dir, task_name, random_loc, args, filter_start=True):

    assert(dof == 3 or dof == 2)
    if task_name != 's2d_point':
        robot = np.load(os.path.join(out_dir, 'robot.npy'))
        robot = torch.from_numpy(robot).float().cuda()
    else:
        robot = None
    
    if args.use_no_coll_split:
        robot_for_cls = np.load(os.path.join(out_dir, 'robot_for_cls.npy'))
        robot_for_cls = torch.from_numpy(robot_for_cls).float().cuda()
        get_dists_coll = get_dist_fn(args.dist_fn_type, dof, robot_for_cls, obstacles, args.in_dim, args.h_dim, args.ckpt_path, 'cuda', args.solid)
        
    get_dists = get_dist_fn(args.dist_fn_type, dof, robot, obstacles, args.in_dim, args.h_dim, args.ckpt_path, 'cuda', args.solid)
    
    if args.use_no_coll_split:
        no_coll_mask_list = []
    OutsideSize = numsamples + 10
    WholeSize = 0
    
    points_list = []
    dists_list = []
    i = 0
    
    while OutsideSize > 0:
    
        start_points, goal_points = sample_and_filter_2d_points(numsamples, dof, env_width, env_height, random_loc)

        if(start_points.shape[0] <= 1):
            print('start points generate size is too small!')
            continue

        start_dists = get_dists(start_points)
        
        if filter_start:
            where_d = (start_dists <= margin) & (start_dists >= offset)
            start_points = start_points[where_d]
            goal_points = goal_points[where_d]
            start_dists = start_dists[where_d]


        if(goal_points.shape[0] <= 1):
            print('goal points generate size is too small!')
            continue
        
        goal_dists = get_dists(goal_points)
        
        if args.use_no_coll_split:
            start_no_coll = check_collision(start_points, get_dists_coll, offset)
            goal_no_coll = check_collision(goal_points, get_dists_coll, offset)
            no_coll_mask = start_no_coll & goal_no_coll
            pair_dists = torch.linalg.norm(start_points - goal_points, dim=1)
            no_coll_mask &= (pair_dists > 0.12)
            no_coll_mask_list.append(no_coll_mask.detach().cpu().numpy())

        points = torch.cat((start_points, goal_points), dim=1)
        dists = torch.stack((start_dists, goal_dists), dim=1)
        
        points_list.append(points.detach().cpu().numpy())
        dists_list.append(dists.detach().cpu().numpy())
        OutsideSize -= points.shape[0]
        WholeSize += points.shape[0]
        print(f'>>> iter {i}, processed {points.shape[0]} points')
        if(WholeSize > numsamples):
            break
        i += 1
    
    points = np.concatenate(points_list, axis=0)[:numsamples]
    dists = np.concatenate(dists_list, axis=0)[:numsamples]

    if args.use_no_coll_split:
        no_coll_mask = np.concatenate(no_coll_mask_list, axis=0)[:numsamples]
    
    def clip_speed(dists):
        dist_clip = np.clip(dists, a_min=offset, a_max=margin)
        speed  = velocity_max * dist_clip / margin
        return speed
    
    start_speed = clip_speed(dists[:, 0])
    goal_speed = clip_speed(dists[:, 1])
    speed = np.stack((start_speed, goal_speed), axis=1)
    
    return_vals = [points, speed]
    
    if args.use_no_coll_split:
        return_vals.append(no_coll_mask)
    return return_vals

def visual_env(out_dir, env_width, env_height, obstacles, robot_type, task_name):
    if task_name != 's2d_point':
        # Load the robot
        robot = np.load(os.path.join(out_dir, 'robot.npy'))
    else:
        robot = shapely.geometry.Point(0, 0)
    # Define the environment boundary
    half_env_width = env_width / 2
    half_env_height = env_height / 2
    origin = (-half_env_width, -half_env_height)
    boundary = shapely.geometry.Polygon([origin, (origin[0]+env_width, origin[1]), (origin[0]+env_width, origin[1]+env_height), (origin[0], origin[1]+env_height)])

    # Visualize the environment
    plt.figure(figsize=(12, 8))
    plt.plot(*boundary.exterior.xy, color='black', label='Boundary')

    for obstacle in obstacles:
        plt.fill(obstacle[0], obstacle[1], color='gray')


    if task_name != 's2d_point':
        if robot_type == 'cuboid':
            robot_corners = robot[-4:]
            robot_corners = np.append(robot_corners, robot_corners[:1], axis=0)
            plt.plot(robot_corners[:, 0], robot_corners[:, 1], color='blue', label='Robot')
        else:
            plt.plot(robot[:, 0], robot[:, 1], color='blue', label='Robot')
    else:
        plt.scatter(*robot.xy, color='blue', label='Robot', s=100)
    # plt.scatter(*start.xy, color='green', label='Start', s=100)
    # plt.scatter(*goal.xy, color='red', label='Goal', s=100)
    plt.legend(loc='best')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Complex 2D Environment with Non-Overlapping Rectangular Obstacles')
    plt.grid(True)
    plt.axis('equal')

    # Save as a .png file
    plt.savefig(os.path.join(out_dir, 'env.png'), dpi=300, bbox_inches='tight')
    # plt.show()

# Visualize the speed
def visual_speed(out_dir, points, speed, pos='start', scale=None):
    plt.figure()
    
    if scale is not None:
        scatter = plt.scatter(points[:, 0], points[:, 1], c=speed, cmap='viridis', s=1, vmin=scale[0], vmax=scale[1])
    else:
        scatter = plt.scatter(points[:, 0], points[:, 1], c=speed, cmap='viridis', s=1)
    plt.colorbar(scatter, label='speed')
    
    file_path = os.path.join(out_dir, f'{pos}_speed.png')
    plt.savefig(file_path, bbox_inches='tight')
    print(f'Saved to {file_path}')
    
def visual_split(out_dir, points, split, pos='start'):
    plt.figure()
    
    scatter = plt.scatter(points[:, 0], points[:, 1], c=split, cmap='viridis', s=1)
    plt.colorbar(scatter, label='split')
    
    plt.savefig(os.path.join(out_dir, f'{pos}_split.png'), bbox_inches='tight')
    
def get_obstacles(data_dir, load_env, num_obstacles, env_width, env_height):
    if load_env:
        obstacles, env_width, env_height = read_hdf5(data_dir, 'env', ['obstacles', 'env_w', 'env_h'])
        env_width = env_width.item()
        env_height = env_height.item()
    else:
        obstacles = gen_2d_map(num_obstacles, env_width, env_height)
        obstacles = np.array([obs.exterior.xy for obs in obstacles])

    env_bound = shapely.geometry.Polygon([[-env_width/2, -env_height/2], [-env_width/2, env_height/2], [env_width/2, env_height/2], [env_width/2, -env_height/2]])
    obstacles = np.concatenate((obstacles, np.array(env_bound.exterior.xy)[np.newaxis,...]), axis=0)
    
    return obstacles, env_width, env_height

def gen_2d_samples(out_dir, numsamples, dof, num_obstacles, env_width, env_height, args):

    obstacles, env_width, env_height = get_obstacles(args.data_dir, args.load_env, num_obstacles, env_width, env_height)
    
    limit = 0.5 * env_width
    velocity_max = 1
    
    margin = (limit / 5) / args.margin_div
    offset = margin / 10 
    # margin = limit / 20
    # offset = margin / 10
     
    args.margin = margin
    args.offset = offset
    
    start = time.time()
    
    r = sample_2d_points_and_speed(numsamples, dof, obstacles, env_width, env_height, margin, offset, velocity_max, out_dir, args.task_name, args.random_loc, args, filter_start=not args.no_filter_start)
    if args.use_no_coll_split:
        points, speed, no_coll_mask = r
    else:
        points, speed = r
    end = time.time()

    print("==> Data generation done, time: {:.6f} s".format(end-start))
    
    print('points shape:', points.shape)
    print('speed shape:', speed.shape)
    
    obstacles = obstacles[:-1, ...]
    
    print('Visualize the environment.')
    visual_env(out_dir, env_width, env_height, obstacles, args.robot, args.task_name)

    write_hdf5(out_dir, ['obstacles', 'env_w', 'env_h'], [obstacles, env_width, env_height], file_name='env')
    
    start_points = points[:, :dof]
    goal_points = points[:, dof:]
    start_speed = speed[:, 0]
    goal_speed = speed[:, 1]
    
    print('Visualize the speed.')
    visual_speed(out_dir, start_points, start_speed, 'start')
    visual_speed(out_dir, goal_points, goal_speed, 'goal')

    write_hdf5(out_dir, ['start_points', 'goal_points', 'start_speed', 'goal_speed'], [start_points, goal_points, start_speed, goal_speed])
    
    if args.use_no_coll_split:
        print('Visualize the split.')
        visual_split(out_dir, start_points, no_coll_mask.astype(int), 'start')
        visual_split(out_dir, goal_points, no_coll_mask.astype(int), 'goal')
        append_hdf5(out_dir, ['no_coll_mask'], [no_coll_mask])

    print('Data saved in {}'.format(out_dir))   

    
    