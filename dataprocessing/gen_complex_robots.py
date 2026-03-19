import matplotlib.pylab as plt
import open3d as o3d
import numpy as np
import os
import json
import glob


def save_robot(dir, sampled_points, name='robot', dim=3):
    # Save sampled points
    if dim == 3:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(sampled_points)

        # o3d.visualization.draw_geometries([point_cloud])
        o3d.io.write_point_cloud(os.path.join(dir, f'{name}.ply'), point_cloud, write_ascii=True)
    
    np.save(os.path.join(dir, f'{name}.npy'), sampled_points)

def gen_line(dir, length=0.4, num_samples=10, direction='y', name='robot'):
    # num_samples = 10
    t = np.linspace(0, 1, num_samples)
    start_points = np.array([[0, 0, 0]])
    end_points = np.array([[0, length, 0]])
    sampled_points = start_points[:, np.newaxis, :] + t[np.newaxis, :, np.newaxis] * (end_points - start_points)[:, np.newaxis, :]
    sampled_points = sampled_points.reshape(-1, 3)
    sampled_points -= np.array([0, length / 2, 0])
    #print(sampled_points)
    if direction=='x':
        rotate = np.array(
            [[np.cos(np.pi/2), -np.sin(np.pi/2),0],
            [np.sin(np.pi/2), np.cos(np.pi/2),0],
            [0,              0,              1]]
        )
        sampled_points = np.dot(rotate, sampled_points.T).T

    save_robot(dir, sampled_points, name=name)
    return sampled_points

def gen_line_2d(dir, length=0.4, num_samples=10, direction='y', name='robot'):
    t = np.linspace(0, 1, num_samples)
    start_points = np.array([[0, 0]])
    end_points = np.array([[0, length]])
    sampled_points = start_points[:, np.newaxis, :] + t[np.newaxis, :, np.newaxis] * (end_points - start_points)[:, np.newaxis, :]
    sampled_points = sampled_points.reshape(-1, 2)
    sampled_points -= np.array([0, length / 2])
    # print(sampled_points)
    
    if direction == 'x':
        rotate = np.array(
            [[np.cos(np.pi/2), -np.sin(np.pi/2)],
            [np.sin(np.pi/2), np.cos(np.pi/2)]]
        )
        
        sampled_points = np.dot(rotate, sampled_points.T).T

    save_robot(dir, sampled_points, name=name, dim=2)
    return sampled_points

def gen_circle_2d(dir, radius=0.4, num_samples=10, name='robot'):
    t = np.linspace(0, 2 * np.pi, num_samples)
    sampled_points = np.stack([radius * np.cos(t), radius * np.sin(t)], axis=1)
    # print(sampled_points)
    
    save_robot(dir, sampled_points, name=name, dim=2)
    return sampled_points

def gen_line_x(dir, length=0.4, num_samples = 10, name='robot'):
    t = np.linspace(0, 1, num_samples)
    start_points = np.array([[0, 0, 0]])
    end_points = np.array([[0, length, 0]])
    sampled_points = start_points[:, np.newaxis, :] + t[np.newaxis, :, np.newaxis] * (end_points - start_points)[:, np.newaxis, :]
    sampled_points = sampled_points.reshape(-1, 3)
    sampled_points -= np.array([0, length / 2, 0])
    # print(sampled_points)
    
    rotate = np.array(
        [[np.cos(np.pi/2), -np.sin(np.pi/2), 0],
         [np.sin(np.pi/2), np.cos(np.pi/2), 0],
         [0, 0, 1]]
    )
    
    sampled_points = np.dot(rotate, sampled_points.T).T

    save_robot(dir, sampled_points, name=name)
    return sampled_points


def gen_cuboid(dir, num_samples=10, name='robot'):
    # Define the size of the cuboid
    # size = [0.05, 0.4, 0.05]
    size = [0.8, 0.06, 0.06]
    # size = [0.1, 0.01, 0.01]

    # Create the cuboid mesh using create_box function
    cuboid_mesh = o3d.geometry.TriangleMesh.create_box(width=size[0], height=size[1], depth=size[2]) # x, y, z -> width, height, depth


    # Regenerate the cuboid
    corners = np.asarray(cuboid_mesh.vertices)
    faces = np.asarray(cuboid_mesh.triangles)

    # Get corner points of cuboid
    corners = np.asarray(cuboid_mesh.vertices)
    # print(corners)

    # Define indices of edges
    edges = np.array([[0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4], [0, 4], [1, 5], [2, 6], [3, 7]])

    # Sample points on edges
    # num_samples = 10
    t = np.linspace(0, 1, num_samples)[1:-1]
    start_points = corners[edges[:, 0]]
    end_points = corners[edges[:, 1]]
    sampled_points = start_points[:, np.newaxis, :] + t[np.newaxis, :, np.newaxis] * (end_points - start_points)[:, np.newaxis, :]

    # Add corner points to sample points
    sampled_points = np.concatenate((sampled_points.reshape(-1, 3), corners), axis=0)

    T = np.array(
        [[1, 0, 0, -size[0]/2],
         [0, 1, 0, -size[1]/2],
         [0, 0, 1, -size[2]/2],
         [0, 0, 0, 1]]
    )
    
    sampled_points = np.dot(np.concatenate((sampled_points, np.ones((sampled_points.shape[0], 1))), axis=1), T.T)[:, :3]

    save_robot(dir, sampled_points, name=name)
    return sampled_points


def gen_cuboid_2d(dir, num_samples = 10, name='robot', size=[0.05, 0.1]):

    # Regenerate the rectangle
    corners = np.asarray([[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]])

    # Define indices of edges
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])

    # Sample points on edges
    t = np.linspace(0, 1, num_samples)[1:-1]
    start_points = corners[edges[:, 0]]
    end_points = corners[edges[:, 1]]
    sampled_points = start_points[:, np.newaxis, :] + t[np.newaxis, :, np.newaxis] * (end_points - start_points)[:, np.newaxis, :]

    # Add corner points to sample points
    sampled_points = np.concatenate((sampled_points.reshape(-1, 2), corners), axis=0)
    sampled_points -= np.array([size[0] / 2, size[1] / 2])

    save_robot(dir, sampled_points, name=name, dim=2)
    return sampled_points

def gen_triangle_2d(dir, num_samples=10, name='robot', size=[0.1, 0.1]):

    # Regenerate the rectangle
    corners = np.asarray([[0, 0], [size[0], 0], [size[0]/2, size[1]/2]])

    # Define indices of edges
    edges = np.array([[0, 1], [1, 2], [2, 0]])

    # Sample points on edges
    t = np.linspace(0, 1, num_samples)[:-1]
    start_points = corners[edges[:, 0]] # (3, 2)
    end_points = corners[edges[:, 1]] # (3, 2)
    sampled_points = start_points[:, np.newaxis, :] + t[np.newaxis, :, np.newaxis] * (end_points - start_points)[:, np.newaxis, :] # (3, num_samples-2, 2)

    # Add corner points to sample points
    # sampled_points = np.concatenate((sampled_points.reshape(-1, 2), corners), axis=0)
    sampled_points = sampled_points.reshape(-1, 2)
    sampled_points = np.vstack((sampled_points, sampled_points[0]))
    sampled_points -= np.array([size[0] / 2, size[1] / 2])

    save_robot(dir, sampled_points, name=name, dim=2)
    return sampled_points
# the origin is at the corner point


def gen_cuboid_o_corer(dir, num_samples = 10, name='robot'):
    # Define the size of the cuboid
    size = [0.05, 0.4, 0.05]

    # Create the cuboid mesh using create_box function
    cuboid_mesh = o3d.geometry.TriangleMesh.create_box(width=size[0], height=size[1], depth=size[2]) # x, y, z -> width, height, depth


    # Regenerate the cuboid
    corners = np.asarray(cuboid_mesh.vertices)
    faces = np.asarray(cuboid_mesh.triangles)

    # Get corner points of cuboid
    corners = np.asarray(cuboid_mesh.vertices)
    # print(corners)

    # Define indices of edges
    edges = np.array([[0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4], [0, 4], [1, 5], [2, 6], [3, 7]])

    # Sample points on edges
    # num_samples = 10
    t = np.linspace(0, 1, num_samples)[1:-1]
    start_points = corners[edges[:, 0]]
    end_points = corners[edges[:, 1]]
    sampled_points = start_points[:, np.newaxis, :] + t[np.newaxis, :, np.newaxis] * (end_points - start_points)[:, np.newaxis, :]

    # Add corner points to sample points
    sampled_points = np.concatenate((sampled_points.reshape(-1, 3), corners), axis=0)

    save_robot(dir, sampled_points, name=name)
    return sampled_points

def sample_points_from_mesh(dir, mesh_path, num_samples, name='robot'):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    # points = mesh.sample_points_uniformly(num_samples)
    points = mesh.sample_points_poisson_disk(num_samples)
    points = np.asarray(points.points)
    save_robot(dir, points, name)
    return points

def load_points(pcd_path, dir=None, name='robot', rotate=False, bbox_length=None):
    pcd = o3d.io.read_point_cloud(os.path.join(pcd_path))  
    points = np.asarray(pcd.points)
    if rotate:
        rotate_m = np.array([
            [1, 0, 0],
            [0, np.cos(np.pi/2), -np.sin(np.pi/2)],
            [0, np.sin(np.pi/2), np.cos(np.pi/2)]
        ])
        points = np.dot(points, rotate_m.T)
    # p_bbox_length = np.max(points, axis=0) - np.min(points, axis=0)
    # print('prev robot bbox length', p_bbox_length)
    if bbox_length is not None:
        points /= bbox_length
    # p_bbox_length = np.max(points, axis=0) - np.min(points, axis=0)
    # print('robot bbox length', p_bbox_length)
    if dir is not None:
        save_robot(dir, points, name)
    return points

def load_robots(split_path, random_scale=False, scale_range=[0.1, 1]):
    robots = {}
    base_dir = os.path.dirname(split_path)
    with open(split_path, 'r') as f:
        json_data = json.load(f)
        for k in json_data: # train, test
            robots[k] = {'dense':[], 'sparse':[]}
            for sub_dir in json_data[k]:
                if random_scale:
                    scale = np.random.uniform(scale_range[0], scale_range[1])
                else:
                    scale = 1
                robot = load_points(glob.glob(os.path.join(base_dir, sub_dir, 'points', '*.ply'))[0])
                robot = robot * scale
                robots[k]['dense'].append(robot)
                robot = load_points(glob.glob(os.path.join(base_dir, sub_dir, 'sparse_points', '*.ply'))[0])
                robot = robot * scale
                robots[k]['sparse'].append(robot)
            robots[k]['dense'] = np.stack(robots[k]['dense'], axis=0)
            robots[k]['sparse'] = np.stack(robots[k]['sparse'], axis=0)
    return robots

def gen_point_2d(dir, name):
    point = np.array([[0, 0]])
    save_robot(dir, point, name=name, dim=2)
    return point

def gen_point(dir, name):
    point = np.array([[0, 0, 0]])
    save_robot(dir, point, name=name)
    return point

def gen_robots(dir, robot='cuboid', length=0.4, num_samples=10, name='robot', dim=3, direction='y', robot_path=None, **kwargs):
    if robot == 'cuboid':
        if dim == 2:
            return gen_cuboid_2d(dir, num_samples, name, size=[0.05, length])
        return gen_cuboid(dir, num_samples, name)
    elif robot == 'line':
        if dim == 2:
            return gen_line_2d(dir, length, num_samples, direction, name)
        return gen_line(dir, length, num_samples, direction, name)
    elif robot == 'triangle':
        if dim == 2:
            return gen_triangle_2d(dir, num_samples, name, size=[length, 0.1]) 
    elif robot == 'line_x':
        return gen_line_x(dir, length, num_samples, name)
    elif robot == 'cuboid_o_corner':
        return gen_cuboid_o_corer(dir, num_samples, name)
    elif robot == 'circle':
        return gen_circle_2d(dir, length, num_samples, name)
    elif robot == 'custom':
        return sample_points_from_mesh(dir, robot_path, num_samples, name)
    elif robot == 'load':
        load_kwargs = {}
        if kwargs.get('bbox_length', None) is not None:
            bbox_length = kwargs['bbox_length'].copy()
            # bbox_length[0], bbox_length[1] = bbox_length[1], bbox_length[0] # change x, y
            print('swapped bbox length', bbox_length)
            load_kwargs['bbox_length'] = bbox_length
        return load_points(robot_path, dir, name, rotate=False, **load_kwargs)
    elif robot == 'point':
        if dim == 2:
            gen_point_2d(dir, name)
        return gen_point(dir, name=name)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../vis_data')
    os.makedirs(dir, exist_ok=True)
    gen_robots(dir, 'line', dim=2)
    # gen_line(dir)