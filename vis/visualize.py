import sys
sys.path.append('.')
import argparse
import matplotlib.pylab as plt
import open3d as o3d
import numpy as np
import os
from scipy.interpolate import griddata
import matplotlib as mpl

from utils.util import read_hdf5


def visual_space_label(args, start_goal, deprecated=False):
    if deprecated:
        sample_points = np.load(os.path.join(args.data_dir, 'sampled_points.npy'))
        label = np.load(os.path.join(args.data_dir, 'space_label.npy'))
        sample_points = sample_points[:, start_goal * args.dof: (start_goal + 1) * args.dof]
        label = label[:, start_goal]
    else:
        start_split, goal_split, start_points, goal_points = read_hdf5(args.data_dir, data_names=['start_split', 'goal_split', 'start_points', 'goal_points'])
        if start_goal == 0: # start
            label = start_split
            sample_points = start_points
        else:
            label = goal_split
            sample_points = goal_points
    print(np.where(label == True))
    print(np.all(~label))
    print(np.sum(label))
    
    print(label)
    label = label.astype(int)
    print(label)
    
    label = label / 2. + 0.1
    cmap = mpl.colormaps['viridis']  # Choose the colormap
    
    colors = cmap(label)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(sample_points[:, :3]) # Keep only xyz
    point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    label_category = 'start' if start_goal == 0 else 'goal'
    o3d.io.write_point_cloud(os.path.join(args.data_dir, f'{label_category}_space_label_cloud.ply'), point_cloud)


def visual_waypoints(data_dir, waypoints, use_waypoints):

    use_waypoints = use_waypoints.astype(int)
    use_waypoints = use_waypoints / 2. + 0.1
    
    # cmap = mpl.colormaps['viridis']  # Choose the colormap
    cmap = mpl.colormaps['Reds']  # Choose the colormap
    
    colors = cmap(use_waypoints)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(waypoints[:, :3]) # Keep only xyz
    point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    o3d.io.write_point_cloud(os.path.join(data_dir, 'waypoints_cloud.ply'), point_cloud)
    print('waypoints saved in {}'.format(os.path.join(data_dir, 'waypoints_cloud.ply')))

def visual_space_multilabel(data_dir, dof, start_goal, deprecated=False):
    if deprecated:
        sample_points = np.load(os.path.join(data_dir, 'sampled_points.npy'))
        label = np.load(os.path.join(data_dir, 'space_label.npy'))
        
        sample_points = sample_points[:, start_goal * dof: (start_goal + 1) * dof]
        label = label[:, start_goal]
    else:
        start_split, goal_split, start_points, goal_points = read_hdf5(data_dir, data_names=['start_split', 'goal_split', 'start_points', 'goal_points'])
        if start_goal == 0: # start
            label = start_split
            sample_points = start_points
        else:
            label = goal_split
            sample_points = goal_points

    label = (label - np.min(label)) / (np.max(label) - np.min(label))
    
    cmap = mpl.colormaps['viridis']  # Choose the colormap
    
    colors = cmap(label)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(sample_points[:, :3]) # Keep only xyz
    point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    label_category = 'start' if start_goal == 0 else 'goal'
    o3d.io.write_point_cloud(os.path.join(data_dir, f'{label_category}_space_multilabel_cloud.ply'), point_cloud)
    print('split space point cloud with multi-label saved in {}'.format(os.path.join(data_dir, f'{label_category}_space_multilabel_cloud.ply')))

def visual_space_labels(data_dir, points, label, pos='start'):
    label = (label - np.min(label)) / (np.max(label) - np.min(label))
    
    cmap = mpl.colormaps['viridis']  # Choose the colormap
    
    colors = cmap(label)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points[:, :3]) # Keep only xyz
    point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    file_path = os.path.join(data_dir, f'{pos}_space_label_cloud.ply')
    o3d.io.write_point_cloud(file_path, point_cloud)
    print('split space point cloud with multi-label saved in {}'.format(file_path))

def visual_dists(data_dir, points, dists, pos='start', scale=None):
    if scale is not None:
        dists = mpl.colors.Normalize(vmin=scale[0], vmax=scale[1])(dists)
    else:
        dists = (dists - np.min(dists)) / (np.max(dists) - np.min(dists))
    
    cmap = mpl.colormaps['viridis']  # Choose the colormap

    colors = cmap(dists)
    print(colors.shape)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points[:, :3]) # Keep only xyz
    point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    file_path = os.path.join(data_dir, f'{pos}_dists_cloud.ply')
    o3d.io.write_point_cloud(file_path, point_cloud)
    print('Dists point cloud saved in {}'.format(file_path))

def visual_point_speed(data_dir, points, speed, pos='start'):
    
    print(points.shape)
    
    speed = (speed - np.min(speed)) / (np.max(speed) - np.min(speed)) # Convert the speed to a range between 0 and 1 to be compatible with the color map
    
    color_map = mpl.colormaps['viridis']  # Choose a color map
    colors = color_map(speed)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points[:, :3]) # Keep only xyz
    point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3]) # Keep only RGB values, discard alpha channel if present
    
    file_path = os.path.join(data_dir, f'{pos}_speed_cloud.ply')
    o3d.io.write_point_cloud(file_path, point_cloud)
    print('speed point cloud saved in {}'.format(file_path))

def visual_start_goal(args, plane, start_goal):
    speed = np.load(os.path.join(args.data_dir, 'speed.npy'))
    sample_points = np.load(os.path.join(args.data_dir, 'sampled_points.npy'))
    
    pos = [-0.25, 0, 0]
    if args.dof == 4:
        pos.append(0)
    offset = 0.05
    zmin = -0.01
    zmax = 0.01
    # zmin = 0.29
    # zmax = 0.31
    # zmin = -0.41
    # zmax = -0.40
    theta = 1/360
    base = start_goal * args.dof
    if args.dof == 4:
        theta_constrain =(sample_points[:, base + 3] > pos[3] - theta) & (sample_points[:, base + 3] < pos[3] + theta)
    if plane == 'xy':
        if args.dof == 4:
            idx = np.argwhere((sample_points[:, base + 2] < pos[2] + offset) & (sample_points[:, base + 2] > pos[2] - offset) & theta_constrain).squeeze()
        else:
            idx = np.argwhere((sample_points[:, base + 2] < pos[2] + offset) & (sample_points[:, base + 2] > pos[2] - offset)).squeeze()
        # idx = np.argwhere((sample_points[:, 2] < zmax) & (sample_points[:, 2] > zmin)).squeeze()
    elif plane == 'xz':
        if args.dof == 4:
            idx = np.argwhere((sample_points[:, base + 1] < pos[1] + offset) & (sample_points[:, base + 1] > pos[1] - offset) & theta_constrain).squeeze()
        else:
            idx = np.argwhere((sample_points[:, base + 1] < pos[1] + offset) & (sample_points[:, base + 1] > pos[1] - offset)).squeeze()
    elif plane == 'yz':
        if args.dof == 4:
            idx = np.argwhere((sample_points[:, base] < pos[0] + offset) & (sample_points[:, base] > pos[0] - offset) & theta_constrain).squeeze()
        else:
            idx = np.argwhere((sample_points[:, base] < pos[0] + offset) & (sample_points[:, base] > pos[0] - offset)).squeeze()
    else:
        raise NotImplementedError

    print('idx shape: {}'.format(idx.shape))
    sample_points = sample_points[idx, :]
    speed = speed[idx, :]


    print('speed shape: {}'.format(speed.shape))
    print('sample_points shape: {}'.format(sample_points.shape))
    limit = 0.5
    xmin     = [-limit,-limit]
    xmax     = [limit,limit]
    spacing  = limit/40.0
    X,Y      = np.meshgrid(np.arange(xmin[0],xmax[0],spacing),np.arange(xmin[1],xmax[1],spacing))
    print('X shape: {}'.format(X.shape))

    # interpolate the data onto the grid
    if plane == 'xy':
        Z = griddata((sample_points[:, base], sample_points[:, base + 1]), speed[:, start_goal], (X, Y), method='cubic')
    elif plane == 'xz':
        Z = griddata((sample_points[:, base], sample_points[:, base + 2]), speed[:, start_goal], (X, Y), method='cubic')
    elif plane == 'yz':
        Z = griddata((sample_points[:, base + 2], sample_points[:, base + 1]), speed[:, start_goal], (X, Y), method='cubic')
    print('Z shape: {}'.format(Z.shape))



    fig = plt.figure()
    ax = fig.add_subplot(111)
    quad1 = ax.pcolormesh(X,Y,Z,vmin=0,vmax=1)
    # ax.contour(X,Y,TT,np.arange(0,3,0.05), cmap='bone', linewidths=0.5)#0.25
    plt.colorbar(quad1,ax=ax, pad=0.1, label='Predicted Velocity')

    # plot the interpolated data as a heatmap
    # heatmap = ax.imshow(Z, extent=[xmin[0], xmax[0], xmin[1], xmax[1]], origin='lower', cmap='jet')

    # add a colorbar legend
    # plt.colorbar(heatmap, ax=ax)

    # show the plot
    # plt.show()
    speed_category = 'start' if start_goal == 0 else 'goal'
    plt.savefig(f"gt_{plane}_{speed_category}.jpg",bbox_inches='tight')


def parse_args():
    parser = argparse.ArgumentParser(description='Complex Model Navigation visialization')
    # parser.add_argument('--exp_name', type=str, required=True, help='experiment name and part of directory to save logs')
    parser.add_argument('--data_dir', type=str, default='./datasets/gibson/0', help='directory to load data')
    parser.add_argument('--dof', type=int, default=4, help = 'degree of freedom')
    
    return parser.parse_args()

def demo():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    # create a sample dataset of 3D points with values
    data = np.random.rand(100, 4)

    # project the 3D point cloud onto the xy plane
    data[:, 2] = 0

    # define the grid for the interpolation
    xi = np.linspace(data[:, 0].min(), data[:, 0].max(), 100)
    yi = np.linspace(data[:, 1].min(), data[:, 1].max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    # interpolate the data onto the grid
    zi = griddata((data[:, 0], data[:, 1]), data[:, 3], (xi, yi), method='cubic')

    # create a 2D plot object
    fig, ax = plt.subplots()

    # plot the interpolated data as a heatmap
    heatmap = ax.imshow(zi, extent=[data[:, 0].min(), data[:, 0].max(), data[:, 1].min(), data[:, 1].max()], origin='lower', cmap='jet')

    # add a colorbar legend
    plt.colorbar(heatmap,ax=ax)

    # set the labels for the x and y axes
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')

    # show the plot
    plt.show()





def main():
    args = parse_args()

    # demo()
    # visual_start_goal(args, 'xy', 0)
    # visual_start_goal(args, 'xz', 0)
    # visual_start_goal(args, 'xy', 1)
    # visual_start_goal(args, 'xz', 1)
    # visual_point_speed(args.data_dir, args.dof, 0)
    # visual_point_speed(args.data_dir, args.dof, 1)
    from dataprocessing.gen_2d_map_and_samples_pytorch import visual_waypoints
    env_width, env_height, obstacles = read_hdf5(args.data_dir, 'env', data_names=['env_w', 'env_h', 'obstacles'])
    start_points, goal_points, waypoints, use_waypoint_mask = read_hdf5(args.data_dir, data_names=['start_points', 'goal_points', 'waypoints', 'use_waypoint_mask'])
    print('use waypoint true number: {}'.format(np.sum(use_waypoint_mask)))
    
    wp = waypoints[:, 0, 0]
    wp_left = np.sum((wp < -0.1) & (wp > -0.3) & use_waypoint_mask)
    wp_right = np.sum((wp < 0.3) & (wp > 0.1) & use_waypoint_mask)
    print('left: {}, right: {}'.format(wp_left, wp_right))
    visual_waypoints(args.data_dir, args.dof, env_width, env_height, obstacles, start_points, goal_points, waypoints[:, 0, :], use_waypoint_mask, 100)
    

if __name__ == '__main__':
    main()