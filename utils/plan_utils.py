import torch
import os 
import numpy as np
import torch


import open3d as o3d
import os
import time
import matplotlib.pyplot as plt
import shapely
import imageio.v2 as imageio
import io
from scipy.interpolate import CubicSpline
import matplotlib as mpl
from utils.util import read_hdf5
from utils.utils_3d import project_mesh_to_plane
from copy import deepcopy

def gen_gif(args):
    # Set parameters
    vis_dir = os.path.join(args.exp_dir, 'vis_train_plot', args.plane)
    gif_dir = os.path.join(args.exp_dir, 'gif', args.plane)
    os.makedirs(gif_dir, exist_ok=True)
    output_gif_path = os.path.join(gif_dir, f'speed_epoch_0-{args.epoch}.gif')
    

    # Get all image files in the directory
    image_files = [f for f in os.listdir(vis_dir) if f.endswith('.jpg') and f.startswith('plots')]
    
    # filter image_files with epoch range
    image_files = [f for f in image_files if int(f.split('_')[0][5:]) in list(range(0, args.epoch + 1, 10)) + [0]]
    

    # Sort the image files to ensure they appear in the GIF in the desired order
    image_files.sort(key=lambda x: int(x.split('_')[0][5:]))

    # Create an image sequence and save it as a GIF
    images = []

    for image_file in image_files:
        image_path = os.path.join(vis_dir, image_file)
        image = imageio.imread(image_path)
        images.append(image)
    
    images2gif(images, output_gif_path, fps=5)

def images2gif(images, gif_path, fps=5, pause_duration=5000, format='gif'):
    """
    images: list of np.array, each array is a RGB image
    gif_path: str, output file path
    fps: int, frames per second
    pause_duration: int, pause duration before looping (ms)
    """
    # frame_duration = 200  # Duration of each frame in ms
    frame_duration = 1000 / fps  # Duration of each frame in ms
    # pause_duration = 5000  # Pause duration before looping (ms)
    
    # Calculate the number of pause frames needed
    pause_frames = int(pause_duration / frame_duration)
    print(f'Pausing for {pause_duration} ms ({pause_frames} frames)')
    
    last_frame = images[-1]
    # Add pause frames (duplicate the last frame)
    for _ in range(pause_frames):
        images.append(last_frame)  # Repeat the last frame to create a pause

    print(f'Totally {len(images)} frames')
    
    if format == 'gif':
        # Save the image sequence as a GIF
        imageio.mimsave(gif_path, images, duration=frame_duration, loop=0)
    elif format == 'mp4':
        imageio.mimsave(gif_path, images, fps=fps)

    print(f'GIF saved as {gif_path}')

def get_fix_dim_and_v(data_dir):
    fix_dim, fix_dim_v = None, None
    try:
        fix_dim, fix_dim_v = read_hdf5(data_dir, data_names=['fix_dim', 'fix_dim_v'])
        print('fix_dim:', fix_dim)
        print('fix_dim_v:', fix_dim_v)
        fix_dim, fix_dim_v = fix_dim[0], fix_dim_v[0]
    except Exception:
        print('no fix dim')
    return fix_dim, fix_dim_v

def append_fix_v(data_dir, points):
    fix_dim, fix_v = get_fix_dim_and_v(data_dir)
    if fix_dim is not None:
        points = add_dim_val(points, fix_dim, fix_v)
    return points


def SE2_render(obstacles, boundary, xyz, is_point, vis_track, epoch, pos_enc, fig_vis_dir, video_vis_dir=None, vis_frames=False, state_points=None):
    """
    xyz: (n, sample_size, dof)
    """
    # plt.figure(figsize=(12, 8))
    plt.figure(figsize=(8, 8))
    # plt.plot(*boundary.exterior.xy, color='black', label='boundary')
    plt.plot(*boundary.exterior.xy, color='#878787', label='boundary')

    for obstacle in obstacles:
        # plt.fill(obstacle[0], obstacle[1], color='gray')
        plt.fill(obstacle[0], obstacle[1], color='#BEBEBE')
    
    
    if not is_point:
        for i in range(xyz.shape[0]):
            if i == 0:
                # plt.plot(xyz[i][:, 0], xyz[i][:, 1], color='red', label='start', linewidth=2)
                plt.plot(xyz[i][:, 0], xyz[i][:, 1], color='#e5989b', label='start', linewidth=2)
                # plt.scatter(xyz[i][:, 0], xyz[i][:, 1], color='red', label='start', s=20)
            elif i == xyz.shape[0] - 1:
                # plt.plot(xyz[i][:, 0], xyz[i][:, 1], color='green', label='goal', linewidth=2)
                plt.plot(xyz[i][:, 0], xyz[i][:, 1], color='#0096c7', label='goal', linewidth=2)
                # plt.scatter(xyz[i][:, 0], xyz[i][:, 1], color='green', label='goal', s=20)
            elif i < xyz.shape[0] // 2:
                # plt.plot(xyz[i][:, 0], xyz[i][:, 1], color='#fec5bb') # ! start - xx
                plt.plot(xyz[i][:, 0], xyz[i][:, 1], color='#fcd5ce') # ! start - xx
            else:
                # plt.plot(xyz[i][:, 0], xyz[i][:, 1], color='blue')
                plt.plot(xyz[i][:, 0], xyz[i][:, 1], color='#bde0fe') #! xx - goal
                # plt.scatter(xyz[i][:, 0], xyz[i][:, 1], color='blue', s=10)
            # plt.scatter(xyz[i, 0, 0], xyz[i, 0, 1], color='#52b788', s=60, alpha=0.6)
    else:
        plt.scatter(xyz[:, 0], xyz[:, 1], color='blue', s=20)

    plt.legend(loc=(0.76,0.84))
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.grid(True)
    # plt.axis('equal')
    plt.axis('off')

    plt.savefig(os.path.join(fig_vis_dir, "path_epoch_{}_{}.png".format(epoch, pos_enc)), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print('==> Path saved to {}'.format(os.path.join(fig_vis_dir, "path_epoch_{}_{}.png".format(epoch, pos_enc))))
    
    if vis_track or vis_frames:
        assert(video_vis_dir is not None)
        # tmp_dir = 'tmp'
        # os.makedirs(tmp_dir, exist_ok=True)
        if vis_frames:
            frame_dir = os.path.join(video_vis_dir, 'frames', f'epoch_{epoch}_' + '_'.join(list(map(lambda x: '{:.3f}'.format(x), np.concatenate((state_points[0], state_points[-1]))))))
            os.makedirs(frame_dir, exist_ok=True)
        images = []
        start = time.time()

        for i in range(xyz.shape[0]):
            plt.figure(figsize=(12, 8))
            plt.plot(*boundary.exterior.xy, color='black', label='boundary')
            for obstacle in obstacles:
                plt.fill(obstacle[0], obstacle[1], color='gray')
            
            if not is_point:
                plt.plot(xyz[i, :, 0], xyz[i, :, 1], color='blue')
                plt.scatter(xyz[i, 0, 0], xyz[i, 0, 1], color='#52b788', s=60, alpha=0.6)
            else:
                plt.scatter(xyz[i, 0], xyz[i, 1], color='blue', s=20)
            
            plt.legend(loc='upper right')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(True)
            plt.axis('equal')
            
            if vis_frames:
                frame_path = os.path.join(frame_dir, '{}_{}.png'.format(f'{i:0>4d}', '_'.join(list(map(lambda x: '{:.3f}'.format(x), state_points[i].flatten())))))
                plt.title('ID: {}, Points: [{}]'.format(f'{i:0>4d}', ', '.join(list(map(lambda x: '{:.3f}'.format(x), state_points[i].flatten())))))
                plt.savefig(frame_path, dpi=300, bbox_inches='tight')
                print('==> Frame {} saved to {}'.format(i, frame_path))
            if vis_track:
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                buffer.seek(0)
                images.append(imageio.imread(buffer))
            
            plt.close()

        # os.system(f"ffmpeg -r 10 -i {tmp_dir}/%d.png -vcodec mpeg4 -y {os.path.join(video_vis_dir, 'track.mp4')}")
        
        # for i in range(xyz.shape[0]):
        #     images.append(imageio.imread(os.path.join(tmp_dir, f"{i}.png")))
        # shutil.rmtree(tmp_dir)
        
        print('images collecting done, time: {:.2f} ms'.format(time.time() - start))
        
        if vis_track:
            file_path = os.path.join(video_vis_dir, 'track_epoch_{}_{}.gif'.format(epoch, pos_enc))
            images2gif(images, file_path, fps=10)
            images2gif(images, os.path.join(video_vis_dir, 'track_epoch_{}_{}.mp4'.format(epoch, pos_enc)), fps=10, pause_duration=1000, format='mp4')
            
            print('==> Track saved to {}'.format(file_path))

def project_to_2d(xyz, mesh_path, fix_dim_v, is_point, vis_track, epoch, pos_enc, fig_vis_dir, video_vis_dir=None, vis_frames=False, state_points=None, scale=None):
    """
    xyz: (n, sample_size, dof)
    """
    # plt.figure(figsize=(12, 8))
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    mesh_proj_points = project_mesh_to_plane(mesh_path, plane_normal=[0, 0, 1], plane_point=[0, 0, fix_dim_v], mesh_scale=scale) 
    
    # for i in range(mesh_proj_points.shape[0]):
    #     ax.plot(mesh_proj_points[i, :, 0], mesh_proj_points[i, :, 1], color='black')
    ax.plot(mesh_proj_points[:, :, 0].T, mesh_proj_points[:, :, 1].T, color='black')
    
    new_fig, new_ax = deepcopy((fig, ax))
    if not is_point:
        for i in range(xyz.shape[0]):
            if i == 0:
                # plt.plot(xyz[i][:, 0], xyz[i][:, 1], color='red', label='start', linewidth=2)
                new_ax.plot(xyz[i][:, 0], xyz[i][:, 1], color='red', label='start', linewidth=2)
                # plt.scatter(xyz[i][:, 0], xyz[i][:, 1], color='red', label='start', s=20)
            elif i == xyz.shape[0] - 1:
                # plt.plot(xyz[i][:, 0], xyz[i][:, 1], color='green', label='goal', linewidth=2)
                new_ax.plot(xyz[i][:, 0], xyz[i][:, 1], color='green', label='goal', linewidth=2)
                # plt.scatter(xyz[i][:, 0], xyz[i][:, 1], color='green', label='goal', s=20)
            elif i < xyz.shape[0] // 2:
                # plt.plot(xyz[i][:, 0], xyz[i][:, 1], color='purple')
                new_ax.plot(xyz[i][:, 0], xyz[i][:, 1], color='purple')
            else:
                # plt.plot(xyz[i][:, 0], xyz[i][:, 1], color='blue')
                new_ax.plot(xyz[i][:, 0], xyz[i][:, 1], color='blue')
                # plt.scatter(xyz[i][:, 0], xyz[i][:, 1], color='blue', s=10)
            # plt.scatter(xyz[i, 0, 0], xyz[i, 0, 1], color='#52b788', s=60, alpha=0.6)
            new_ax.scatter(xyz[i, 0, 0], xyz[i, 0, 1], color='#52b788', s=60, alpha=0.6)
    else:
        # plt.scatter(xyz[:, 0], xyz[:, 1], color='blue', s=20)
        new_ax.scatter(xyz[:, 0], xyz[:, 1], color='blue', s=20)

    new_ax.legend(loc='best')
    new_ax.set_xlabel('X')
    new_ax.set_xlabel('Y')
    new_ax.grid(True)
    new_ax.axis('equal')

    new_fig.savefig(os.path.join(fig_vis_dir, "path_epoch_{}_{}.png".format(epoch, pos_enc)), dpi=300, bbox_inches='tight')
    plt.close(new_fig)
    print('==> Path saved to {}'.format(os.path.join(fig_vis_dir, "path_epoch_{}_{}.png".format(epoch, pos_enc))))
    
    if vis_track or vis_frames:
        assert(video_vis_dir is not None)
        # tmp_dir = 'tmp'
        # os.makedirs(tmp_dir, exist_ok=True)
        if vis_frames:
            frame_dir = os.path.join(video_vis_dir, 'frames', f'epoch_{epoch}_' + '_'.join(list(map(lambda x: '{:.3f}'.format(x), np.concatenate((state_points[0], state_points[-1]))))))
            os.makedirs(frame_dir, exist_ok=True)
        images = []
        start = time.time()

        for i in range(xyz.shape[0]):
            # plt.figure(figsize=(12, 8))
            
            new_fig, new_ax = deepcopy((fig, ax))
            
            if not is_point:
                new_ax.plot(xyz[i, :, 0], xyz[i, :, 1], color='blue')
                new_ax.scatter(xyz[i, 0, 0], xyz[i, 0, 1], color='#52b788', s=60, alpha=0.6)
            else:
                new_ax.scatter(xyz[i, 0], xyz[i, 1], color='blue', s=20)
            
            # new_ax.legend(loc='upper right')
            new_ax.set_xlabel('X')
            new_ax.set_ylabel('Y')
            new_ax.grid(True)
            new_ax.axis('equal')
            
            if vis_frames:
                frame_path = os.path.join(frame_dir, '{}_{}.png'.format(f'{i:0>4d}', '_'.join(list(map(lambda x: '{:.3f}'.format(x), state_points[i].flatten())))))
                ax.set_title('ID: {}, Points: [{}]'.format(f'{i:0>4d}', ', '.join(list(map(lambda x: '{:.3f}'.format(x), state_points[i].flatten())))))
                new_fig.savefig(frame_path, dpi=300, bbox_inches='tight')
                print('==> Frame {} saved to {}'.format(i, frame_path))
            if vis_track:
                buffer = io.BytesIO()
                new_fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                buffer.seek(0)
                images.append(imageio.imread(buffer))
            
            plt.close(new_fig)

        # os.system(f"ffmpeg -r 10 -i {tmp_dir}/%d.png -vcodec mpeg4 -y {os.path.join(video_vis_dir, 'track.mp4')}")
        
        # for i in range(xyz.shape[0]):
        #     images.append(imageio.imread(os.path.join(tmp_dir, f"{i}.png")))
        # shutil.rmtree(tmp_dir)
        
        print('images collecting done, time: {:.2f} ms'.format(time.time() - start))
        
        if vis_track:
            file_path = os.path.join(video_vis_dir, 'track_epoch_{}_{}.gif'.format(epoch, pos_enc))
            images2gif(images, file_path, fps=10)
            images2gif(images, os.path.join(video_vis_dir, 'track_epoch_{}_{}.mp4'.format(epoch, pos_enc)), fps=10, pause_duration=1000, format='mp4')
            
            print('==> Track saved to {}'.format(file_path))
    plt.close(fig)

def save_3d(xyz, centers, env, is_point, vis_dir, epoch, pos_enc, scale=None):
    """
    xyz: (n, sample_size, 3) [or (n, 3) for point]
    centers: (n, 3)
    """
    if is_point:
        gradient_colors = mpl.colormaps["jet"](np.linspace(0, 1, xyz.shape[0]))[:, :3]
        colors = gradient_colors
    else:
        # gradient_colors = np.linspace([0, 0, 1], [1, 0, 0], xyz.shape[0])[:, np.newaxis, :]
        gradient_colors = mpl.colormaps["jet"](np.linspace(0, 1, xyz.shape[0]))[:, np.newaxis, :3]
        
        colors = np.tile(gradient_colors, (1, xyz.shape[1], 1))
    
    xyz = xyz.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    mesh = o3d.io.read_triangle_mesh(env)
    
    if scale is not None:
        v = np.asarray(mesh.vertices)
        v = v * scale
        mesh.vertices = o3d.utility.Vector3dVector(v)
        mesh_file = 'room_scale_back.ply'
    else:
        mesh_file = 'room.ply'

    # mesh.scale(20, center=(0,0,0))

    mesh.compute_vertex_normals()

    # pcd.paint_uniform_color([0, 0.706, 1])
    
    o3d.io.write_triangle_mesh(os.path.join(vis_dir, mesh_file), mesh, write_ascii=True)
    file_path = os.path.join(vis_dir, "path_epoch_{}_{}.ply".format(epoch, pos_enc))
    o3d.io.write_point_cloud(file_path, pcd, write_ascii=True)
    print("Path saved to {}".format(file_path))

    if not is_point:
        pcd_c = o3d.geometry.PointCloud()
        pcd_c.points = o3d.utility.Vector3dVector(centers)
        pcd_c.paint_uniform_color([1, 1, 0])
        file_path = os.path.join(vis_dir, "center_path_epoch_{}_{}.ply".format(epoch, pos_enc))
        o3d.io.write_point_cloud(file_path, pcd_c, write_ascii=True)
        print(f"Center path saved to {file_path}")

    # o3d.visualization.draw_geometries([mesh,pcd])

def plot_single_val(vals, x_label, y_label, title, file_path):
    x = np.arange(len(vals))
    y = np.array(vals)
    
    plt.figure()
    plt.scatter(x, y, color='#bc4749')
    if x.shape[0] > 1:
        # Use cubic spline interpolation
        cs = CubicSpline(x, y)
    
        # Generate new x values for plotting the curve
        x_fit = np.linspace(min(x), max(x), 100)
    
        # Calculate corresponding y values using the cubic spline
        y_fit = cs(x_fit)
    
        # Plot interpolated curve
        plt.plot(x_fit, y_fit, label='Interpolated Curve', color='#0077b6')
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    plt.legend()
    plt.grid(True)
    
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def visual_eik_time(vis_dir, eik_times, epoch, pos_enc):
    file_path = os.path.join(vis_dir, "time_epoch_{}_{}.png".format(epoch, pos_enc))
    plot_single_val(eik_times, 'Steps', 'Time', 'Planning stepped eikonal time', file_path)
    print('==> Time saved to {}'.format(file_path))

def visual_speed(vis_dir, speeds, epoch, pos_enc):
    file_path = os.path.join(vis_dir, "speed_epoch_{}_{}.png".format(epoch, pos_enc))
    plot_single_val(speeds, 'Steps', 'Speed', 'Planning speed', file_path)
    print('==> Speed saved to {}'.format(file_path))

def calc_path_length(points):
    """
    points: [n, dof]
    """
    dist = torch.linalg.norm(points[1:] - points[:-1], dim=1)
    return torch.sum(dist)

def calc_safe_margin(points,get_dist_fn):
    """
    points: [n, dof]
    """
    dists = get_dist_fn(points)
    ave_margin = torch.mean(dists)

    return ave_margin
                                                                                                      


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

def remove_dim_val(x, dim):
    """
    args:
        x: [..., dof]
        dim: int
    return:
        x: [..., dof-1]
    """
    x = torch.cat((x[..., :dim], x[..., dim+1:]), dim=-1)
    return x

def add_dim_val_np(x, dim, val):
    """
    x: [..., dof]
    dim: int
    v: float
    """
    x = np.insert(x, dim, val, axis=-1)
    return x

def get_bbox(data_dir):
    bbox = None
    try:
        bbox = read_hdf5(data_dir, data_names=['bbox'])[0]
        print('bbox:', bbox)
    except Exception as e:
        print('no bbox find!')
    return bbox

def get_ori_bbox(data_dir):
    bbox = None
    try:
        bbox = read_hdf5(data_dir, data_names=['ori_bbox'])[0]
        print('ori bbox:', bbox)
    except Exception as e:
        print('ori no bbox find!')
    return bbox