import os
import matplotlib.pylab as plt
import open3d as o3d
import numpy as np
from matplotlib import cm
import argparse

def parse_args():
    
    parser = argparse.ArgumentParser(description='Data process')
    parser.add_argument('--mesh_path', type=str, default=None, help='model directory')
    parser.add_argument('--n_points', type=int, default=128, help='point number')
    parser.add_argument('--out_dir', type=str, default='obj_env_udf/colored_pc', help='output directory')
    parser.add_argument('--name', type=str, default='output', help='output name')

    return parser.parse_args()

def main():

    args = parse_args()
    
    radius = 0.01  # 搜索半径
    max_nn = 30  # 邻域内用于估算法线的最大点数
    
    mesh = o3d.io.read_triangle_mesh(args.mesh_path)
    num_points = args.n_points
    # point_cloud = mesh.sample_points_uniformly(number_of_points=num_points)
    point_cloud = mesh.sample_points_poisson_disk(number_of_points=num_points)
    # o3d.geometry.estimate_normals(point_cloud)
    # point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))
    point_cloud.estimate_normals()

    # normals = np.asarray(point_cloud.normals)
    # print(normals)
    # normals_rgb = (normals + 1) / 2  # 将法线映射到[0, 1]范围
    # normals_rgb = (normals_rgb * 255).astype(np.uint8)  # 将映射到[0, 255]范围
    # normals_rgb = normals_rgb / 255.0
    
    x_coords = np.asarray(point_cloud.points)[:, 1]

    # 映射X坐标到Jet colormap的0到1范围
    jet_colors = cm.jet((x_coords - np.min(x_coords)) / (np.max(x_coords) - np.min(x_coords)))

    theta = -np.pi / 6
    rot_mat = np.array(
        [[1, 0, 0],
         [0, np.cos(theta), -np.sin(theta)],
         [0, np.sin(theta), np.cos(theta)]]
    )
    
    rot_mat = np.array(
        [[np.cos(theta), 0, np.sin(theta)],
         [0, 1, 0],
         [-np.sin(theta), 0, np.cos(theta)]]
    )
    
    point_cloud.rotate(rot_mat, center=(0, 0, 0))

    color_point_cloud = o3d.geometry.PointCloud()
    color_point_cloud.points = point_cloud.points
    color_point_cloud.colors = o3d.utility.Vector3dVector(jet_colors[:, :3])

    os.makedirs(args.out_dir, exist_ok=True)
    output_ply_path = os.path.join(args.out_dir, f"{args.name}.ply")
    o3d.io.write_point_cloud(output_ply_path, color_point_cloud)
    np.save(output_ply_path.replace('.ply', '.npy'), np.array(point_cloud.points))
    print(f'==> Save to {output_ply_path}')
    

if __name__ == '__main__':
    main()