import open3d as o3d
import numpy as np
import argparse
import os

def sample_points_of_obj(in_obj_path, out_pcd_path, num_points=1024, normalize=False, normalize_size=1):
    # Read the OBJ file
    mesh = o3d.io.read_triangle_mesh(in_obj_path)
    
    # Sample points from the mesh
    # pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    pcd = mesh.sample_points_poisson_disk(num_points)
    
    if normalize:
        bbox = mesh.get_axis_aligned_bounding_box()
        scale = 1. / np.max(np.array(bbox.get_max_bound()) - np.array(bbox.get_min_bound())).item() * normalize_size
        pcd.scale(scale=scale, center=mesh.get_center())
        pcd.translate(-mesh.get_center())
    
    # Save the sampled points
    o3d.io.write_point_cloud(out_pcd_path, pcd)
    
    print(f'Saved to {out_pcd_path}')
    
    
def parse_args():
    parser = argparse.ArgumentParser(description='Sample points from an OBJ file')
    parser.add_argument('--in_obj_path', type=str, help='The input OBJ file path')
    parser.add_argument('--out_pcd_dir', type=str, help='The output point cloud file directory')
    parser.add_argument('--num_points', type=int, default=1024, help='The number of points to sample')
    parser.add_argument('--normalize', action='store_true', help='Whether to normalize the point cloud')
    parser.add_argument('--normalize_size', type=float, default=1, help='The size to normalize the point cloud')
    
    return parser.parse_args()

def main():
    args = parse_args()
    out_pcd_dir = os.path.join(args.out_pcd_dir, str(args.num_points))
    os.makedirs(out_pcd_dir, exist_ok=True)
    out_pcd_path = os.path.join(out_pcd_dir, os.path.splitext(os.path.basename(args.in_obj_path))[0] + '.ply')
    sample_points_of_obj(args.in_obj_path, out_pcd_path, args.num_points, args.normalize, args.normalize_size)

if __name__ == '__main__':
    main()