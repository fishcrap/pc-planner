import os
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
import matplotlib.pylab as plt
import open3d as o3d
import glob
import random
import json
import numpy as np

import argparse

def parse_args():
    
    parser = argparse.ArgumentParser(description='Data process')
    parser.add_argument('--model_dir', type=str, default=None, help='model directory')
    parser.add_argument('--ori_model_dir', type=str, default=None, help='original model directory')
    parser.add_argument('--sample_single', action='store_true', help='sample single model')
    parser.add_argument('--remove_single', action='store_true', help='remove texture of single model')
    parser.add_argument('--scale_single', action='store_true', help='scale single model')
    parser.add_argument('--in_dir', type=str, default=None, help='input directory')
    parser.add_argument('--out_dir', type=str, default=None, help='output directory')
    parser.add_argument('--sparse_n', type=int, default=128, help='sparse point number')
    parser.add_argument('--single_n', type=int, default=32, help='single sample num_points')
    parser.add_argument('--single_out_dir', type=str, default=None, help='single sample output directory')
    parser.add_argument('--scale', type=float, default=0.05, help='scale')
    parser.add_argument('--normalize', action='store_true', help='normalize')

    return parser.parse_args()

def remove_texture_of_obj(input_obj_path, output_obj_path):
    # Read the OBJ file
    mesh = o3d.io.read_triangle_mesh(input_obj_path)

    # Clear the texture information
    mesh.textures = []

    # Save the processed model
    o3d.io.write_triangle_mesh(output_obj_path, mesh)
    

def remove_texture_of_obj_in_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    obj_dirs = os.listdir(input_dir)
    
    # Process all the OBJ files
    for obj_dir in obj_dirs:
        # Get the full directory of the input OBJ file
        in_obj_dir = os.path.join(input_dir, obj_dir, 'models')
        
        # Get the full directory of the output OBJ file
        out_obj_dir = os.path.join(output_dir, obj_dir, 'models')
        os.makedirs(out_obj_dir, exist_ok=True)
        
        in_obj_paths = glob.glob(os.path.join(in_obj_dir, '*.obj'))
        print(in_obj_paths)
        for in_obj_path in in_obj_paths:
            out_obj_path = os.path.join(out_obj_dir, os.path.basename(in_obj_path))
        
            # Remove the texture from the input OBJ file and save the result to the output OBJ file
            remove_texture_of_obj(in_obj_path, out_obj_path)
    
    print('input_dir:', input_dir)
    print('output_dir:', output_dir)
    print(f'processed {len(obj_dirs)} obj files')
    

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

def sample_points_of_obj_in_folder(in_dir, sparse=False, num_points=1024, normalize=False, normalize_size=1):
    
    
    obj_dirs = os.listdir(in_dir)
    
    obj_dirs = [f for f in obj_dirs if os.path.isdir(os.path.join(in_dir, f))]
    
    name = 'sparse_points' if sparse else 'points'
    
    # Process all the OBJ files
    for obj_dir in obj_dirs:
        full_obj_dir = os.path.join(in_dir, obj_dir)
        # if not os.path.isdir(full_obj_dir):
        #     continue
        # Get the full directory of the input OBJ file
        in_obj_dir = os.path.join(full_obj_dir, 'models')
        
        # Get the full directory of the output OBJ file
        out_pcd_dir = os.path.join(full_obj_dir, name)
        os.makedirs(out_pcd_dir, exist_ok=True)
        
        in_obj_paths = glob.glob(os.path.join(in_obj_dir, '*.obj'))
        print(in_obj_paths)
        for in_obj_path in in_obj_paths:
            out_pcd_path = os.path.join(out_pcd_dir, os.path.splitext(os.path.basename(in_obj_path))[0] + '.ply')
        
            # sample points of the objects
            sample_points_of_obj(in_obj_path, out_pcd_path, num_points, normalize, normalize_size)
    
    print('obj_dir:', in_dir)
    print(f'processed {len(obj_dirs)} obj files')

def train_test_split(obj_dir, train_ratio=0.7):
    all_objs = os.listdir(obj_dir)
    all_objs = [f for f in all_objs if os.path.isdir(os.path.join(obj_dir, f))]
    random.shuffle(all_objs)
    
    num_train = int(len(all_objs) * train_ratio)
    
    train_objs = all_objs[:num_train]
    test_objs = all_objs[num_train:]
    
    data = {'train': train_objs, 'test': test_objs}
    json_path = os.path.join(obj_dir, 'splits.json')
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f'Saved to {json_path}')
    

def sample_points_of_obj_seperate(in_dir, dir_name, num_points=1024, normalize=False, normalize_size=1):
    obj_dir = os.path.join(in_dir, 'models')
    out_pcd_dir = os.path.join(in_dir, dir_name)
    os.makedirs(out_pcd_dir, exist_ok=True)
    
    obj_paths = glob.glob(os.path.join(obj_dir, '*.obj'))
    print(obj_paths)

    for obj_path in obj_paths:
        out_pcd_path = os.path.join(out_pcd_dir, os.path.splitext(os.path.basename(obj_path))[0] + '.ply')

        # sample points of the objects
        sample_points_of_obj(obj_path, out_pcd_path, num_points, normalize, normalize_size)
        

def scale_obj(in_obj_path, out_obj_path, normalize=False, normalize_size=1):
    # Read the OBJ file
    mesh = o3d.io.read_triangle_mesh(in_obj_path)
    
    if normalize:
        bbox = mesh.get_axis_aligned_bounding_box()
        scale = 1. / np.max(np.array(bbox.get_max_bound()) - np.array(bbox.get_min_bound())).item() * normalize_size
        mesh.scale(scale=scale, center=mesh.get_center())
        mesh.translate(-mesh.get_center())
        rotation_angle = np.pi / 2.0
        rotation_matrix = np.array([[1, 0, 0],
                            [0, np.cos(rotation_angle), -np.sin(rotation_angle)],
                            [0, np.sin(rotation_angle), np.cos(rotation_angle)]])
        mesh.rotate(rotation_matrix, mesh.get_center())
    
    # bbox = np.array([[-9.13167,  -0.538764, -0.028483],
    #         [ 0.625705,  5.90462,   2.29166 ]])
    
    
    # rotation_angle = np.pi / 2.0
    # rotation_matrix = np.array([[1, 0, 0],
    #                     [0, np.cos(rotation_angle), -np.sin(rotation_angle)],
    #                     [0, np.sin(rotation_angle), np.cos(rotation_angle)]])
    # mesh.rotate(rotation_matrix, mesh.get_center())
    
    # v = np.asarray(mesh.vertices)
    # scale = bbox[1] - bbox[0]
    # scale[2] = scale[1]
    # v *= scale * 0.03
    # mesh.vertices = o3d.utility.Vector3dVector(v)
    # Save the sampled points
    o3d.io.write_triangle_mesh(out_obj_path, mesh)
    
    print(f'Saved to {out_obj_path}')

def scale_obj_seperate(in_dir, dir_name, normalize=False, normalize_size=1):
    in_obj_dir = os.path.join(in_dir, 'models')
    out_obj_dir = os.path.join(in_dir, dir_name)
    os.makedirs(out_obj_dir, exist_ok=True)
    
    obj_paths = glob.glob(os.path.join(in_obj_dir, '*.obj'))
    print(obj_paths)

    for obj_path in obj_paths:
        out_obj_path = os.path.join(out_obj_dir, os.path.splitext(os.path.basename(obj_path))[0] + '.obj')

        # sample points of the objects
        scale_obj(obj_path, out_obj_path, normalize, normalize_size)

def remove_texture_of_obj_seperate(in_dir, out_dir):
    obj_dir = os.path.join(in_dir, 'models')
    out_obj_dir = os.path.join(out_dir, 'models')
    os.makedirs(out_obj_dir, exist_ok=True)
    
    obj_paths = glob.glob(os.path.join(obj_dir, '*.obj'))
    print(obj_paths)

    for obj_path in obj_paths:
        out_obj_path = os.path.join(out_obj_dir, os.path.basename(obj_path))
        
        print('out:', out_obj_path)

        # Remove the texture from the input OBJ file and save the result to the output OBJ file
        remove_texture_of_obj(obj_path, out_obj_path)

def main():
    args = parse_args()
        
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
    
    if args.remove_single:
        remove_texture_of_obj_seperate(args.ori_model_dir, args.model_dir)
        return
    
    if args.scale_single:
        scale_obj_seperate(args.model_dir, args.single_out_dir, normalize=args.normalize, normalize_size=args.scale)
        return
    
    if args.sample_single:

        sample_points_of_obj_seperate(args.model_dir, args.single_out_dir, num_points=args.single_n, normalize=args.normalize, normalize_size=args.scale)
        return
    
    in_dir = args.in_dir
    out_dir = args.out_dir
    # in_dir = '/mnt/nas_8/datasets/ShapeNetCore.v2/03928116'
    # out_dir = 'obj_env_udf/processed_shapenet/piano_normalize_0.05'
    # in_dir = '/mnt/nas_8/datasets/ShapeNetCore.v2/02958343'
    # out_dir = 'obj_env_udf/processed_shapenet/car_normalize_0.1'
    
    # Remove the texture from the input OBJ file and save the result to the output OBJ file
    remove_texture_of_obj_in_folder(in_dir, out_dir)
    
    train_test_split(out_dir)
    
    normalize = True
    normalize_size = args.scale
    
    sample_points_of_obj_in_folder(out_dir, normalize=normalize, normalize_size=normalize_size)
    sample_points_of_obj_in_folder(out_dir, sparse=True, num_points=args.sparse_n, normalize=normalize, normalize_size=normalize_size)

if __name__ == '__main__':
    main()