import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()

    # Experiment Setup
    parser.add_argument('--config', is_config_file=True, default='configs/shapenet_cars.txt',
                        help='config file path')
    parser.add_argument("--exp_name", type=str, default=None,
                        help='Experiment name, used as folder name for the experiment. If left blank, a \
                         name will be auto generated based on the configuration settings.')
    parser.add_argument("--data_dir", type=str,
                        help='input data directory')
    parser.add_argument("--out_data_dir", type=str, default=None,
                        help='output data directory')
    parser.add_argument("--task_name", type=str, default='arm',
                        help='task[arm, gibson, ...]')
    parser.add_argument("--input_data_glob", type=str,
                        help='glob expression to find raw input files')
    # Training Data Parameters
    parser.add_argument("--num_samples", type=int, default=1e6,
                        help='Number of start-goal pairs sampled in space.')
    parser.add_argument("--num_dim", type=int, default=3,
                        help='Number of dimension for configuration space.')

    parser.add_argument("--bb_min", default=-0.5, type=float,
                        help='Training and testing shapes are normalized to be in a common bounding box.\
                             This value defines the min value in x,y and z for the bounding box.')
    parser.add_argument("--bb_max", default=0.5, type=float,
                        help='Training and testing shapes are normalized to be in a common bounding box.\
                             This value defines the max value in x,y and z for the bounding box.')
    
    
    parser.add_argument('--robot', type=str, default=None, choices=['cuboid', 'line', 'triangle', 'line_x', 'cuboid_o_corner', 'circle', 'custom', 'load', 'point'], help='robot type')
    parser.add_argument('--dense_robot', action='store_true', help='robot dense or not')
    parser.add_argument('--line_length', type=float, default=0.1, help='line length')
    parser.add_argument('--robot_samples', type=int, default=10, help='number of robot samples')
    parser.add_argument('--robot_samples_cls', type=int, default=120, help='number of robot samples for cls')
    parser.add_argument('--check_colls', action='store_true', help='check collisions')
    parser.add_argument('--colls_prob', type=float, default=0.6, help='collision probability(low means more feasible points)')
    parser.add_argument('--speed_model', type=str, default='linear', choices=['linear', 'exp', 'log', 'exp_inv'], help='speed model')
    parser.add_argument('--rotate_axis', type=str, default='x', choices=['x', 'y', 'z'], help='rotate axis')
    parser.add_argument('--margin_div', type=int, default=1, help='margin div')
    
    parser.add_argument('--gen_2d', action='store_true', help='generate 2d samples')
    parser.add_argument('--num_obs', type=int, default=10, help='number of obstacles')
    parser.add_argument('--no_filter_start', action='store_true', help='do not filter start points according to margin and limit')
    parser.add_argument('--load_env', action='store_true', help='use exist environment')
    parser.add_argument('--env_width', type=float, default=1, help='environment width')
    parser.add_argument('--env_height', type=float, default=1, help='environment height')
    parser.add_argument('--robot_direction', type=str, default='y', choices=['x', 'y'], help='robot direction')
    parser.add_argument('--solid', action='store_true', help='solid obstacles')
    parser.add_argument('--use_no_coll_split', action='store_true', help='use no collision split')

    parser.add_argument('--random_loc', action='store_true', help='random location')
    parser.add_argument('--fix_seed', action='store_true', help='fix random seed')

    parser.add_argument('--robot_path', type=str, default='./robots/piano/piano.obj', help='robot mesh path')
    parser.add_argument('--robot_sparse_path', type=str, default=None, help='sparse robot path, if None, equal to robot_path')
    parser.add_argument('--no_scale', action='store_true', help='do not scale the env')
    parser.add_argument('--fix_dim', nargs='+', type=int, default=None, help='fix dimension [xyz->012] [for example, fix z-axis for SE2 in 3d env]')
    parser.add_argument('--fix_dim_v', nargs='+', type=float, default=None, help='fix z-axis value [for example, select z=1]')
    
    parser.add_argument('--dist_fn_type', type=str, default='acc', choices=['acc', 'udf', 'obj_udf', 'obj_gen_udf'], help='dist func type, default is accurate, like bvh')
    parser.add_argument('--in_dim', type=int, default=3, help='in_dim for udf')
    parser.add_argument('--h_dim', type=int, default=128, help='hidden dim for udf')
    parser.add_argument('--ckpt_path', type=str, default=None, help='udf ckpt path')
    
    parser.add_argument('--filter_outer', action='store_true', help='filter the points outside the env, which means filter points whose sdf < 0')
    parser.add_argument('--ref_obj', type=str, default='mesh_z_up.obj', help='ref object path for filtering points outside the env')
    
    parser.add_argument('--split_path', type=str, default='obj_env_udf/processed_shapenet/piano/splits.json', help='train test split path')
    parser.add_argument('--scale_robot',  action='store_true', help='scale robot with env')
    parser.add_argument('--random_scale', action='store_true', help='random scaling for udf training data')
    
    parser.add_argument('--sim', action='store_true', help='use sim arm if robot is arm')
    
    parser.add_argument('--vmax', type=float, default=1, help='max speed')
    
    return parser


def check_config(cfg):
    if cfg.fix_dim is not None and cfg.fix_dim_v is not None:
        assert(len(cfg.fix_dim) == len(cfg.fix_dim_v))

    if cfg.robot_samples != 10:
        t = cfg.robot_samples_cls
        cfg.robot_samples_cls = cfg.robot_samples * 12
        print(f"!!!!!!!!!!! change robot_sample_cls from {t} to {cfg.robot_samples_cls} (=robot_samples*12) automatically!!!!!!!!!!!")
    
    if cfg.robot_sparse_path is None:
        cfg.robot_sparse_path = cfg.robot_path

def get_config():
    parser = config_parser()
    cfg = parser.parse_args()
    
    if cfg.out_data_dir is None:
        cfg.out_data_dir = cfg.data_dir
    
    check_config(cfg)

    return cfg
