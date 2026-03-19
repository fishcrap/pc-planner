import argparse

def __get_network_cfg_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--exp_name', type=str, required=True, help='experiment name and part of directory to save logs')
    """network"""
    parser.add_argument('--data_dir', type=str, default='./datasets/room', help='directory to load data')
    parser.add_argument('--dof', type=int, default=4, help = 'degree of freedom')
    parser.add_argument('--SE2', action='store_true', help='(x, y theta)')
    parser.add_argument('--point', action='store_true', help='use point for navigation')
    parser.add_argument('--basic_block', type=str, default='resnet', choices=['resnet', 'resnext', 'unet'], help='basic block type')
    parser.add_argument('--enc_d', type=int, default=7,  help='encoder dimension') # 5 for ntfields
    parser.add_argument('--dec_d', type=int, default=7, help='decoder dimension')
    parser.add_argument('--use_unet', action='store_true', help='use unet')
    parser.add_argument('--h_size', type=int, default=256, help='hidden size for MLP')
    parser.add_argument('--no_enc_env', action='store_true', default=True, help='do not encode environment explicitly')
    parser.add_argument('--pos_enc', action='store_true', help='postional encoding')
    parser.add_argument('--full_coords_enc', action='store_true', help='use full coordinates for encoding => f(x, y) (instead of f(x)) and f(y, x) for symmetry')
    parser.add_argument('--symm_op_type', type=str, default='min_max', choices=['mean', 'min_max', 'f'], help='symmetry operation type')
    parser.add_argument('--vmax', type=float, default=1, help='max velocity')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='device to use')
    return parser

def __get_trainer_cfg_parser():
    parser = argparse.ArgumentParser(add_help=False)

    """basic training"""
    parser.add_argument('--batch_size', type=int, default=10000, help='batch size')
    parser.add_argument('--sampler', type=str, default='batch_weighted', choices=['weighted', 'random', 'batch_weighted', 'dynamic_weighted'], help='sampler type')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--resume_epoch', type=int, default=1, help='resume epoch')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--sched', type=str, choices=['cosine', 'custom'], default='custom', help='lr scheduler')
    parser.add_argument('--loss_ratio', type=float, default=1.2, help='loss ratio bounding')
    parser.add_argument('--data_format', type=str, choices=['h5', 'npy'], default='h5', help='data format')
    parser.add_argument('--fix_seed', action='store_true', help='fix random seed')
    parser.add_argument('--start_epoch_loss_val', type=float, default=1.2, help='start epoch loss val')
    
    """Exploratory byproduct. Maybe not useful"""
    parser.add_argument('--record_loss', type=bool, default=True, help='record loss')
    
    """path loss"""
    parser.add_argument('--use_path_loss', action='store_true', help='generate path waypoint when training, use path loss(T == Ts + Tg)')
    parser.add_argument('--one_step', action='store_true', help='generate path with one step')
    parser.add_argument('--beta', type=float, default=0.2, help='coeffient for path loss T <= Ts + Tg')
    parser.add_argument('--beta1', type=float, default=0.2, help='coeffient for path loss T > T_new')
    parser.add_argument('--beta2', type=float, default=0.2, help='coeffient for path loss T > Ts')
    parser.add_argument('--beta3', type=float, default=0.2, help='coeffient for path loss T > Tg')
    parser.add_argument('--beta5', type=float, default=0, help='coeffient for path loss T < T_coll_s')
    parser.add_argument('--beta6', type=float, default=0, help='coeffient for path loss T < T_coll_g')
    parser.add_argument('--use_no_coll_split', action='store_true', help='use no collision split')
    parser.add_argument('--path_loss_start_epoch', type=int, default=50, help='path loss start epoch')
    
    """halmiton loss, maybe not work"""
    parser.add_argument('--speed_loss', type=str, default='sq', choices=['sq', 'hamilton'], help='speed loss type')
    parser.add_argument('--hamilton_p', type=float, default=1, help='hamiltonian p')
    
    """for dynamic weighted sampler"""
    parser.add_argument('--dynmaic_epoch', type=int, default=500, help='dynamic epoch to change the weights of sampler')

    """visualization scale"""
    parser.add_argument('--env_w', type=float, default=1.0, help='environment width')
    parser.add_argument('--env_h', type=float, default=1.0, help='environment height')

    return parser
    
def __get_tester_cfg_parser():
    parser = argparse.ArgumentParser(add_help=False)
    """for test"""
    parser.add_argument('--epoch', type=int, default=500, help = 'epoch for pretrained model')
    return parser

def __get_coll_check_cfg_parser():
    parser = argparse.ArgumentParser(add_help=False)
    """collision checking"""
    parser.add_argument('--offset', type=float, default=0.01, help='offset for collision checking')
    parser.add_argument('--solid', action='store_true', help='solid obstacles for collision checking')
    parser.add_argument('--rotate_axis', type=str, default='z', choices=['x', 'y', 'z'], help='rotate axis')
    parser.add_argument('--robot_name', type=str, default='robot', choices=['arm', 'point', 'robot'], help='robot name')
    parser.add_argument('--sim', action='store_true', help='use sim arm if robot is arm')
    
    """udf args for collision checking"""
    parser.add_argument('--dist_fn_type', type=str, default='acc', choices=['acc', 'udf', 'obj_udf', 'obj_gen_udf', 'obj_gen_udf_baseline'], help='dist func type, default is accurate, like bvh')
    parser.add_argument('--in_dim', type=int, default=3, help='in_dim for udf')
    parser.add_argument('--h_dim', type=int, default=128, help='hidden dim for udf')
    parser.add_argument('--ckpt_path', type=str, default=None, help='udf ckpt path')
    
    return parser

def train_parse_args():
    trainer_parser = __get_trainer_cfg_parser()
    network_parser = __get_network_cfg_parser()
    coll_parser = __get_coll_check_cfg_parser()
    
    parser = argparse.ArgumentParser(description='Complex Model Navigation', parents=[trainer_parser, network_parser, coll_parser])
    return parser.parse_args()

def test_parse_args():
    tester_parser = __get_tester_cfg_parser()
    network_parser = __get_network_cfg_parser()
    coll_parser = __get_coll_check_cfg_parser()
    
    parser = argparse.ArgumentParser(description='Complex Model Navigation', parents=[tester_parser, network_parser, coll_parser])
    """geodesic distance"""
    # parser.add_argument('--geodesic', action='store_true', help='plan geodesic distance')
    parser.add_argument('--dist_tol', type=float, default=0.06, help='distance tolerance')
    """visualization"""
    parser.add_argument('--vis_track', action='store_true', default=False, help='whether to visualize track')
    parser.add_argument('--step', type=float, default=0.03, help='step for planning') # step 0.0001, max_iters 10000
    parser.add_argument('--max_iters', type=int, default=500, help='max iters for planning')
    parser.add_argument('--vis_eik_time', action='store_true', help='whether to visualize eikonal time')
    parser.add_argument('--vis_frames', action='store_true', help='whether to visualize frames')

    parser.add_argument('--spacing', type=float, default=0.001, help='spacing for collision checking')

    parser.add_argument('--scale_env', action='store_true', help='whether to scale environment back')
    parser.add_argument('--adaptive_planning', action='store_true', help='perform adaptive planning')
    parser.add_argument('--coll_robot_dir', type=str, default=None, help='robot dir for check collision')
    parser.add_argument('--no_load_ckpt', action='store_true', help='no load ckpt')
    return parser.parse_args()

def succ_parse_args():
    tester_parser = __get_tester_cfg_parser()
    network_parser = __get_network_cfg_parser()
    coll_parser = __get_coll_check_cfg_parser()
    
    parser = argparse.ArgumentParser(description='Complex Model Navigation', parents=[tester_parser, network_parser, coll_parser])
    parser.add_argument('--test_points_n', type=int, default=10000, help='number of test points')
    parser.add_argument('--load_test_points', action='store_true', help='load test points')
    parser.add_argument('--spacing', type=float, default=0.001, help='spacing for collision checking')
    parser.add_argument('--more_acc_check', action='store_true', 
                        help='pass start_goal into network one by one, avoid the loss of accuracy due to the matrix calculation')
    parser.add_argument('--test_points_dir', type=str, default=None, help='test points directory')
    parser.add_argument('--test_points_path', type=str, default=None, help='test points path')
    parser.add_argument('--vis_failure', action='store_true', help='visualize failure cases')
    parser.add_argument('--vis_succ', action='store_true', help='visualize successful cases')
    parser.add_argument('--scale_env', action='store_true', help='whether to scale environment back')   
    """adaptive planning args"""
    parser.add_argument('--vis_adaptive', action='store_true', help='visualize adapative cases')
    parser.add_argument('--dist_fn_type_for_adaptive', type=str, default='acc', choices=['acc', 'udf', 'obj_udf', 'obj_gen_udf'], help='dist func type, default is accurate, like bvh')
    parser.add_argument('--adaptive_planning', action='store_true', help='perform adaptive planning')

    parser.add_argument('--vis_eik_time', action='store_true', help='whether to visualize eikonal time')
    parser.add_argument('--grid_size', type=int, default=100, help='size of grid')
    parser.add_argument('--succ_n', type=int, default=100, help='number of success cases')
    parser.add_argument('--coll_robot_dir', type=str, default=None, help='robot dir for check collision')
    parser.add_argument('--ref_data_dir', type=str, default=None, help='reference data dir for check collision')

    return parser.parse_args()

def __get_genudf_network_cfg_parser():
    parser = argparse.ArgumentParser(add_help=False)
    """network"""
    parser.add_argument('--exp_name', type=str, required=True, help='experiment name and part of directory to save logs')
    parser.add_argument('--dim', type=int, default=3, help='dimension of the space')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='device to use')
    parser.add_argument('--h_dim', type=int, default=128, help='hidden dimension')
    return parser

def genudf_train_parse_args():
    network_parser = __get_genudf_network_cfg_parser()
    parser = argparse.ArgumentParser(description='genudf', parents=[network_parser])
    """train"""
    parser.add_argument('--data_dir', type=str, default='./datasets/room', help='directory to load data')
    parser.add_argument('--batch_size', type=int, default=10000, help='batch size')
    parser.add_argument('--epochs', type=int, default=2500, help='number of epochs')
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--resume_epoch', type=int, default=1, help='resume epoch')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--fix_seed', action='store_true', help='fix random seed')
    parser.add_argument('--save_freq', type=int, default=10, help='checkpoint save frequecy')
    
    parser.add_argument('--load_pretrain_enc', action='store_true', help='load pretrained model encoder')
    parser.add_argument('--pretrain_path', type=str, default='obj_env_udf/pretrain_ckpts/gensdf.ckpt', help='pretrained model ckpt path')
    
    parser.add_argument('--alpha', type=float, default=1, help='alpha for dist loss')
    parser.add_argument('--net_type', type=str, default='genudf', choices=['genudf', 'baseline', 'genudf_no_dist'], help='network type')
    
    return parser.parse_args()

def genudf_test_parse_args():
    network_parser = __get_genudf_network_cfg_parser()
    parser = argparse.ArgumentParser(description='genudf', parents=[network_parser])
    parser.add_argument('--epoch', type=int, default=500, help = 'epoch for pretrained model')
    parser.add_argument('--data_dir', type=str, default='./datasets/room', help='directory to load data')
    parser.add_argument('--test_data_dir', type=str, default=None, help='directory to load test data')
    parser.add_argument('--load_test', action='store_true', help='load test case from specific directory')
    parser.add_argument('--load_test_path', type=str, default=None, help='load test case from specific path')
    parser.add_argument('--test_time', action='store_true', help='test time')

    return parser.parse_args()

def genudf_eval_parse_args():
    network_parser = __get_genudf_network_cfg_parser()
    parser = argparse.ArgumentParser(description='genudf', parents=[network_parser])
    parser.add_argument('--epoch', type=int, default=500, help = 'epoch for pretrained model')
    parser.add_argument('--data_dir', type=str, default='./datasets/room', help='directory to load data')
    parser.add_argument('--test_data_dir', type=str, default=None, help='directory to load test data')
    parser.add_argument('--load_test', action='store_true', help='load test case from specific directory')
    parser.add_argument('--load_test_path', type=str, default=None, help='load test case from specific path')
    parser.add_argument('--robot_name', type=str, default='robot', choices=['arm', 'point', 'robot'], help='robot name')
    parser.add_argument('--sim', action='store_true', help='use sim arm if robot is arm')
    parser.add_argument('--rotate_axis', type=str, default='z', choices=['x', 'y', 'z'], help='rotate axis')
    parser.add_argument('--dof', type=int, default=4, help = 'degree of freedom')
    parser.add_argument('--ckpt_path', type=str, default=None, help='udf ckpt path')

    return parser.parse_args()