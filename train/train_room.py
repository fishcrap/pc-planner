import os
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
import sys
sys.path.append('.')
from models import model_3d_room_refactor as md_refactor
from utils.util import get_logger
import os
import shutil
import time
from utils.util import seed_everything, dump_train_config
import warnings
from config_parser.base_config_parser import train_parse_args

def _init(args):
    global output_dir
    output = 'output'
    exp_dir = os.path.join(output, args.exp_name)
    args.exp_dir = exp_dir
    args.log_dir = os.path.join(exp_dir, 'logs')
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, 'train-{}.log'.format(time.strftime("%Y%m%d_%H%M%S", time.localtime())))
    backup_dir = os.path.join(exp_dir, 'backup')
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(backup_dir, exist_ok=True)
    os.makedirs(os.path.join(backup_dir, 'models'), exist_ok=True)
    shutil.copy(__file__, os.path.join(backup_dir, os.path.basename(__file__) + '.backup'))
    
    py_files = [f for f in os.listdir('models') if f.endswith(".py")]

    for py_file in py_files:
        src_path = os.path.join('models', py_file)
        dst_path = os.path.join(os.path.join(backup_dir, 'models'), py_file + '.backup')
        shutil.copy(src_path, dst_path)
    global logger
    logger = get_logger(log_path, console=True)
    
def check_args(args):
    if args.resume:
        assert(args.resume_epoch > 0)
        
    if args.SE2:
        if not args.no_enc_env:
            args.no_enc_env = True
            warnings.warn('SE2 do not use env_encoding, already changing no_enc_env to True!', UserWarning)

def main():
    args = train_parse_args()
    
    _init(args)
    
    check_args(args)
    
    logger.info(args)
    
    if args.fix_seed:
        seed_everything(1)
    
    dump_train_config(args)
    
    start_time = time.time()

    # pos for the initial position for time field visualization, you could set in free space
    if args.SE2:
        pos = [0, 0]
        if args.dof == 3:
            pos.append(0)
    else:
        pos = [-0.3,-0.2, 0]
    
    if args.dof == 4:
        pos.append(0)
    elif args.dof == 6:
        pos.extend([0,0,0])

    model = md_refactor.Model(args.exp_dir, args.data_dir, args.dof, pos, logger=logger, device='cuda:0', args=args)

    model.train()
    
    train_time = (time.time() - start_time) / 60 ** 2
    logger.info(f"==> Exp {args.exp_dir} finished!")
    logger.info(f"==> Training done! Time: {train_time:.2f}h")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception(f'main exception: {str(e)}')
