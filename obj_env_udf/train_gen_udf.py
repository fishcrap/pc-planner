import os
os.environ["MKL_NUM_THREADS"] = "4"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"
import sys
sys.path.append('.')

from obj_env_udf.models.model_gen_udf import trainer as genudf_trainer
from utils.util import get_logger
import os
import shutil
import time
import yaml

from utils.util import seed_everything, dump_train_config

import warnings

from config_parser.base_config_parser import genudf_train_parse_args

def _init(args):
    global output_dir
    output = 'output'
    exp_dir = os.path.join(output, 'robotudf', args.exp_name)
    args.exp_dir = exp_dir
    args.log_dir = os.path.join(exp_dir, 'logs')
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, 'train-{}.log'.format(time.strftime("%Y%m%d_%H%M%S", time.localtime())))
    backup_dir = os.path.join(exp_dir, 'backup')
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(backup_dir, exist_ok=True)
    model_dir = os.path.join('obj_env_udf', 'models')
    os.makedirs(os.path.join(backup_dir, model_dir), exist_ok=True)
    shutil.copy(__file__, os.path.join(backup_dir, os.path.basename(__file__) + '.backup'))
    
    py_files = [f for f in os.listdir(model_dir) if f.endswith(".py")]

    for py_file in py_files:
        src_path = os.path.join(model_dir, py_file)
        dst_path = os.path.join(os.path.join(backup_dir, model_dir), py_file + '.backup')
        shutil.copy(src_path, dst_path)
    global logger
    logger = get_logger(log_path, console=True)


def main():
    
    args = genudf_train_parse_args()
    
    _init(args)
    logger.info(args)
    
    if args.fix_seed:
        seed_everything(1)
    
    dump_train_config(args)
    
    start_time = time.time()

    trainer = genudf_trainer(args, logger)

    trainer.train()
    
    train_time = (time.time() - start_time) / 60 ** 2
    logger.info(f"==> Training done! Time: {train_time:.2f}h")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception(f'main exception: {str(e)}')
