# PC-Planner
> PC-Planner: Physics-Constrained Self-Supervised Learning for Robust Neural Motion Planning with Shape-Aware Distance Function
> 
> Xujie Shen\*, Haocheng Peng\*, Zesong Yang, Juzhan Xu, Hujun Bao, Ruizhen Hu, Zhaopeng Cui

[Project Page](https://zju3dv.github.io/pc-planner/) | [arXiv](https://arxiv.org/abs/2410.12805) | [Paper](https://raw.githubusercontent.com/fishcrap/open_access_assets/main/pc-planner/paper/pc-planner_24siga.pdf) | *Published in  SIGGRAPH Asia 2024*
![](https://raw.githubusercontent.com/fishcrap/open_access_assets/main/pc-planner/figs/pc-planner-logo.png)
## Introduction
This repository is the official implementation of "PC-Planner: Physics-Constrained Self-Supervised Learning for Robust Neural Motion Planning with Shape-Aware Distance Function".

## Installation
Tested on **Ubuntu 20.04, torch 1.10.2, CUDA 11.3, GCC 9.4.0**(Must match because of the setting of bvh)
```
conda env create -f pc_planner_env.yml
conda activate pc-planner
pip install -r requirements.txt
cd ./bvh-distance-queries
python setup.py install
pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.10.2_cu113.html
```
If you have trouble in installing bvh, maybe you could set as below:
```
export CUDA_HOME=/usr/local/cuda-11.3
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
export NVCC_CCBIN=/usr/bin/g++-9
```
Download datasets and pretrained models, exact and put datasets/ output/ to the repository directory:

[Datasets and pretrained models](https://drive.google.com/drive/folders/1RLqqRDLEFOIe4R0brtTpTKGFMuMgqiwm?usp=sharing)
## Prepare training data
### 2D Env
```
# test_2d.txt
exp_name = test_2d
data_dir = datasets/test_2d
num_dim = 3 # set 2 for [x, y], set 3 for [x, y, theta]
num_samples = 1000000
gen_2d = True
```
Prepare data for point with random scenario generation, run:
```
python dataprocessing/preprocess.py --config configs/test_2d.txt --robot point --solid --num_obs 10 --env_width 1 --env_height 1 --use_no_coll_split
```
Prepare data for shaped robot with random scenario generation, run:
```
python dataprocessing/preprocess.py --config configs/test_2d.txt --robot line --line_length 0.1 --robot_direction x --robot_samples 10 --num_obs 10 --env_width 1 --env_height 1 --use_no_coll_split
```
You can change the config [--robot] to use other type shaped robots.

Prepare data base on previous generated scenario, run:
```
python dataprocessing/preprocess.py --config configs/test_2d.txt --load_env --out_data_dir [destination_dir] --robot circle --line_length 0.05 --robot_samples 10 --num_obs 10 --env_width 1 --env_height 1 --use_no_coll_split
```
### 3D Env
```
# test_3d.txt
exp_name = test_3d
data_dir = datasets/test_3d
input_data_glob = /mesh_z_up_clip.obj # env mesh
num_dim = 4 # set 3 for [x, y, z], set 4 for [x, y, z, theta], usually fix dim z
num_samples = 1000000
```
In 3D settings, You need prepare scenario mesh, robot mesh for data generation.

Prepare data for custom robot, run:
```
python dataprocessing/preprocess.py --config configs/test_3d.txt --task_name gibson_complex --robot custom --rotate_axis z --robot_path datasets/test_3d/bear.obj --robot_samples 10 --robot_samples_cls 120 --filter_outer --ref_obj ref/mesh_z_up.obj --fix_dim 2 --fix_dim_v 0 --use_no_coll_split
```
The robot mesh bear.obj is normalized based on the scaled scenario, or you could load the already prepared sampled points bear.ply from mesh surface, run:
```
python dataprocessing/preprocess.py --config configs/test_3d.txt --task_name gibson_complex --robot load --rotate_axis z --robot_path /mnt/nas_9/group/zhouboyang/PC-Planner/datasets/test_3d/robot.ply --filter_outer --ref_obj ref/mesh_z_up.obj --fix_dim 2 --fix_dim_v 0 --use_no_coll_split
```
You can also use the simple robot by changing config [--robot], and setting the size rightly based on scaled scenario.

### Arm
```
# test_arm.txt
exp_name = test_arm
data_dir = datasets/test_arm/4DOF # datasets/test_arm/6DOF
input_data_glob = /model.obj
num_dim = 4
num_samples = 1000000
task_name = arm
```
Prepare data for arm, run:
```
python dataprocessing/preprocess.py --config configs/test_arm.txt --sim --use_no_coll_split
```
Config [--sim] only for test_arm/4DOF and test_arm/6DOF.
## Training
### 2D Env
To train model without pathloss, run:
```
python train/train_room.py --exp_name test_2d --data_dir datasets/test_2d --SE2 --dof 3 --loss_ratio 1.2 --h_size 256 --no_enc_env --sampler batch_weighted --enc_d 7 --dec_d 7 --start_epoch_loss_val 1.3 --use_no_coll_split --epoch 600 [--resume --resume_epoch 200]
```
To train model with pathloss, run:
```
python train/train_room.py --exp_name test_2d --data_dir datasets/test_2d --SE2 --dof 3 --loss_ratio 1.2 --h_size 256 --no_enc_env --sampler batch_weighted --enc_d 7 --dec_d 7 --start_epoch_loss_val 1.3 --use_no_coll_split --use_path_loss --path_loss_start_epoch 100 --beta 0 --beta1 0 --beta2 0.08 --beta3 0.08 --beta5 0 --beta6 0 --offset 0.01 --epoch 600 
```
You need to set the config [--offset] based on config in datapreprocessing.
### 3D Env
To train model, run:
```
python train/train_room.py --exp_name test_3d --data_dir datasets/test_3d --dof 3 --robot_name robot --rotate_axis z --loss_ratio 1.2 --h_size 256 --no_enc_env --sampler batch_weighted --enc_d 7 --dec_d 7 --start_epoch_loss_val 1.5 --use_no_coll_split --use_path_loss --path_loss_start_epoch 100 --beta 0 --beta1 0 --beta2 0.08 --beta3 0.08 --beta5 0 --beta6 0 --offset 0.0025 --epoch 600 
```
Because of the fixing dim, the dof is set to 3.

### Arm
To train model, run:
```
python train/train_room.py --exp_name test_arm --data_dir datasets/test_arm/4DOF --dof 4 --robot_name arm --sim --loss_ratio 1.2 --h_size 256 --no_enc_env --sampler batch_weighted --enc_d 7 --dec_d 7 --start_epoch_loss_val 2.0 --use_no_coll_split --use_path_loss --path_loss_start_epoch 100 --beta 0 --beta1 0 --beta2 0.08 --beta3 0.08 --beta5 0 --beta6 0 --offset 0.0033 --epoch 600 
```
## Evaluation
### 2D Env
To plan a single path, run:
```
python test/room_plan.py --exp_name test_2d --data_dir datasets/test_2d --SE2 --dof 3 --enc_d 7 --dec_d 7 --no_enc_env --dist_tol 0.06 --step 0.03 --offset 0.01 --epoch 600 [--adaptive_planning] [--vis_track]
```
To plan serval paths to evaluate, run:
```
python test/room_succ.py --exp_name test_2d --data_dir datasets/test_2d --SE2 --dof 3 --enc_d 7 --dec_d 7 --no_enc_env --offset 0.01 --epoch 600 --test_points_n 100 [--adaptive_planning] [--vis_succ] [--vis_failure] [--vis_adaptive]
```
You need to set the config [--offset] based on config in datapreprocessing.
### 3D Env
To plan a single path, run:
```
python test/room_plan.py --exp_name test_3d --data_dir datasets/test_3d --dof 3 --robot_name robot --rotate_axis z --coll_robot_dir datasets/test_3d --enc_d 7 --dec_d 7 --no_enc_env --dist_tol 0.06 --step 0.03 --offset 0.0025 --epoch 600 [--adaptive_planning] [--vis_track]
```
To plan serval paths to evaluate, run:
```
python test/room_succ.py --exp_name test_3d --data_dir datasets/test_3d --dof 3 --robot_name robot --rotate_axis z --coll_robot_dir datasets/test_3d --ref_data_dir datasets/test_3d/ref --enc_d 7 --dec_d 7 --no_enc_env --offset 0.0025 --epoch 600 --test_points_n 10000 [--adaptive_planning] [--vis_succ] [--vis_failure] [--vis_adaptive]
```
### Arm
To plan a single path, run:
```
python test/room_plan.py --exp_name test_arm --data_dir datasets/test_arm/4DOF --dof 4 --robot_name arm --sim  --enc_d 7 --dec_d 7 --no_enc_env --dist_tol 0.06 --step 0.03 --offset 0.0033 --epoch 600 [--adaptive_planning]
```
To plan serval paths to evaluate, run:
```
python test/room_succ.py --exp_name test_arm --data_dir datasets/test_arm/4DOF --dof 4 --robot_name arm --sim --enc_d 7 --dec_d 7 --no_enc_env --offset 0.0033 --epoch 600 --test_points_n 10000 [--adaptive_planning] [--vis_succ] [--vis_failure] [--vis_adaptive]
```

## BibTeX
```
@inproceedings{shen2024pc,
  title={Pc-planner: Physics-constrained self-supervised learning for robust neural motion planning with shape-aware distance function},
  author={Shen, Xujie and Peng, Haocheng and Yang, Zesong and Xu, Juzhan and Bao, Hujun and Hu, Ruizhen and Cui, Zhaopeng},
  booktitle={SIGGRAPH Asia 2024 Conference Papers},
  pages={1--11},
  year={2024}
}
```
## License
PC-Planner is released under the MIT License. See the LICENSE file for more details.
