#!/bin/bash
source ~/.bashrc
source ~/miniconda3/bin/activate
source ~/miniconda3/bin/activate
conda activate toolstar
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/toolstar/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/toolstar/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

python model_merger.py --local_dir "/share/home/zxwang/sxjiang-zx/Tool-Star-OCT/Tool_Star_RL/verl_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_two_epoch_no_warm_up_down_progressive_seq_mean_void_turns_mask-data_filiter_1104-advantage_shaping/global_step_100/actor/"
