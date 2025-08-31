#!/bin/bash
source ~/.bashrc
source ~/anaconda3/bin/activate
conda activate toolstar
export LD_LIBRARY_PATH=$HOME/anaconda3/envs/toolstar/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

python model_merger.py --local_dir "/share/home/jfliang/Project/sxjiang/Tool-Star-OCT/Tool_Star_RL/verl_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-origin-bz_128-clip_ratio_0.2_one_epoch_no_warm_up_no_progressive_seq_mean/global_step_78/actor/"