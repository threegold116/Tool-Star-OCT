#!/bin/bash
source ~/.bashrc
source ~/miniconda3/bin/activate
conda activate toolstar
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/toolstar/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
cd /share/home/sxjiang/myproject/Tool-Star-OCT/Tool_Star_RL
python model_merger.py --local_dir "/share/home/sxjiang/myproject/Tool-Star-OCT/Tool_Star_RL/verl_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.28_one_epoch_tool_star_no_oct/global_step_78/actor/"