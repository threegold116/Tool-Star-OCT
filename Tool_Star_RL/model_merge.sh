#!/bin/bash
source ~/.bashrc
source ~/miniconda3/bin/activate
conda activate toolstar
cd /share/home/sxjiang/myproject/Tool-Star-OCT/Tool_Star_RL
python model_merger.py --local_dir "/share/home/sxjiang/myproject/Tool-Star-OCT/Tool_Star_RL/verl_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.28/global_step_80/actor"