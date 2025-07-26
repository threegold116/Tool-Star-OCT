#!/bin/bash
sleep 1800
source ~/.bashrc

source ~/miniconda3/bin/activate
python /share/home/jfliang/Project/sxjiang/Tool-Star-OCT/Tool_Star_RL/kill.py
conda activate retriever
cd /share/home/jfliang/Project/sxjiang/Tool-Star-OCT/
bash ./retriever_launch_hit.sh &
sleep 20
cd /share/home/jfliang/Project/sxjiang/Tool-Star-OCT/Tool_Star_RL
conda activate toolstar
bash ./run_tool_star_hit.sh


