#!/bin/bash
source ~/.bashrc
source ~/miniconda3/bin/activate

for i in {1..10}  
do
python kill.py
cd /share/home/gtang/sxjiang-gt/Tool-Star-OCT
bash ./retriever_launch_hit.sh &
sleep 20
cd Tool_Star_RL
conda activate toolstar
bash ./run_tool_star_hit.sh

done
