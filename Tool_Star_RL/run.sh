#!/bin/bash
source ~/.bashrc
source ~/miniconda3/bin/activate
for j in {1..12000}
do
echo $j
sleep 1
echo "sleep for next run"

done

for i in {1..6}  
do
cd ~
python kill.py
cd /share/home/gtang/sxjiang-gt/Tool-Star-OCT
bash ./retriever_launch_hit.sh &
sleep 20
cd Tool_Star_RL
conda activate toolstar
bash ./run_tool_star_hit.sh

done
