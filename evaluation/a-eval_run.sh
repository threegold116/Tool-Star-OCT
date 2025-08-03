#!/bin/bash
source ~/.bashrc
source ~/miniconda3/bin/activate
cd ~
python kill.py
cd /home/sxjiang/myproject/agent/Tool-Star-OCT
bash ./retriever_launch.sh &
sleep 20
cd evaluation
conda activate toolstar
bash ./a-eval.sh