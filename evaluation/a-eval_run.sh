#!/bin/bash
source ~/.bashrc
source ~/miniconda3/bin/activate
cd ~
python kill.py
cd /home/sxjiang/myproject/agent/Tool-Star-OCT
bash ./retriever_launch.sh &
for i in {1..60};
do
    echo $i
    sleep 1
    echo "retriever_launch.sh is loading"
done
cd evaluation
conda activate toolstar
bash ./a-eval.sh