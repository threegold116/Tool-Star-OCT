#!/bin/bash

source ~/.bashrc
source ~/miniconda3/bin/activate
conda activate retriever
cd evaluation/search
python host_wiki.py \
    --config serving_config.yaml \
    --num_retriever 1 \
    --port 8000