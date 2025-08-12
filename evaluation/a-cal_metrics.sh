#!/bin/bash
export ALIYUN_API_KEY=sk-2030e8bbeb6b4287bc929cdc24887d7b
run_mertic(){
    local OUTPUT_PATH=$1
    local TASK=$2
    local DATASET_NAME=$3

    python /home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/evaluate/scripts/evaluate.py\
        --output_path $OUTPUT_PATH\
        --task $TASK\
        --dataset_name $DATASET_NAME\
        --use_llm \
        --extract_answer \

    echo "Metrics calculated for $DATASET_NAME"
}
run_dataset(){
    local TASK=$1
    local DATASET_NAME=$2
    local INPUT_DIR="/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/result/$DATASET_NAME/"
    # 遍历所有 JSON 文件
    find "$INPUT_DIR" -type f -name "result.json" | while read -r input_file; do
        echo "Running evaluation on $input_file"
        run_mertic $input_file $TASK $DATASET_NAME
    done
}

TASK="qa"
DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki hle gaia webwalker)
for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
    run_dataset $TASK $DATASET_NAME

done

TASK="math"
DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
    run_dataset $TASK $DATASET_NAME
    
done