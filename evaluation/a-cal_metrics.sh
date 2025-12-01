#!/bin/bash
source ~/.bashrc
source ~/miniconda3/bin/activate
conda activate toolstar
export ALIYUN_MODEL_NAME=Qwen2.5-72B-Instruct-GPTQ-Int4
run_mertic(){
    local OUTPUT_PATH=$1
    local TASK=$2
    local DATASET_NAME=$3
    python /home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/evaluate/scripts/evaluate.py\
        --output_path $OUTPUT_PATH\
        --task $TASK\
        --dataset_name $DATASET_NAME\
        --use_llm \
        --extract_answer 
    # else
    #     python /home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/evaluate/scripts/evaluate.py\
    #         --output_path $OUTPUT_PATH\
    #         --task $TASK\
    #         --dataset_name $DATASET_NAME\
    #         --use_llm \
    #         --extract_answer 
    # fi

    echo "Metrics calculated for $DATASET_NAME"
}

run_dataset_model_mode(){
    local TASK=$1
    local DATASET_NAME=$2
    local MODEL=$3
    local MODE=$4
    local INPUT_DIR="/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/result/$DATASET_NAME/"
    echo "Processing dataset: $DATASET_NAME with model: $MODEL in mode: $MODE"
    # 遍历所有 JSON 文件
    find "$INPUT_DIR" -type f -name "result.json" | while read -r input_file; do\
        if [[ "$input_file" == *"$MODEL"* ]]; then
            if [[ "$input_file" == *"$MODE"* ]]; then
                echo "$input_file"
                run_mertic $input_file $TASK $DATASET_NAME
            fi
            # run_mertic $input_file $TASK $DATASET_NAME
        fi
    done
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

# TASK="qa"


# no_budget_limit_run
MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_two_epoch_no_warm_up_down_progressive_seq_mean-data_filiter_1104-batch_normlization-global_step_100
MODE="budget_no_limit_run"
TASK="qa"
DATASET_NAMEs=(musique bamboogle 2wiki beerqa)
for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
    run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
done

MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_two_epoch_no_warm_up_down_progressive_seq_mean-data_filiter_1104-batch_normlization-global_step_100
MODE="budget_no_limit_run"
TASK="math"
DATASET_NAMEs=(gsm8k aime25 math OlymBench-math)
for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
    run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
done