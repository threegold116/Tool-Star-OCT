#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS

export BING_ENDPOINT=https://api.langsearch.com/v1/web-search
export PYTHONPATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
run_evaluation() {
    local MODEL_PATH=$1
    local MODEL=$2
    local TASK=$3
    local MODE=$4
    local MAX_RESPONSE_LENGTH=8192
    local MAX_INPUT_LEN=8192
    local DATASET_NAMEs=("${@:5}")
    for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
        echo "Evaluating $DATASET_NAME"
        echo "MODEL_PATH: $MODEL_PATH"
        echo "MODEL: $MODEL"
        echo "TASK: $TASK"
        echo "MODE: $MODE"
        echo "DATASET_NAME: $DATASET_NAME"
        echo "DATASET_NAMEs: ${DATASET_NAMEs[@]}"
        echo "--------------------------------"
    done
    for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
        python run_budget.py \
            --model_path "$MODEL_PATH" \
            --dataset_name "$DATASET_NAME" \
            --task "$TASK" \
            --gpu_use 0.8 \
            --max_input_len $MAX_INPUT_LEN \
            --max_response_length $MAX_RESPONSE_LENGTH \
            --prompt_type code_search \
            --output_path "/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/result/$DATASET_NAME/$MODE/$MODEL/result.json" \
            --counts 500 \
            --batch_size 100 \
            --max_calling_times 10 \
            --max_tool_budget 3000 \
            --python_budget 5 \
            --search_budget 2 \
            --resume_evaluate\
            --all_wiki 0
    done
}
budget_limit_run_evaluation() {
    local MODEL_PATH=$1
    local MODEL=$2
    local TASK=$3
    local MODE=$4
    local BUDGET=$5
    local MAX_RESPONSE_LENGTH=8192
    local MAX_INPUT_LEN=8192
    local DATASET_NAMEs=("${@:6}")
    # local BUDGET_IDXS=(0 1 2 3 4 5 6 7 8 9)
    local BUDGET_IDXS=(0 1 2 3 4)
    echo $MODE
    # for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
    #     echo "Evaluating $DATASET_NAME"
    #     echo "MODEL_PATH: $MODEL_PATH"
    #     echo "MODEL: $MODEL"
    #     echo "TASK: $TASK"
    #     echo "MODE: $MODE"
    #     echo "DATASET_NAME: $DATASET_NAME"
    #     echo "DATASET_NAMEs: ${DATASET_NAMEs[@]}"
    #     echo "--------------------------------"
    # done
    for BUDGET_IDX in "${BUDGET_IDXS[@]}"; do
        for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
            python run_budget.py \
                --model_path "$MODEL_PATH" \
                --dataset_name "$DATASET_NAME" \
                --task "$TASK" \
                --gpu_use 0.8 \
                --max_input_len $MAX_INPUT_LEN \
                --max_response_length $MAX_RESPONSE_LENGTH \
                --prompt_type code_search_with_budget \
                --output_path "/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/result/$DATASET_NAME/$MODE"_"$BUDGET/budget_idx_$BUDGET_IDX/$MODEL/result.json" \
                --counts 500 \
                --batch_size 100 \
                --max_calling_times 10 \
                --max_tool_budget "$BUDGET" \
                --python_budget 5 \
                --search_budget 2 \
                --all_wiki 0 \
                --budget_idx "$BUDGET_IDX"\
                # --resume_evaluate
            done
    done
}
# Tool-Star-Qwen-3b-Origin
# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_origin
# MODEL_PATH=/home/sxjiang/model/Tool-Star-Qwen-3B

# DATASET_NAMEs=(webwalker musique bamboogle hotpotqa 2wiki hle gaia)
# TASK=qa
# ##调用函数
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"


# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# TASK=math
# ##调用函数
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"



# #Tool-Star-Qwen-3b-SFT
# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_sft
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki hle gaia webwalker)
# TASK=qa
# ##调用函数
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"


# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# TASK=math
# ##调用函数
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# #Tool-Star-Qwen-3b-OCT-Clip-Radio-0.28-Global-Step-80
# MODEL=tool_star_qwen_3b_oct_clip_radio_028_global_step_80
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.28_global_step_80

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki hle gaia webwalker)
# TASK=qa
# ##调用函数
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# TASK=math
# ##调用函数
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"



# #Tool-Star-Qwen-3b-OCT-Clip-Radio-GradClip-0.28-Global-Step-70
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_028_global_step_70
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.28_grad_clip_no_masked_soft-global_step_70

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki hle gaia webwalker)
# TASK=qa
# ##调用函数
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"


# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# TASK=math
# ##调用函数
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# #Tool-Star-Qwen-3b-OCT-Clip-Radio-GradClip-0.28-One-Epoch-Global-Step-40
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_028_one_epoch_global_step_40
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.28_one_epoch-global_step_40

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki hle gaia webwalker)
# TASK=qa
# ##调用函数
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# TASK=math
# ##调用函数
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# #Tool-Star-Qwen-3b-OCT-Clip-Radio-GradClip-0.28-One-Epoch-WarmUp-Global-Step-96
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_028_one_epoch_warmup_global_step_96
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.28_one_epoch_warm_up-global_step_96

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki hle gaia webwalker)
# TASK=qa
# ##调用函数
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# TASK=math
# ##调用函数
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# #Tool-Star-Qwen-3b-OCT-Clip-Radio-GradClip-0.28-One-Epoch-WarmUp-Global-Step-78
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_028_one_epoch_warmup_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.28_one_epoch_warm_up-global_step_78

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki hle gaia webwalker)
# TASK=qa
# ##调用函数
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# TASK=math
# ##调用函数
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# #Tool-Star-Qwen-3b-OCT-Clip-Radio-GradClip-0.28-One-Epoch-WarmUp-0.95-Global-Step-78
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_028_one_epoch_warmup_095_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.28_one_epoch_warm_up_0.95-global_step_78

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki hle gaia webwalker)
# TASK=qa
# ##调用函数
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# TASK=math
# ##调用函数
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# #Tool-Star-Qwen-3b-OCT-Clip-Radio-GradClip-0.28-One-Epoch-Mini-Batch-32-Global-Step-78
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_028_one_epoch_mini_batch_32_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.28_one_epoch_mini_batch_32-global_step_78

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki hle gaia webwalker)
# TASK=qa
# ##调用函数
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# TASK=math
# ##调用函数
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"


BUDGET=10
MODE=budget_limit_run
MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_028_one_epoch_warmup_095_global_step_78
MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.28_one_epoch_warm_up_0.95-global_step_78

DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki hle gaia webwalker)
TASK=qa
##调用函数
budget_limit_run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "$BUDGET" "${DATASET_NAMEs[@]}"

DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
TASK=math
##调用函数
budget_limit_run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "$BUDGET" "${DATASET_NAMEs[@]}"


# Tool-Star-Qwen-3b-Origin
MODE=budget_limit_run
MODEL=tool_star_qwen_3b_origin
MODEL_PATH=/home/sxjiang/model/Tool-Star-Qwen-3B

DATASET_NAMEs=(webwalker musique bamboogle hotpotqa 2wiki hle gaia)
TASK=qa
##调用函数
budget_limit_run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "$BUDGET" "${DATASET_NAMEs[@]}"


DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
TASK=math
##调用函数
budget_limit_run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "$BUDGET" "${DATASET_NAMEs[@]}"



#Tool-Star-Qwen-3b-SFT
MODE=budget_limit_run
MODEL=tool_star_qwen_3b_sft
MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52

DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki hle gaia webwalker)
TASK=qa
##调用函数
budget_limit_run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "$BUDGET" "${DATASET_NAMEs[@]}"

DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
TASK=math
##调用函数
budget_limit_run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "$BUDGET" "${DATASET_NAMEs[@]}"


