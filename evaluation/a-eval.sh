#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS
# export ALIYUN_API_KEY=sk-2030e8bbeb6b4287bc929cdc24887d7b
export BING_API_KEY_1=sk-a8f1b7c96a684889bedf6f35e3c008a3
export BING_API_KEY_2=sk-06aee9a048ef4aaea1cf1b84df1fa857
export BING_API_KEY_3=sk-ee3c6636fd7f40c98e7cb1f9d755aab3
export BING_API_KEY=sk-3393021642df400d92fe955b63ea33cd
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
        python run_budget_CIR.py \
            --model_path "$MODEL_PATH" \
            --dataset_name "$DATASET_NAME" \
            --task "$TASK" \
            --gpu_use 0.85 \
            --max_input_len $MAX_INPUT_LEN \
            --max_response_length $MAX_RESPONSE_LENGTH \
            --prompt_type CIR \
            --output_path "/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/result/$DATASET_NAME/$MODE/$MODEL/result.json" \
            --counts 500 \
            --batch_size 100 \
            --max_calling_times 10 \
            --max_tool_budget 3000 \
            --python_budget 5 \
            --search_budget 2 \
            --all_wiki 0 \
            # --resume_evaluate

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
    local BUDGET_IDXS=(0)
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
            for i in {1..1}; do
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
                    --budget_idx "$BUDGET_IDX"\         # 注释这个，然后自定义Budget
                    # --resume_evaluate
            done
        done
    done
}


BUDGET=10
# # # Tool-Star-Qwen-3b-Origin
# echo $BUDGET
# MODE=budget_limit_run

# MODEL=tool_star_qwen_3b_origin
# MODEL_PATH=/home/sxjiang/model/Tool-Star-Qwen-3B

# DATASET_NAMEs=(webwalker musique bamboogle hotpotqa 2wiki)
# TASK=qa
# ##调用函数
# budget_limit_run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "$BUDGET" "${DATASET_NAMEs[@]}"

# TASK=math
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# budget_limit_run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "$BUDGET" "${DATASET_NAMEs[@]}"


# # # Tool-Star-Qwen-3b-Origin
# MODE=budget_limit_run
# MODEL=tool_star_qwen_3b_sft
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52

# DATASET_NAMEs=(webwalker musique bamboogle hotpotqa 2wiki)
# TASK=qa
# ##调用函数
# budget_limit_run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "$BUDGET" "${DATASET_NAMEs[@]}"
# TASK=math
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# budget_limit_run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "$BUDGET" "${DATASET_NAMEs[@]}"


# # # Tool-Star-Qwen-3b-Origin
# MODE=budget_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_028_one_epoch_new_no_warm_up_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.28_one_epoch_tool_star_no_warm_up_new-global_step_78

# DATASET_NAMEs=(webwalker musique bamboogle hotpotqa 2wiki)
# TASK=qa
# ##调用函数
# budget_limit_run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "$BUDGET" "${DATASET_NAMEs[@]}"
# TASK=math
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# budget_limit_run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "$BUDGET" "${DATASET_NAMEs[@]}"


# # # Tool-Star-Qwen-3b-Origin
# MODE=budget_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_028_one_epoch_new_origin_no_warm_up_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.28_one_epoch_tool_star_new_origin_no_warm_up-global_step_78

# DATASET_NAMEs=(webwalker musique bamboogle hotpotqa 2wiki)
# TASK=qa
# ##调用函数
# budget_limit_run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "$BUDGET" "${DATASET_NAMEs[@]}"
# TASK=math
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# budget_limit_run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "$BUDGET" "${DATASET_NAMEs[@]}"


## Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.28_one_epoch_tool_star_new_no_warm_up_no_oct-global_step_78
## 测试不带oct的情况
# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_028_one_epoch_new_no_warm_up_no_oct_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.28_one_epoch_tool_star_new_no_warm_up_no_oct-global_step_78
# DATASET_NAMEs=(webwalker musique bamboogle hotpotqa 2wiki)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# TASK=math
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "$BUDGET" "${DATASET_NAMEs[@]}"

## Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.28_one_epoch_tool_star_new_no_warm_up_no_oct-global_step_78
## 测试不带oct的情况,并换为f1_score
# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_028_one_epoch_new_no_warm_up_f1_score_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.28_one_epoch_tool_star_new_f1_score_no_warm_up-global_step_78
# DATASET_NAMEs=(webwalker musique bamboogle hotpotqa 2wiki)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# TASK=math
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "$BUDGET" "${DATASET_NAMEs[@]}"


## Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.28_grad_clip_epoch1_warm_up_new-global_step_78
## 对比测试budget作为惩罚和times作为惩罚的效果，先测budget
# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_028_one_epoch_new_budget_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.28_grad_clip_epoch1_warm_up_new-global_step_78
# DATASET_NAMEs=(webwalker musique bamboogle hotpotqa 2wiki)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# TASK=math
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "$BUDGET" "${DATASET_NAMEs[@]}"

# ## 再测试times的
# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_028_one_epoch_new_times_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.28_grad_clip_epoch1_warm_up_new_times-global_step_78
# DATASET_NAMEs=(webwalker musique bamboogle hotpotqa 2wiki)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# TASK=math
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "$BUDGET" "${DATASET_NAMEs[@]}"

## 测试origin和oct在gpqa上的表现（这个gpqa的answer又有文字类型又有math类型，感觉不好评） 用的是web_search
## 先测origin的（作者的origin）
# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_origin
# MODEL_PATH=/home/sxjiang/model/Tool-Star-Qwen-3B
# DATASET_NAMEs=(gpqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# ## 再测我们的oct的
# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_028_one_epoch_no_warmup_new_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.28_one_epoch_tool_star_no_warm_up_new-global_step_78
# DATASET_NAMEs=(gpqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# ## 还是测试origin和oct在gpqa上的表现，用wiki_search
# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_origin_wiki
# MODEL_PATH=/home/sxjiang/model/Tool-Star-Qwen-3B
# DATASET_NAMEs=(gpqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# ## 再测我们的oct的
# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_028_one_epoch_no_warmup_new_wiki_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.28_one_epoch_tool_star_no_warm_up_new-global_step_78
# DATASET_NAMEs=(gpqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

## 还是测试origin和oct在gpqa上的表现，用wiki_search, prompt里鼓励一下multi-tool
# MODE=budget_no_limit_multi_tool_hint_prompt_all_wiki
# MODEL=tool_star_qwen_3b_origin
# MODEL_PATH=/home/sxjiang/model/Tool-Star-Qwen-3B
# DATASET_NAMEs=(gpqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# ## 再测我们的oct的
# MODE=budget_no_limit_multi_tool_hint_prompt_all_wiki
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_028_one_epoch_no_warmup_new_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.28_one_epoch_tool_star_no_warm_up_new-global_step_78
# DATASET_NAMEs=(gpqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

####################################################################################################################
# 测Search-r1,先测3B
# MODE=budget_no_limit_run
# MODEL=Search-R1_3b
# MODEL_PATH=/home/sxjiang/model/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki gpqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# TASK=math
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "$BUDGET" "${DATASET_NAMEs[@]}"

# # 测search-r1,测7B
# MODE=budget_no_limit_run
# MODEL=Search-R1_7b
# MODEL_PATH=/home/sxjiang/model/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-grpo
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki gpqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# TASK=math
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "$BUDGET" "${DATASET_NAMEs[@]}"

####################################################################################################################
# MODE=budget_no_limit_run
# MODEL=ARPO_3b
# MODEL_PATH=/home/sxjiang/model/Qwen2.5-3B-ARPO
# DATASET_NAMEs=(2wiki)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=auto-tir_7b
# MODEL_PATH=/home/sxjiang/model/AutoTIR-Qwen2.5-7B-Instruct
# DATASET_NAMEs=(2wiki)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# ####################################################################################################################
# # 测Search-r1,先测3B
# MODE=budget_no_limit_run
# MODEL=Search-R1_3b
# MODEL_PATH=/home/sxjiang/model/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki gpqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# TASK=math
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "$BUDGET" "${DATASET_NAMEs[@]}"

# ####################################################################################################################
# 2025-08-20
# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_032_one_epoch_no_progressive_seq_mean_new_times_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.32_one_epoch_no_warm_up_no_progressive_seq_mean_times-global_step_78
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_032_one_epoch_no_progressive_seq_mean_new_times_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.32_one_epoch_no_warm_up_no_progressive_seq_mean_times-global_step_78
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# ####################################################################################################################
# 2025-08-21
# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_032_one_epoch_no_progressive_seq_mean_new_negative_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.32_one_epoch_no_warm_up_no_progressive_seq_mean_negative_smooth-global_step_78
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_032_one_epoch_no_progressive_seq_mean_new_negative_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.32_one_epoch_no_warm_up_no_progressive_seq_mean_negative_smooth-global_step_78
# DATASET_NAMEs=(aime24)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=auto-tir_7b_prompt
# MODEL_PATH=/home/sxjiang/model/AutoTIR-Qwen2.5-7B-Instruct
# DATASET_NAMEs=(musique bamboogle hotpotqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=auto-tir_7b_prompt
# MODEL_PATH=/home/sxjiang/model/AutoTIR-Qwen2.5-7B-Instruct
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=ARPO_7b
# MODEL_PATH=/home/sxjiang/model/Qwen2.5-7B-ARPO
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=ARPO_7b
# MODEL_PATH=/home/sxjiang/model/Qwen2.5-7B-ARPO
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# ####################################################################################################################
# 2025-08-22

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_one_epoch_no_progressive_seq_mean_new_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_one_epoch_no_warm_up_no_progressive_seq_mean_smooth_specific-global_step_78
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_one_epoch_no_progressive_seq_mean_new_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_one_epoch_no_warm_up_no_progressive_seq_mean_smooth_specific-global_step_78
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# no-tool-inference

# MODE=no-tool-inference
# MODEL=tool_star_qwen_3b_origin
# MODEL_PATH=/home/sxjiang/model/Tool-Star-Qwen-3B
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=no-tool-inference
# MODEL=tool_star_qwen_3b_origin
# MODEL_PATH=/home/sxjiang/model/Tool-Star-Qwen-3B
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=no-tool-inference
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_032_one_epoch_no_progressive_seq_mean_new_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.32_one_epoch_no_warm_up_no_progressive_seq_mean_specific_smooth
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=no-tool-inference
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_032_one_epoch_no_progressive_seq_mean_new_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.32_one_epoch_no_warm_up_no_progressive_seq_mean_specific_smooth
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=no-tool-inference
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_one_epoch_no_progressive_seq_mean_new_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_one_epoch_no_warm_up_no_progressive_seq_mean_smooth_specific-global_step_78
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=no-tool-inference
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_one_epoch_no_progressive_seq_mean_new_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_one_epoch_no_warm_up_no_progressive_seq_mean_smooth_specific-global_step_78
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# no-tool-inference-2

# MODE=no-tool-inference-test
# MODEL=tool_star_qwen_3b_origin
# MODEL_PATH=/home/sxjiang/model/Tool-Star-Qwen-3B
# DATASET_NAMEs=(2wiki)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# ####################################################################################################################
# 2025-08-23

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_032_one_epoch_no_progressive_seq_mean_new_add-oct_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.32_one_epoch_no_warm_up_no_progressive_seq_mean_smooth_specific-add_oct-global_step_78
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_032_one_epoch_no_progressive_seq_mean_new_add-oct_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.32_one_epoch_no_warm_up_no_progressive_seq_mean_smooth_specific-add_oct-global_step_78
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=qwen2p5_instruct_3b
# MODEL_PATH=/home/sxjiang/model/Qwen2.5-3B-Instruct
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# TASK=math
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "$BUDGET" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=qwen2p5_instruct_7b
# MODEL_PATH=/home/sxjiang/model/Qwen2.5-7B-Instruct
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# TASK=math
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "$BUDGET" "${DATASET_NAMEs[@]}"

# ####################################################################################################################
# 2025-08-24

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_032_one_epoch_down_progressive_seq_mean_new_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.32_one_epoch_no_warm_up_down_progressive_seq_mean_smooth_specific-global_step_78
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_032_one_epoch_down_progressive_seq_mean_new_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.32_one_epoch_no_warm_up_down_progressive_seq_mean_smooth_specific-global_step_78
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=no-tool-inference2
# MODEL=tool_star_qwen_3b_origin
# MODEL_PATH=/home/sxjiang/model/Tool-Star-Qwen-3B
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=no-tool-inference2
# MODEL=tool_star_qwen_3b_origin
# MODEL_PATH=/home/sxjiang/model/Tool-Star-Qwen-3B
# DATASET_NAMEs=(aime24)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run_test_2
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_032_one_epoch_no_progressive_seq_mean_new_group_smooth_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.32_one_epoch_no_warm_up_no_progressive_seq_mean_smooth_specific_group_smooth-global_step_78

# DATASET_NAMEs=(aime24)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run_test_5
# MODEL=tool_star_qwen_3b_origin
# MODEL_PATH=/home/sxjiang/model/Tool-Star-Qwen-3B

# DATASET_NAMEs=(aime24)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run_test_gpu18
# MODEL=tool_star_qwen_3b_origin
# MODEL_PATH=/home/sxjiang/model/Tool-Star-Qwen-3B

# DATASET_NAMEs=(math)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"


# MODE=budget_no_limit_run_test_gpu15
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_028_one_epoch_new_origin_no_warm_up_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.28_one_epoch_tool_star_no_warm_up_new-global_step_78

# DATASET_NAMEs=(math)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# ####################################################################################################################
# 2025-08-25

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_origin
# MODEL_PATH=/home/sxjiang/model/Tool-Star-Qwen-3B

# DATASET_NAMEs=(amc23)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_032_one_epoch_no_progressive_seq_mean_new_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.32_one_epoch_no_warm_up_no_progressive_seq_mean_specific_smooth

# DATASET_NAMEs=(amc23)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_one_epoch_no_progressive_seq_mean_new_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_one_epoch_no_warm_up_no_progressive_seq_mean_smooth_specific-global_step_78

# DATASET_NAMEs=(amc23)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_032_one_epoch_no_progressive_seq_mean_new_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.32_one_epoch_no_warm_up_no_progressive_seq_mean_specific_smooth

# DATASET_NAMEs=(aime24)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_one_epoch_no_progressive_seq_mean_new_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_one_epoch_no_warm_up_no_progressive_seq_mean_smooth_specific-global_step_78

# DATASET_NAMEs=(aime24)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# ####################################################################################################################
# 2025-08-26

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_origin_gpu18
# MODEL_PATH=/home/sxjiang/model/Tool-Star-Qwen-3B
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_032_one_epoch_no_progressive_seq_mean_new_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.32_one_epoch_no_warm_up_no_progressive_seq_mean_specific_smooth

# DATASET_NAMEs=(nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(amc23 OlymMATH-EASY OlymMATH-HARD)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_one_epoch_no_progressive_seq_mean_new_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_one_epoch_no_warm_up_no_progressive_seq_mean_smooth_specific-global_step_78

# DATASET_NAMEs=(nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(amc23 OlymMATH-EASY OlymMATH-HARD)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_032_one_epoch_down_progressive_seq_mean_new2_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.32_one_epoch_no_warm_up_down_progressive_seq_mean_smooth_specific_new-global_step_78

# DATASET_NAMEs=(nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(amc23 OlymMATH-EASY OlymMATH-HARD)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_032_one_epoch_no_progressive_seq_mean_new_group_smooth_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.32_one_epoch_no_warm_up_no_progressive_seq_mean_specific_group_smooth_new-global_step_78
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# ####################################################################################################################
# 2025-08-27

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_one_epoch_no_progressive_seq_mean_new_add_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_one_epoch_no_warm_up_no_progressive_seq_mean_smooth_specific_add-global_step_78

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# ####################################################################################################################
# 2025-08-28

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_origin
# MODEL_PATH=/home/sxjiang/model/Tool-Star-Qwen-3B

# DATASET_NAMEs=(fanout)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(OlymBench-math OlymBench-physics)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_one_epoch_no_progressive_seq_mean_new_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_one_epoch_no_warm_up_no_progressive_seq_mean_smooth_specific-global_step_78

# DATASET_NAMEs=(OlymBench-math)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_two_epoch_down_progressive_seq_mean_new_add_global_step_110
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_two_epoch_no_warm_up_down_progressive_seq_mean_smooth_add-global_step_110

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# ####################################################################################################################
# 2025-08-29

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_new_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_one_epoch_no_warm_up_down_progressive_seq_mean_smooth-global_step_78
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_7b_origin
# MODEL_PATH=/home/sxjiang/model/Tool-Star-Qwen-7B
# DATASET_NAMEs=(nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(amc23 OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# ####################################################################################################################
# 2025-08-30

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_smooth_binary_f1_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_one_epoch_no_warm_up_down_progressive_seq_mean_smooth_multiply_binary_f1-global_step_78

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"


# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_6_seq_mean_smooth_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_one_epoch_no_warm_up_down_progressive_6_seq_mean_smooth-global_step_78

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"


# MODE=budget_no_limit_run_gpu18_test
# MODEL=tool_star_qwen_3b_origin
# MODEL_PATH=/home/sxjiang/model/Tool-Star-Qwen-3B

# DATASET_NAMEs=(math500)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# ####################################################################################################################
# 2025-08-31

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_smooth_binary_f1_math_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_one_epoch_no_warm_up_down_progressive_seq_mean_smooth_multiply_binary_f1_math-global_step_78

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_one_epoch_no_progressive_seq_mean_origin_no_multi_reward_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_one_epoch_no_warm_up_no_progressive_seq_mean-origin_no_multi_reward-global_step_78

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# ####################################################################################################################
# 2025-09-01

# MODE=budget_no_limit_run
# MODEL=ARPO_3b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_smooth_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/ARPO-OCT/transfer_checkpoints/ARPO_global_16_init_8_beam_2_random_0_arpo_0.2_entropy_oct_downprogressive_em_score_seq_mean_specific_smooth-origin-global_step_78

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_7b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_new_global_step_78_gpu09
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-7B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_one_epoch_no_warm_up_down_progressive_seq_mean_smooth_multiply-global_step_78

# DATASET_NAMEs=(nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"


# ####################################################################################################################
# 2025-09-02

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_two_epoch_down_progressive_seq_mean_new_global_step_110
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_two_epoch_no_warm_up_down_progressive_seq_mean-global_step_110

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_7b_oct_clip_radio_gradclip_02_two_epoch_down_progressive_seq_mean_new_global_step_80
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-7B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_two_epoch_no_warm_up_down_progressive_seq_mean_smooth_multiply-global_step_80

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# ####################################################################################################################
# 2025-09-03

# MODE=budget_no_limit_run
# MODEL=ARPO_3b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_em_score_seq_mean_smooth_origin_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/ARPO-OCT/transfer_checkpoints/ARPO_3B_global_16_init_8_beam_2_random_0_arpo_0.2_entropy_oct_downprogressive_em_score_seq_mean_specific_smooth-origin-global_step_78

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# ####################################################################################################################
# 2025-09-05

# MODE=budget_no_limit_run
# MODEL=ARPO_7b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_em_score_seq_mean_smooth_origin_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/ARPO-OCT/transfer_checkpoints/ARPO_7B_global_16_init_8_beam_2_random_0_arpo_0.2_one_epoch_entropy_oct_downprogressive_em_score_seq_mean_specific_smooth-origin-global_step_78

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=ARPO_3b
# MODEL_PATH=/home/sxjiang/model/Qwen2.5-3B-ARPO

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=ARPO_7b
# MODEL_PATH=/home/sxjiang/model/Qwen2.5-7B-ARPO

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# ####################################################################################################################
# 2025-09-06

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_7b_oct_clip_radio_gradclip_02_one_epoch_no_progressive_seq_mean_smooth_multiply_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-7B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_one_epoch_no_warm_up_no_progressive_seq_mean_smooth_multiply-global_step_78

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# ####################################################################################################################
# 2025-09-07

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_smooth_multiply__no_optim_cost_estimate_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_one_epoch_no_warm_up_down_progressive_seq_mean_smooth_multiply_no_optim_cost_estimate-global_step_78

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_7b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_smooth_multiply__no_optim_cost_estimate_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-7B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_one_epoch_no_warm_up_down_progressive_seq_mean_no_optim_cost_estimate-global_step_78

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# ####################################################################################################################
# 2025-09-08

# MODE=budget_no_limit_run
# MODEL=qwen2p5_instruct_3b
# MODEL_PATH=/home/sxjiang/model/Qwen2.5-3B-Instruct

# DATASET_NAMEs=( nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(amc23 OlymBench-math)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"


# MODE=budget_no_limit_run
# MODEL=qwen2p5_instruct_7b
# MODEL_PATH=/home/sxjiang/model/Qwen2.5-7B-Instruct

# DATASET_NAMEs=( nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(amc23 OlymBench-math)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# ####################################################################################################################
# 2025-09-09

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_7b_clip_radio_gradclip_02_one_epoch_down_progressive_2_seq_mean_global_step_70
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-7B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_one_epoch_no_warm_up_down_progressive_2_seq_mean-global_step_70

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=Search-R1_7b
# MODEL_PATH=/home/sxjiang/model/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-ppo

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"


# ####################################################################################################################
# 2025-09-10

# MODE=budget_no_limit_run
# MODEL=ToRL-1.5B
# MODEL_PATH=/home/sxjiang/model/ToRL-1.5B

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=ToRL-7B
# MODEL_PATH=/home/sxjiang/model/ToRL-7B

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_7b_clip_radio_gradclip_02_one_epoch_down_progressive_2_seq_mean_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-7B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_one_epoch_no_warm_up_down_progressive_2_seq_mean-global_step_78

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# ####################################################################################################################
# 2025-09-13

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_sft
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_7b_sft
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-7B-Instruct-final_sft_edition10-52

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=ARPO_3b_sft
# MODEL_PATH=/home/sxjiang/myproject/agent/ARPO-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-arpo_final_sft_edition10-52

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"


# MODE=budget_no_limit_run
# MODEL=ReSearch-7b
# MODEL_PATH=/home/sxjiang/model/ReSearch-Qwen-7B-Instruct

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"


# ####################################################################################################################
# 2025-09-15

# MODE=budget_no_limit_run
# MODEL=CIR-7B-origin_prompt
# MODEL_PATH=/home/sxjiang/model/Qwen-Math-7B-CIR

# TASK=math
# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"


# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_g16W_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_one_epoch_no_warm_up_down_progressive_seq_mean_g16W-global_step_78

# TASK=math
# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"


# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_g16W_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_one_epoch_no_warm_up_down_progressive_seq_mean_g16W-global_step_78

# DATASET_NAMEs=(nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"


# MODE=budget_no_limit_run
# MODEL=tool_star_3b_two_epoch_add_test
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_two_epoch_no_warm_up_down_progressive_seq_mean_smooth_add-global_step_110

# TASK=math
# DATASET_NAMEs=(gsm8k)
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(bamboogle 2wiki)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"


# MODE=budget_no_limit_run
# MODEL=tool_star_3b_origin_test
# MODEL_PATH=/home/sxjiang/model/Tool-Star-Qwen-3B

# TASK=math
# DATASET_NAMEs=(gsm8k math500)
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(2wiki)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"


# ####################################################################################################################
# 2025-09-17


# MODE=budget_no_limit_run
# MODEL=Search-R1_7b_origin_prompt
# MODEL_PATH=/home/sxjiang/model/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-ppo

# TASK=math
# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"


# ####################################################################################################################
# 2025-09-18

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_fix_cost_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_one_epoch_no_warm_up_down_progressive_seq_mean_fix_cost-global_step_78

# TASK=math
# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"


# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_no_entropy_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_one_epoch_no_warm_up_down_progressive_seq_mean_no_entropy-global_step_78

# TASK=math
# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_7b_origin
# MODEL_PATH=/home/sxjiang/model/Tool-Star-Qwen-7B
# DATASET_NAMEs=(beerqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_7b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_new_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-7B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_one_epoch_no_warm_up_down_progressive_seq_mean_smooth_multiply-global_step_78
# DATASET_NAMEs=(beerqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_origin_gpu18
# MODEL_PATH=/home/sxjiang/model/Tool-Star-Qwen-3B
# DATASET_NAMEs=(beerqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_two_epoch_down_progressive_seq_mean_new_add_global_step_110
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_two_epoch_no_warm_up_down_progressive_seq_mean_smooth_add-global_step_110
# DATASET_NAMEs=(beerqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# ####################################################################################################################
# 2025-09-19

# MODE=budget_no_limit_run
# MODEL=ARPO_3b
# MODEL_PATH=/home/sxjiang/model/Qwen2.5-3B-ARPO
# DATASET_NAMEs=(beerqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=ARPO_3b_oct_clip_radio_gradclip_02_two_epoch_down_progressive_seq_mean_smooth_global_step_156
# MODEL_PATH=/home/sxjiang/myproject/agent/ARPO-OCT/transfer_checkpoints/ARPO_3B_global_16_init_8_beam_2_random_0_arpo_0.2_two_epoch_entropy_oct_downprogressive_em_score_seq_mean_specific_smooth-global_step_156
# DATASET_NAMEs=(beerqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=ARPO_7b
# MODEL_PATH=/home/sxjiang/model/Qwen2.5-7B-ARPO
# DATASET_NAMEs=(beerqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=ARPO_7b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_em_score_seq_mean_smooth_origin_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/ARPO-OCT/transfer_checkpoints/ARPO_7B_global_16_init_8_beam_2_random_0_arpo_0.2_one_epoch_entropy_oct_downprogressive_em_score_seq_mean_specific_smooth-origin-global_step_78
# DATASET_NAMEs=(beerqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=qwen2p5_instruct_3b
# MODEL_PATH=/home/sxjiang/model/Qwen2.5-3B-Instruct
# DATASET_NAMEs=(beerqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=qwen2p5_instruct_7b
# MODEL_PATH=/home/sxjiang/model/Qwen2.5-7B-Instruct
# DATASET_NAMEs=(beerqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=Search-R1_3b_origin_prompt
# MODEL_PATH=/home/sxjiang/model/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo
# DATASET_NAMEs=(beerqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=Search-R1_7b_origin_prompt
# MODEL_PATH=/home/sxjiang/model/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-ppo
# DATASET_NAMEs=(beerqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=ReSearch-7b_origin_prompt
# MODEL_PATH=/home/sxjiang/model/ReSearch-Qwen-7B-Instruct
# DATASET_NAMEs=(beerqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=auto-tir_7b_prompt
# MODEL_PATH=/home/sxjiang/model/AutoTIR-Qwen2.5-7B-Instruct
# DATASET_NAMEs=(beerqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

##########################################


# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_7b_origin
# MODEL_PATH=/home/sxjiang/model/Tool-Star-Qwen-7B
# DATASET_NAMEs=(beerqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_7b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_new_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-7B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_one_epoch_no_warm_up_down_progressive_seq_mean_smooth_multiply-global_step_78
# DATASET_NAMEs=(beerqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_origin_gpu18
# MODEL_PATH=/home/sxjiang/model/Tool-Star-Qwen-3B
# DATASET_NAMEs=(beerqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_two_epoch_down_progressive_seq_mean_new_add_global_step_110
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_two_epoch_no_warm_up_down_progressive_seq_mean_smooth_add-global_step_110
# DATASET_NAMEs=(beerqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=ARPO_3b
# MODEL_PATH=/home/sxjiang/model/Qwen2.5-3B-ARPO
# DATASET_NAMEs=(beerqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=ARPO_3b_oct_clip_radio_gradclip_02_two_epoch_down_progressive_seq_mean_smooth_global_step_156
# MODEL_PATH=/home/sxjiang/myproject/agent/ARPO-OCT/transfer_checkpoints/ARPO_3B_global_16_init_8_beam_2_random_0_arpo_0.2_two_epoch_entropy_oct_downprogressive_em_score_seq_mean_specific_smooth-global_step_156
# DATASET_NAMEs=(beerqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=ARPO_7b
# MODEL_PATH=/home/sxjiang/model/Qwen2.5-7B-ARPO
# DATASET_NAMEs=(beerqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=ARPO_7b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_em_score_seq_mean_smooth_origin_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/ARPO-OCT/transfer_checkpoints/ARPO_7B_global_16_init_8_beam_2_random_0_arpo_0.2_one_epoch_entropy_oct_downprogressive_em_score_seq_mean_specific_smooth-origin-global_step_78
# DATASET_NAMEs=(beerqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# ####################################################################################################################
# 2025-09-21

# MODE=budget_no_limit_run
# MODEL=ReSearch-7b_origin_prompt
# MODEL_PATH=/home/sxjiang/model/ReSearch-Qwen-7B-Instruct
# DATASET_NAMEs=(OlymBench-math)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

##############################

# MODE=budget_no_limit_run
# MODEL=ARPO_7b
# MODEL_PATH=/home/sxjiang/model/Qwen2.5-7B-ARPO
# DATASET_NAMEs=(mintqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=ARPO_7b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_em_score_seq_mean_smooth_origin_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/ARPO-OCT/transfer_checkpoints/ARPO_7B_global_16_init_8_beam_2_random_0_arpo_0.2_one_epoch_entropy_oct_downprogressive_em_score_seq_mean_specific_smooth-origin-global_step_78
# DATASET_NAMEs=(mintqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_7b_origin
# MODEL_PATH=/home/sxjiang/model/Tool-Star-Qwen-7B
# DATASET_NAMEs=(mintqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_7b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_new_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-7B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_one_epoch_no_warm_up_down_progressive_seq_mean_smooth_multiply-global_step_78
# DATASET_NAMEs=(mintqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=ARPO_3b
# MODEL_PATH=/home/sxjiang/model/Qwen2.5-3B-ARPO
# DATASET_NAMEs=(mintqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=ARPO_3b_oct_clip_radio_gradclip_02_two_epoch_down_progressive_seq_mean_smooth_global_step_156
# MODEL_PATH=/home/sxjiang/myproject/agent/ARPO-OCT/transfer_checkpoints/ARPO_3B_global_16_init_8_beam_2_random_0_arpo_0.2_two_epoch_entropy_oct_downprogressive_em_score_seq_mean_specific_smooth-global_step_156
# DATASET_NAMEs=(mintqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_origin_gpu18
# MODEL_PATH=/home/sxjiang/model/Tool-Star-Qwen-3B
# DATASET_NAMEs=(mintqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_two_epoch_down_progressive_seq_mean_new_add_global_step_110
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_two_epoch_no_warm_up_down_progressive_seq_mean_smooth_add-global_step_110
# DATASET_NAMEs=(mintqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=qwen2p5_instruct_3b
# MODEL_PATH=/home/sxjiang/model/Qwen2.5-3B-Instruct
# DATASET_NAMEs=(mintqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=qwen2p5_instruct_7b
# MODEL_PATH=/home/sxjiang/model/Qwen2.5-7B-Instruct
# DATASET_NAMEs=(mintqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=auto-tir_7b_prompt
# MODEL_PATH=/home/sxjiang/model/AutoTIR-Qwen2.5-7B-Instruct
# DATASET_NAMEs=(mintqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=Search-R1_3b_origin_prompt
# MODEL_PATH=/home/sxjiang/model/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo
# DATASET_NAMEs=(mintqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=Search-R1_7b_origin_prompt
# MODEL_PATH=/home/sxjiang/model/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-ppo
# DATASET_NAMEs=(mintqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=ReSearch-7b_origin_prompt
# MODEL_PATH=/home/sxjiang/model/ReSearch-Qwen-7B-Instruct
# DATASET_NAMEs=(mintqa)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

##############################

# MODE=budget_no_limit_run
# MODEL=ARPO_7b
# MODEL_PATH=/home/sxjiang/model/Qwen2.5-7B-ARPO
# DATASET_NAMEs=(squad)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=ARPO_7b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_em_score_seq_mean_smooth_origin_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/ARPO-OCT/transfer_checkpoints/ARPO_7B_global_16_init_8_beam_2_random_0_arpo_0.2_one_epoch_entropy_oct_downprogressive_em_score_seq_mean_specific_smooth-origin-global_step_78
# DATASET_NAMEs=(squad)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=ARPO_3b
# MODEL_PATH=/home/sxjiang/model/Qwen2.5-3B-ARPO
# DATASET_NAMEs=(squad)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=ARPO_3b_oct_clip_radio_gradclip_02_two_epoch_down_progressive_seq_mean_smooth_global_step_156
# MODEL_PATH=/home/sxjiang/myproject/agent/ARPO-OCT/transfer_checkpoints/ARPO_3B_global_16_init_8_beam_2_random_0_arpo_0.2_two_epoch_entropy_oct_downprogressive_em_score_seq_mean_specific_smooth-global_step_156
# DATASET_NAMEs=(squad)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_7b_origin_gpu18
# MODEL_PATH=/home/sxjiang/model/Tool-Star-Qwen-7B
# DATASET_NAMEs=(squad)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_7b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_new_global_step_78_gpu18
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-7B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_one_epoch_no_warm_up_down_progressive_seq_mean_smooth_multiply-global_step_78
# DATASET_NAMEs=(squad)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# ####################################################################################################################
# 2025-09-23

# MODE=budget_no_limit_run
# MODEL=torl_qwen_instruct_7b
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-7B-Instruct-zero-torl-200step
# DATASET_NAMEs=(OlymBench-math)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

##################################

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_origin_gpu18
# MODEL_PATH=/home/sxjiang/model/Tool-Star-Qwen-3B
# DATASET_NAMEs=(squad)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_two_epoch_down_progressive_seq_mean_new_add_global_step_110
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_two_epoch_no_warm_up_down_progressive_seq_mean_smooth_add-global_step_110
# DATASET_NAMEs=(squad)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=qwen2p5_instruct_3b
# MODEL_PATH=/home/sxjiang/model/Qwen2.5-3B-Instruct
# DATASET_NAMEs=(squad)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=qwen2p5_instruct_7b
# MODEL_PATH=/home/sxjiang/model/Qwen2.5-7B-Instruct
# DATASET_NAMEs=(squad)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=Search-R1_3b_origin_prompt
# MODEL_PATH=/home/sxjiang/model/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo
# DATASET_NAMEs=(squad)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=Search-R1_7b_origin_prompt
# MODEL_PATH=/home/sxjiang/model/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-ppo
# DATASET_NAMEs=(squad)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=ReSearch-7b_origin_prompt
# MODEL_PATH=/home/sxjiang/model/ReSearch-Qwen-7B-Instruct
# DATASET_NAMEs=(squad)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=auto-tir_7b_prompt
# MODEL_PATH=/home/sxjiang/model/AutoTIR-Qwen2.5-7B-Instruct
# DATASET_NAMEs=(squad)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"


###############

# MODE=budget_no_limit_run
# MODEL=torl_qwen_instruct_7b_step150
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-7B-Instruct-zero-torl-150step
# DATASET_NAMEs=(gsm8k)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=cir_qwen_instruct_7b_step150
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-7B-Instruct-Zero-CIR-global_step_150
# DATASET_NAMEs=(aime25 math gsm8k OlymBench-math)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=cir_qwen_instruct_7b_step150
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-7B-Instruct-Zero-CIR-global_step_150
# DATASET_NAMEs=(musique bamboogle 2wiki beerqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# ####################################################################################################################
# 2025-09-26

# MODE=budget_no_limit_run
# MODEL=cir_qwen_instruct_7b_new_step140
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-7B-Instruct-Zero-CIR-New-global_step_140
# DATASET_NAMEs=(musique bamboogle 2wiki beerqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(aime25 math gsm8k OlymBench-math)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# ####################################################################################################################
# 2025-09-28

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_smooth_multiply_times_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_one_epoch_no_warm_up_down_progressive_seq_mean_smooth_multiply_times-global_step_78
# DATASET_NAMEs=(beerqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_smooth_multiply__no_optim_cost_estimate_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_one_epoch_no_warm_up_down_progressive_seq_mean_smooth_multiply_no_optim_cost_estimate-global_step_78
# DATASET_NAMEs=(beerqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_no_oct_global_step_78
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_one_epoch_no_warm_up_down_progressive_seq_mean_no_oct-global_step_78
# DATASET_NAMEs=(beerqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# ####################################################################################################################
# 2025-10-01

# MODE=budget_no_limit_run
# MODEL=cir_qwen_instruct_3b_new_step140
# MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-Zero-CIR-New-global_step_140
# DATASET_NAMEs=(musique bamboogle 2wiki beerqa)
# TASK=qa
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

# DATASET_NAMEs=(aime25 math gsm8k OlymBench-math)
# TASK=math
# run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"


MODE=budget_no_limit_run
MODEL=cir_qwen_instruct_3b_new_step120
MODEL_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.2_one_epoch_no_warm_up_no_progressive_seq_mean-no_oct_first_oct_second-global_step_78
DATASET_NAMEs=(gsm8k aime25 math OlymBench-math)
TASK=math
run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"

DATASET_NAMEs=(musique bamboogle 2wiki beerqa)
TASK=qa
run_evaluation "$MODEL_PATH" "$MODEL" "$TASK" "$MODE" "${DATASET_NAMEs[@]}"
