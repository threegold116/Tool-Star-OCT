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
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki hle gaia webwalker)

# MODE="budget_limit_run_10"
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
# done

# MODEL="tool_star_qwen_3b_oct_clip_radio_gradclip_028_one_epoch_new_origin_no_warm_up_epoch1_new_no_warmup_oct_epoch1_step_60"
# MODE="budget_no_limit_run"
# TASK="qa"
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki hle gaia webwalker)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE="budget_no_limit_run"
# TASK="math"
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done


# MODEL="tool_star_qwen_3b_oct_clip_radio_gradclip_028_one_epoch_new_no_warm_up_f1_score_global_step_78"
# MODE="budget_no_limit_run"
# TASK="qa"
# DATASET_NAMEs=(webwalker musique bamboogle hotpotqa 2wiki)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODEL="tool_star_qwen_3b_oct_clip_radio_gradclip_028_one_epoch_new_no_warm_up_f1_score_global_step_78"
# MODE="budget_no_limit_run"
# TASK="math"
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done


# MODEL=""
# MODE="budget_no_limit_run"
# TASK="math"
# DATASET_NAMEs=(math math500 gsm8k)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

####################################################################################
# # 先测times作为惩罚
# MODEL="tool_star_qwen_3b_oct_clip_radio_gradclip_028_one_epoch_new_times_global_step_78"
# MODE="budget_no_limit_run"
# TASK="qa"
# DATASET_NAMEs=(webwalker musique bamboogle hotpotqa 2wiki)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODEL="tool_star_qwen_3b_oct_clip_radio_gradclip_028_one_epoch_new_times_global_step_78"
# MODE="budget_no_limit_run"
# TASK="math"
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# #################################################################################
# # 测budget作为惩罚
# MODEL="tool_star_qwen_3b_oct_clip_radio_gradclip_028_one_epoch_new_budget_global_step_78"
# MODE="budget_no_limit_run"
# TASK="qa"
# DATASET_NAMEs=(webwalker musique bamboogle hotpotqa 2wiki)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODEL="tool_star_qwen_3b_oct_clip_radio_gradclip_028_one_epoch_new_budget_global_step_78"
# MODE="budget_no_limit_run"
# TASK="math"
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# #############################################################################
# # 测origin的gpqa的表现
# MODEL="tool_star_qwen_3b_origin_wiki"
# MODE="budget_no_limit_run"
# TASK="qa"
# DATASET_NAMEs=(gpqa)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# # 测oct的gpqa的表现
# MODEL="tool_star_qwen_3b_oct_clip_radio_gradclip_028_one_epoch_no_warmup_new_wiki_global_step_78"
# MODE="budget_no_limit_run"
# TASK="qa"
# DATASET_NAMEs=(gpqa)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# #############################################################################
# 测budget_limit_run_10情况下，数学数据集的表现
# MODEL="tool_star_qwen_3b_sft"
# MODE="budget_limit_run_10"
# TASK="math"
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODEL=""
# MODE="tool_star_qwen_3b_oct_clip_radio_gradclip_032_one_epoch_no_progressive_seq_mean_new_global_step_78"
# TASK="qa"
# DATASET_NAMEs=(webwalker musique bamboogle hotpotqa 2wiki)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done
# TASK="math"
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done


# MODEL="tool_star_qwen_3b_oct_clip_radio_gradclip_028_one_epoch_new_origin_no_warm_up_global_step_78"
# MODE="budget_limit_run_10"
# TASK="math"
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# #############################################################################
# MODEL="ARPO_3b"
# MODE="budget_no_limit_run"
# TASK="qa"
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODEL="ARPO_3b"
# MODE="budget_no_limit_run"
# TASK="math"
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# # #############################################################################
# MODEL="auto-tir_7b"
# MODE="budget_no_limit_run"
# TASK="qa"
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODEL="auto-tir_7b"
# MODE="budget_no_limit_run"
# TASK="math"
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# #############################################################################
# 2025-08-20

# MODEL="tool_star_qwen_3b_oct_clip_radio_gradclip_032_one_epoch_no_progressive_seq_mean_new_times_global_step_78"
# MODE="budget_no_limit_run"
# TASK="qa"
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODEL="tool_star_qwen_3b_oct_clip_radio_gradclip_032_one_epoch_no_progressive_seq_mean_new_times_global_step_78"
# MODE="budget_no_limit_run"
# TASK="math"
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# #############################################################################
# 2025-08-21

# MODEL="tool_star_qwen_3b_oct_clip_radio_gradclip_032_one_epoch_no_progressive_seq_mean_new_negative_global_step_78"
# MODE="budget_no_limit_run"
# TASK="qa"
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODEL="tool_star_qwen_3b_oct_clip_radio_gradclip_032_one_epoch_no_progressive_seq_mean_new_negative_global_step_78"
# MODE="budget_no_limit_run"
# TASK="math"
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODEL="auto-tir_7b_prompt"
# MODE="budget_no_limit_run"
# TASK="qa"
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODEL="auto-tir_7b_prompt"
# MODE="budget_no_limit_run"
# TASK="math"
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODEL="ARPO_7b"
# MODE="budget_no_limit_run"
# TASK="qa"
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODEL="ARPO_7b"
# MODE="budget_no_limit_run"
# TASK="math"
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# #############################################################################
# 2025-08-22

# MODEL="tool_star_qwen_3b_oct_clip_radio_gradclip_02_one_epoch_no_progressive_seq_mean_new_global_step_78"
# MODE="budget_no_limit_run"
# TASK="qa"
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODEL="tool_star_qwen_3b_oct_clip_radio_gradclip_02_one_epoch_no_progressive_seq_mean_new_global_step_78"
# MODE="budget_no_limit_run"
# TASK="math"
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# #############################################################################
# 2025-08-22

# MODEL="tool_star_qwen_3b_oct_clip_radio_gradclip_032_one_epoch_no_progressive_seq_mean_new_add-oct_global_step_78"
# MODE="budget_no_limit_run"
# TASK="qa"
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODEL="tool_star_qwen_3b_oct_clip_radio_gradclip_032_one_epoch_no_progressive_seq_mean_new_add-oct_global_step_78"
# MODE="budget_no_limit_run"
# TASK="math"
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODEL="qwen2p5_instruct_3b"
# MODE="budget_no_limit_run"
# TASK="qa"
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODEL="qwen2p5_instruct_3b"
# MODE="budget_no_limit_run"
# TASK="math"
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODEL="qwen2p5_instruct_7b"
# MODE="budget_no_limit_run"
# TASK="qa"
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODEL="qwen2p5_instruct_7b"
# MODE="budget_no_limit_run"
# TASK="math"
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# #############################################################################
# 2025-08-24

# MODEL="tool_star_qwen_3b_oct_clip_radio_gradclip_032_one_epoch_down_progressive_seq_mean_new_global_step_78"
# MODE="budget_no_limit_run"
# TASK="qa"
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODEL="tool_star_qwen_3b_oct_clip_radio_gradclip_032_one_epoch_down_progressive_seq_mean_new_global_step_78"
# MODE="budget_no_limit_run"
# TASK="math"
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODEL="tool_star_qwen_3b_sft"
# MODE="budget_limit_run_5"
# TASK="qa"
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODEL="tool_star_qwen_3b_sft"
# MODE="budget_limit_run_5"
# TASK="math"
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODEL="tool_star_qwen_3b_oct_clip_radio_gradclip_032_one_epoch_no_progressive_seq_mean_new_global_step_78"
# MODE="budget_limit_run_5"
# TASK="qa"
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# TASK="math"
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODEL="tool_star_qwen_3b_oct_clip_radio_gradclip_032_one_epoch_no_progressive_seq_mean_new_group_smooth_global_step_78"
# MODE="budget_no_limit_run"
# TASK="qa"
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# TASK="math"
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# #############################################################################
# 2025-08-25

# MODEL="tool_star_qwen_3b_oct_clip_radio_gradclip_02_one_epoch_no_progressive_seq_mean_new_global_step_78"
# MODE="budget_limit_run_5"
# TASK="qa"
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# TASK="math"
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODEL="tool_star_qwen_3b_origin"
# MODE="budget_no_limit_run"

# TASK="math"
# DATASET_NAMEs=(amc23)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODEL="tool_star_qwen_3b_oct_clip_radio_gradclip_02_one_epoch_no_progressive_seq_mean_new_global_step_78"
# MODE="budget_no_limit_run_t6_p95"

# TASK="math"
# DATASET_NAMEs=(amc23)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODEL="tool_star_qwen_3b_oct_clip_radio_gradclip_032_one_epoch_no_progressive_seq_mean_new_global_step_78"
# MODE="budget_no_limit_run_t6_p95"

# TASK="math"
# DATASET_NAMEs=(amc23)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# #############################################################################
# 2025-08-26

# MODEL="tool_star_qwen_3b_origin_gpu18"
# MODE="budget_no_limit_run"
# TASK="qa"
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# TASK="math"
# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODEL="tool_star_qwen_3b_origin_gpu09"
# MODE="no-tool-inference"
# TASK="qa"
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# TASK="math"
# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE="budget_no_limit_run"
# MODEL="tool_star_qwen_3b_oct_clip_radio_gradclip_032_one_epoch_no_progressive_seq_mean_new_global_step_78"
# TASK="qa"
# DATASET_NAMEs=(nq SimpleQA)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# TASK="math"
# DATASET_NAMEs=(amc23 OlymMATH-EASY OlymMATH-HARD)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE="budget_no_limit_run"
# MODEL="tool_star_qwen_3b_oct_clip_radio_gradclip_02_one_epoch_no_progressive_seq_mean_new_global_step_78"
# TASK="qa"
# DATASET_NAMEs=(nq SimpleQA)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# TASK="math"
# DATASET_NAMEs=(amc23 OlymMATH-EASY OlymMATH-HARD)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE="budget_no_limit_run"
# MODEL="tool_star_qwen_3b_oct_clip_radio_gradclip_032_one_epoch_down_progressive_seq_mean_new2_global_step_78"
# TASK="qa"
# DATASET_NAMEs=(nq SimpleQA)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# TASK="math"
# DATASET_NAMEs=(amc23 OlymMATH-EASY OlymMATH-HARD)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE="budget_no_limit_run"
# MODEL="tool_star_qwen_3b_oct_clip_radio_gradclip_028_one_epoch_no_warmup_new_global_step_78"
# TASK="qa"
# DATASET_NAMEs=(nq SimpleQA)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# TASK="math"
# DATASET_NAMEs=(amc23 OlymMATH-EASY OlymMATH-HARD)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODEL="tool_star_qwen_3b_oct_clip_radio_gradclip_02_one_epoch_no_progressive_seq_mean_new_global_step_78_gpu09"
# MODE="no-tool-inference"
# TASK="qa"
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# TASK="math"
# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODEL="tool_star_qwen_3b_oct_clip_radio_gradclip_032_one_epoch_no_progressive_seq_mean_new_group_smooth_global_step_78"
# MODE="budget_no_limit_run"
# TASK="qa"
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# TASK="math"
# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# #############################################################################
# 2025-08-27

# MODEL="tool_star_qwen_3b_oct_clip_radio_gradclip_02_one_epoch_no_progressive_token_mean_new_global_step_78"
# MODE="budget_no_limit_run"
# TASK="qa"
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# TASK="math"
# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODEL="tool_star_qwen_3b_oct_clip_radio_gradclip_02_one_epoch_no_progressive_seq_mean_new_global_step_78"
# MODE="budget_no_limit_run_t10_p95_new"
# TASK="math"

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE="budget_no_limit_run_t10_p95_new"
# MODEL="tool_star_qwen_3b_origin"
# TASK="math"

# DATASET_NAMEs=(aime24 aime25 math gsm8k amc23 OlymMATH-EASY OlymMATH-HARD)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODEL="tool_star_qwen_3b_oct_clip_radio_gradclip_02_one_epoch_no_progressive_seq_mean_new_add_global_step_78"
# MODE="budget_no_limit_run"
# TASK="qa"
# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# TASK="math"
# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE="budget_no_limit_run_t10_p95_new"
# MODEL="tool_star_qwen_3b_oct_clip_radio_gradclip_02_one_epoch_no_progressive_seq_mean_new_add_global_step_78"
# TASK="math"

# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k amc23 OlymMATH-EASY OlymMATH-HARD)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# #############################################################################
# 2025-08-28

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_origin

# DATASET_NAMEs=(fanout)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(OlymBench-math OlymBench-physics)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=budget_no_limit_run_t10_p95_new
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_032_one_epoch_no_progressive_seq_mean_new_global_step_78

# TASK=math
# DATASET_NAMEs=(aime24 aime25 math math500 gsm8k amc23 OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=budget_no_limit_run_t10_p95_new
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_one_epoch_no_progressive_seq_mean_new_add_global_step_78

# TASK=math
# DATASET_NAMEs=(OlymBench-math)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=budget_no_limit_run_t10_p95_new
# MODEL=tool_star_qwen_3b_origin

# TASK=math
# DATASET_NAMEs=(OlymBench-math)
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_two_epoch_down_progressive_seq_mean_new_add_global_step_110

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=budget_no_limit_run_t10_p95_new
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_one_epoch_no_progressive_seq_mean_new_global_step_78

# DATASET_NAMEs=(OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=budget_no_limit_run_t10_p95_new
# MODEL=02_test_gpu18

# DATASET_NAMEs=(OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# #############################################################################
# 2025-08-29

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_new_global_step_78

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_7b_origin_gpu06

# DATASET_NAMEs=(musique)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(gsm8k)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_7b_origin

# DATASET_NAMEs=(nq SimpleQA)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=no-tool-inference_new
# MODEL=tool_star_qwen_3b_origin

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# #############################################################################
# 2025-08-30

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_smooth_binary_f1_global_step_78

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done


# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_6_seq_mean_smooth_global_step_78

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# #############################################################################
# 2025-08-31

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_smooth_binary_f1_math_global_step_78

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_one_epoch_no_progressive_seq_mean_origin_no_multi_reward_global_step_78

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done


# #############################################################################
# 2025-09-01

# MODE=budget_limit_run_5
# MODEL=tool_star_qwen_7b_origin

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done


# MODE=budget_limit_run_5
# MODEL=tool_star_qwen_7b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_new_global_step_78

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=no-tool-inference_origin_prompt
# MODEL=tool_star_qwen_3b_origin

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done


# MODE=no-tool-inference_origin_prompt
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_two_epoch_down_progressive_seq_mean_new_add_global_step_110

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done


# MODE=no-tool-inference_new
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_two_epoch_down_progressive_seq_mean_new_add_global_step_110

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# #############################################################################
# 2025-09-02

# MODE=budget_no_limit_run
# MODEL=ARPO_3b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_smooth_global_step_78

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=no-tool-inference_origin_prompt
# MODEL=tool_star_qwen_3b_sft

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_two_epoch_down_progressive_seq_mean_new_global_step_110

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_two_epoch_down_progressive_seq_mean_new_global_step_130

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=no-tool-inference_origin_prompt
# MODEL=tool_star_qwen_7b_origin

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done


# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_7b_oct_clip_radio_gradclip_02_two_epoch_down_progressive_seq_mean_new_global_step_80

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done


# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_7b_oct_clip_radio_gradclip_02_two_epoch_down_progressive_seq_mean_new_global_step_90

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=no-tool-inference_origin_prompt
# MODEL=tool_star_qwen_7b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_new_global_step_78

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# #############################################################################
# 2025-09-02

# MODE=no-tool-inference_origin_prompt
# MODEL=tool_star_qwen_3b_origin_greedy

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA logiqa)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=no-tool-inference_origin_prompt
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_two_epoch_down_progressive_seq_mean_new_add_global_step_110_greedy

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA logiqa)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done


# MODE=no-tool-inference_origin_prompt
# MODEL=tool_star_qwen_7b_origin_t10_p95

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA logiqa)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=no-tool-inference_origin_prompt
# MODEL=tool_star_qwen_7b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_new_global_step_78_t10_p95

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA logiqa)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=no-tool-inference_origin_prompt
# MODEL=tool_star_qwen_7b_origin_t10_p95

# DATASET_NAMEs=(logiqa)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=no-tool-inference_origin_prompt
# MODEL=tool_star_qwen_7b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_new_global_step_78_t10_p95

# DATASET_NAMEs=(logiqa)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# ####################################################################################################################
# 2025-09-04

# MODE=budget_no_limit_run
# MODEL=ARPO_3b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_em_score_seq_mean_smooth_origin_global_step_78

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA logiqa)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# ####################################################################################################################
# 2025-09-05

# MODE=budget_no_limit_run
# MODEL=ARPO_7b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_em_score_seq_mean_smooth_origin_global_step_78

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA logiqa)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=budget_no_limit_run
# MODEL=ARPO_3b

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA logiqa)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_no_oct_global_step_78

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA logiqa)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_smooth_multiply_times_global_step_78

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA logiqa)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=budget_no_limit_run
# MODEL=ARPO_7b

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA logiqa)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_7b_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_smooth_multiply_times_global_step_78

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA logiqa)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# ####################################################################################################################
# 2025-09-06

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_7b_oct_clip_radio_gradclip_02_one_epoch_no_progressive_seq_mean_smooth_multiply_global_step_78

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA logiqa)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=no-tool-inference_origin_prompt
# MODEL=tool_star_qwen_7b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_new_global_step_78
# DATASET_NAMEs=(nq logiqa)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODEL=tool_star_qwen_3b_sft
# DATASET_NAMEs=(logiqa)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODEL=tool_star_qwen_3b_origin_greedy
# DATASET_NAMEs=(logiqa)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_two_epoch_down_progressive_seq_mean_new_add_global_step_110_greedy
# DATASET_NAMEs=(logiqa)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODEL=tool_star_qwen_7b_origin
# DATASET_NAMEs=(logiqa)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_7b_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_no_oct_global_step_78

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymMATH-EASY OlymMATH-HARD OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# ####################################################################################################################
# 2025-09-07

# MODE=no-tool-inference_origin_prompt
# MODEL=ARPO_3b_sft_t10_p95_new

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA logiqa)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done


# MODE=no-tool-inference_origin_prompt
# MODEL=ARPO_7b_sft_t10_p95_new

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA logiqa)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done


# MODE=no-tool-inference_origin_prompt
# MODEL=ARPO_7b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_em_score_seq_mean_smooth_origin_global_step_78_t10_p95_new

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA logiqa)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done


# MODE=no-tool-inference_origin_prompt
# MODEL=ARPO_3b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_em_score_seq_mean_smooth_origin_global_step_78_t10_p95_new

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA logiqa)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done


# MODE=no-tool-inference_origin_prompt
# MODEL=ARPO_3b_t10_p95_new

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA logiqa)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done


# MODE=no-tool-inference_origin_prompt
# MODEL=ARPO_7b_t10_p95_new

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA logiqa)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done


# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_7b_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_global_step_70

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=no-tool-inference_origin_prompt
# MODEL=tool_star_qwen_7b_sft_t10_p95

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA logiqa)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_7b_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_global_step_70

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(aime25)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done


# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_smooth_multiply__no_optim_cost_estimate_global_step_78

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done


# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_7b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_smooth_multiply__no_optim_cost_estimate_global_step_78

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done


# MODE=no-tool-inference_origin_prompt
# MODEL=tool_star_qwen_7b_sft_t10_p95

# DATASET_NAMEs=(logiqa)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# ####################################################################################################################
# 2025-09-08

# MODE=no-tool-inference_origin_prompt
# MODEL=ARPO_7b_sft_t10_p95_new

# DATASET_NAMEs=(logiqa)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=no-tool-inference_origin_prompt
# MODEL=tool_star_qwen_7b_sft_t10_p95

# DATASET_NAMEs=(logiqa)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=budget_no_limit_run
# MODEL=auto-tir_7b_prompt

# DATASET_NAMEs=(nq SimpleQA)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=budget_no_limit_run
# MODEL=qwen2p5_instruct_3b

# DATASET_NAMEs=(nq SimpleQA)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=budget_no_limit_run
# MODEL=qwen2p5_instruct_7b

# DATASET_NAMEs=(nq SimpleQA)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_new2_global_step_78

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=no-tool-inference_origin_prompt
# MODEL=tool_star_qwen_3b_oct_clip_radio_gradclip_02_two_epoch_down_progressive_seq_mean_new_add_global_step_110_seed1234

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA logiqa)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# ####################################################################################################################
# 2025-09-09

# MODE=no-tool-inference_origin_prompt
# MODEL=tool_star_qwen_3b_sft_seed1234

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA logiqa)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done


# MODE=no-tool-inference_origin_prompt
# MODEL=ARPO_3b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_em_score_seq_mean_smooth_origin_global_step_78_t10_p95_new_seed1234

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA logiqa)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=no-tool-inference_origin_prompt
# MODEL=ARPO_3b_seed1234

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA logiqa)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=budget_no_limit_run
# MODEL=tool_star_qwen_7b_clip_radio_gradclip_02_one_epoch_down_progressive_2_seq_mean_global_step_70

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# ####################################################################################################################
# 2025-09-10

# MODE=budget_no_limit_run
# MODEL=ARPO_3b_oct_clip_radio_gradclip_02_two_epoch_down_progressive_seq_mean_smooth_global_step_156

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=budget_no_limit_run
# MODEL=Search-R1_3b

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=budget_no_limit_run
# MODEL=Search-R1_7b

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# MODE=no-tool-inference_origin_prompt
# MODEL=ARPO_3b_sft_seed1234

# DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA logiqa)
# TASK=qa
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

# DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
# TASK=math
# for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
#     run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
# done

MODE=budget_no_limit_run
MODEL=ToRL-1.5B

DATASET_NAMEs=(musique bamboogle hotpotqa 2wiki nq SimpleQA)
TASK=qa
for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
    run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
done

DATASET_NAMEs=(amc23 aime24 aime25 math math500 gsm8k OlymBench-math)
TASK=math
for DATASET_NAME in "${DATASET_NAMEs[@]}"; do
    run_dataset_model_mode $TASK $DATASET_NAME $MODEL $MODE
    
done