data_name=toolstar_mix_train_
export PYTHONPATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/Tool_Star_RL/src/verl:$PYTHONPATH
# export MKL_SERVICE_FORCE_INTEL=1
# export MKL_THREADING_LAYER=GNU
export CUDA_VISIBLE_DEVICES=0,1
export DATA_PATH=//home/sxjiang/myproject/agent/Tool-Star-OCT/Tool_Star_RL/mix_grpo/
export WANDB_MODE=offline
export BASE_MODEL='/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52'
export EXPERIMENT_NAME=Qwen2.5-3B-Instruct-final_sft_edition10-52-random_budget-local_search-grpo-mix_rules-qa_mode-is_multi_tool_false-oct_budget-progressive_2_4-bz_32-x
export WAND_PROJECT="Tool-Star-OCT"
export RAY_DEBUG_MODE="2"
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues
bash scripts/train/train.sh \
    --train_batch_size 2 \
    --ppo_mini_batch_size 2 \
    --rollout_n 2 \
    --apply_chat True \
    --prompt_template_name re_search_template_with_budget_sys \
    --actor_model_path $BASE_MODEL \
    --project_name toolstar \
    --experiment_name $EXPERIMENT_NAME \
    --nnodes 1 \
    --n_gpus_per_node 2 \
    --search_mode wikipedia \
    --save_freq 100 \
    --test_freq 100 \
    --total_epochs 2 \
    --save_path //home/sxjiang/myproject/agent/Tool-Star-OCT/Tool_Star_RL/verl_checkpoints/$EXPERIMENT_NAME \
    --train_files $DATA_PATH/grpo_mix_train_shuffle.parquet \
    --test_files $DATA_PATH/grpo_mix_test.parquet \
    --top_n 3\
    --max_calling_times 3 \
    --mix_rules True \
    --qa_rule em_score \
    --is_multi_tool False \
    --progressive_calling_times_stages 3 \
    # --wandb_api_key {your_wandb_api_key} \
    

