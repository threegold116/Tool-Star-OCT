data_name=toolstar_mix_train_
export PYTHONPATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/Tool_Star_RL/src/verl:$PYTHONPATH
# export MKL_SERVICE_FORCE_INTEL=1
# export MKL_THREADING_LAYER=GNU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/toolstar/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export DATA_PATH=//home/sxjiang/myproject/agent/Tool-Star-OCT/Tool_Star_RL/mix_grpo/
export WANDB_MODE=offline
export BASE_MODEL='/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-7B-Instruct-final_sft_edition10-52'
export EXPERIMENT_NAME=Qwen2.5-7B-Instruct-final_sft_edition10-52-grpo_debug-bz_128
export WAND_PROJECT="Tool-Star-OCT"
export RAY_DEBUG_MODE="123"
bash scripts/train/train.sh\
    --train_batch_size 128 \
    --ppo_mini_batch_size 16\
    --rollout_n 8 \
    --apply_chat True \
    --oct_penalty budget\
    --no_positive_penalty True\
    --prompt_template_name re_search_template_with_budget_sys \
    --actor_model_path $BASE_MODEL \
    --project_name toolstar \
    --experiment_name $EXPERIMENT_NAME \
    --nnodes 1 \
    --n_gpus_per_node 8 \
    --search_mode wikipedia \
    --save_freq 10 \
    --test_freq 200 \
    --total_epochs 1 \
    --save_path /share/home/gtang/sxjiang-gt/Tool-Star-OCT/Tool_Star_RL/verl_checkpoints/$EXPERIMENT_NAME \
    --train_files $DATA_PATH/rl_data_auto_verify_qwen3_32b.parquet \
    --test_files $DATA_PATH/grpo_mix_test.parquet \
    --gup_memory_utilization 0.8 \
    --top_n 3\
    --max_calling_times 5 \
    --lr_warmup_steps_ratio -1\
    --mix_rules True \
    --qa_rule em_score \
    --math_rule em_score \
    --binary_f1_threshold 0.5 \
    --is_multi_tool False \
    --progressive_calling_times_stages 3 \
    --radio_clip False \
    --use_oct_cofficient False \
    --use_oct_cofficient_advantage_shaping True \
    --optim_cost_estimate True \
    --clip_ratio_high 0.2\
    --apply_mode "multiply"\
    --loss_agg_mode seq-mean-token-mean\
    --group_smooth False\
    --normlization_mode group_normlization\
    # --wandb_api_key {your_wandb_api_key} \
    

