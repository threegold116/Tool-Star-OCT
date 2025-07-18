data_name=toolstar_mix_train_
export PYTHONPATH=/src/verl:$PYTHONPATH
# export MKL_SERVICE_FORCE_INTEL=1
# export MKL_THREADING_LAYER=GNU
export DATA_PATH=/home/sxjiang/myproject/agent/Tool-Star-OCT/data/
export WANDB_MODE=offline
export BASE_MODEL='/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52'
export EXPERIMENT_NAME=Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug
export WAND_PROJECT="Tool-Star-OCT"
bash scripts/train/train.sh \
    --train_batch_size 128 \
    --ppo_mini_batch_size 16 \
    --rollout_n 8 \
    --apply_chat True \
    --prompt_template_name re_search_template_sys \
    --actor_model_path $BASE_MODEL \
    --project_name toolstar \
    --experiment_name $EXPERIMENT_NAME \
    --nnodes 1 \
    --n_gpus_per_node 8 \
    --save_freq 10 \
    --test_freq 10 \
    --total_epochs 2 \
    --save_path /home/sxjiang/myproject/agent/Tool-Star-OCT/verl_checkpoints/$EXPERIMENT_NAME \
    --train_files $DATA_PATH/grpo_mix_train_shuffle.parquet \
    --test_files $DATA_PATH/grpo_mix_test.parquet \
    # --wandb_api_key {your_wandb_api_key} \
