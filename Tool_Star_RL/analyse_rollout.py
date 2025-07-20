import jsonlines
import os
import json
data_dir = "/share/home/sxjiang/myproject/Tool-Star-OCT/Tool_Star_RL/verl_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128/rollout"


rollout_step2metrics={}
for rollout_file in os.listdir(data_dir):
    rollout_step = rollout_file.split("_")[1].split(".")[0]
    score_sum = 0
    max_calling_times = 0
    count = 0
    with jsonlines.open(os.path.join(data_dir, rollout_file)) as reader:
        for line in reader:
            count += 1
            score_sum += line["score"]
            max_calling_times = max(max_calling_times, line["is_search"]+line["is_python"])
    rollout_step2metrics[int(rollout_step)] = {"score": score_sum / count, "max_calling_times": max_calling_times}

print(sorted(rollout_step2metrics.items(), key=lambda x: x[0]))
with open("/share/home/sxjiang/myproject/Tool-Star-OCT/Tool_Star_RL/verl_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128/rollout_step2metrics.json", "w") as f:
    json.dump(sorted(rollout_step2metrics.items(), key=lambda x: x[0]), f,indent=4)
