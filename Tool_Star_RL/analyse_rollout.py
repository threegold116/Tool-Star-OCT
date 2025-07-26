import jsonlines
import os
import json
import re
expertment_name="Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.28_masked_mean_soft"
data_dir = f"/share/home/jfliang/Project/sxjiang/Tool-Star-OCT/Tool_Star_RL/verl_checkpoints/{expertment_name}/rollout"
specific_rollout_iter_num = 44


def get_question(sentence):
    sentence = sentence.split("<|im_end|>\n<|im_start|>assistant\n")[0]
    sentence = sentence.split("<|im_start|>user\n")[1]
    return sentence

rollout_step2metrics={}
for rollout_file in os.listdir(data_dir):
    rollout_step = rollout_file.split("_")[1].split(".")[0]
    score_sum = 0
    max_calling_times = 0
    max_length = 0
    count = 0
    questions = []
    print(rollout_file)
    if str(specific_rollout_iter_num) not in rollout_file and specific_rollout_iter_num != -1:
        continue
    with jsonlines.open(os.path.join(data_dir, rollout_file)) as reader:
        for line in reader:
            count += 1
            score_sum += line["score"]
            max_calling_times = max(max_calling_times, line["is_search"]+line["is_python"])
            max_length = max(max_length, len(line["sequences_str"]))
            questions.append(get_question(line["sequences_str"]))
    rollout_step2metrics[int(rollout_step)] = {"score": score_sum / count, "max_calling_times": max_calling_times, "max_length": max_length, "question": questions}

# print(sorted(rollout_step2metrics.items(), key=lambda x: x[0]))
with open(f"//share/home/jfliang/Project/sxjiang/Tool-Star-OCT/Tool_Star_RL/{expertment_name}_{str(specific_rollout_iter_num)}_rollout_step2metrics.json", "w") as f:
    json.dump(sorted(rollout_step2metrics.items(), key=lambda x: x[0]), f,indent=4)
