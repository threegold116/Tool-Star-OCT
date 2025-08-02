import jsonlines
import os
import json
import re
rollout_dir = "/share/home/sxjiang/myproject/Tool-Star-OCT/Tool_Star_RL/verl_checkpoints"

for root, dirs, files in os.walk(rollout_dir, topdown=False):
    for file in files:
        if file.endswith("jsonl"):
            print(file)
    with jsonlines.open(os.path.join(data_dir, rollout_file)) as reader:
        for line in reader:
            count += 1
            score_sum += line["score"]
            max_calling_times = max(max_calling_times, line["is_search"]+line["is_python"])
            max_length = max(max_length, len(line["sequences_str"]))
            questions.append(get_question(line["sequences_str"]))