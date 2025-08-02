import jsonlines
import os
import json
import re
expertment_name="Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.28_grad_clip_epoch1_workerddied_test"
data_dir = f"/share/home/jfliang/Project/sxjiang/Tool-Star-OCT/Tool_Star_RL/verl_checkpoints/{expertment_name}/rollout"
specific_rollout_iter_num = -1


def get_question(sentence):
    sentence = sentence.split("<|im_end|>\n<|im_start|>assistant\n")[0]
    sentence = sentence.split("<|im_start|>user\n")[1]
    return sentence
def find_wrong(reson_str,sequences_str):
    
    if "bad format" not in reson_str:
        return False
    if sequences_str.count("</answer>")>3:
        return True
    if "<<" in sequences_str:
        return True
    if "</answer> not found" in reson_str:
        return False
    if sequences_str.count("</answer>")==2:
        return False
    if sequences_str.count("</answer>")==2 and sequences_str.count("<|im_end|>")==3:
        return False
    if sequences_str.count("</answer>")!=sequences_str.count("<|im_end|>"):
        return True
    
    return False

rollout_step2metrics={}
wrong_rollout_idx = []
# print(data_dir)
wrong_rollout_num_dict={}
for rollout_file in os.listdir(data_dir):
    rollout_step = rollout_file.split("_")[1].split(".")[0]
    score_sum = 0
    max_calling_times = 0
    max_length = 0
    count = 0
    questions = []
    wrong_rollout_num=0
    duplicate_rollout_num=0
    # print(rollout_file)
    if str(specific_rollout_iter_num) not in rollout_file and specific_rollout_iter_num != -1:
        continue
    with jsonlines.open(os.path.join(data_dir, rollout_file)) as reader:
        
        
        for line in reader:
            count += 1
            score_sum += line["score"]
            max_calling_times = max(max_calling_times, line["is_search"]+line["is_python"])
            max_length = max(max_length, len(line["sequences_str"]))
            questions.append(get_question(line["sequences_str"]))
            if find_wrong(line["reason"],line["sequences_str"]):
                wrong_rollout_num+=1
    rollout_step2metrics[int(rollout_step)] = {
        "score": score_sum / count, 
        "max_calling_times": max_calling_times, 
        "max_length": max_length, 
        "question": questions,
        "wrong_rollout_num": wrong_rollout_num,
        }
print(expertment_name)
import matplotlib.pyplot as plt
x = [int(i) for i in rollout_step2metrics.keys()]
x = sorted(x)
y = [rollout_step2metrics[i]["wrong_rollout_num"] for i in x]
result_dir = f"./analyse/{expertment_name}"
os.makedirs(result_dir, exist_ok=True)
plt.figure(figsize=(12, 5))  # 宽度=12，高度=5，单位是英寸
plt.plot(x, y, marker='o', label='Line')  # 画折线图并加点
max_x = x[y.index(max(y))]
max_y = max(y)
# 添加一条竖线
plt.axvline(x=max_x, color='red', linestyle='--', label='Max Value')

# 添加文字标注
plt.text(max_x, max_y + 1, f'Max: {max_x}', ha='center', color='red', fontsize=10)

plt.savefig(f"{result_dir}/wrong_rollout_num.png")
plt.close()

y = [rollout_step2metrics[i]["max_length"] for i in x]
plt.plot(x, y)
plt.savefig(f"{result_dir}/max_length.png")
plt.close()

# print(sorted(rollout_step2metrics.items(), key=lambda x: x[0]))
# with open(f"/share/home/sxjiang/myproject/Tool-Star-OCT/Tool_Star_RL/{expertment_name}_{str(specific_rollout_iter_num)}_rollout_step2metrics.json", "w") as f:
#     json.dump(sorted(rollout_step2metrics.items(), key=lambda x: x[0]), f,indent=4)
# with open(f"/share/home/sxjiang/myproject/Tool-Star-OCT/Tool_Star_RL/{expertment_name}_{str(specific_rollout_iter_num)}_rollout_step2metrics.json", "w") as f:
#     json.dump(sorted(rollout_step2metrics.items(), key=lambda x: x[0]), f,indent=4)
