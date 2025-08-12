import jsonlines
import os
import json
import re
# from analyse_research import get_search_questions
import sys
if len(sys.argv) > 1:
    expertment_name = sys.argv[1]
else:
    expertment_name="Qwen2.5-3B-Instruct-origin_epoch_1_new_no_warmup-grpo_debug-bz_128-clip_ratio_0.28_one_epoch_tool_star_new_no_warm_up"
data_dir = f"/share/home/sxjiang/myproject/Tool-Star-OCT/Tool_Star_RL/verl_checkpoints/{expertment_name}/rollout"
specific_rollout_iter_num = -1

def get_question(sentence):
    sentence = sentence.split("<|im_end|>\n<|im_start|>assistant\n")[0]
    sentence = sentence.split("<|im_start|>user\n")[1]
    return sentence
def find_wrong(reson_str,sequences_str):
    
    if "bad format" not in reson_str:
        return False
    # # if "<think> </think> not paired" in reson_str:
    # #     return False
    # if sequences_str.count("</answer>")>3 or sequences_str.count("<answer>")>3:
    #     return True
    if "<<" in sequences_str:
        return True
    if "< <" in sequences_str:
        return True
    # if sequences_str.count("</answer>")==2:
    #     return False
    # if sequences_str.count("</answer>")==2 and sequences_str.count("<|im_end|>")==3:
    #     return False
    # if sequences_str.count("</answer>")!=sequences_str.count("<|im_end|>"):
    #     return True
    # if not sequences_str.strip().endswith("<|im_end|>") and not sequences_str.strip().endswith("</python>") and not sequences_str.strip().endswith("</search>"):
    #     print(sequences_str[-10:])
    #     return True
    # if sequences_str.strip().endswith("</python><|im_end|>") or sequences_str.strip().endswith("</search><|im_end|>"):
    #     return True
    return False
def draw_with_max(x,y,result_dir,name):
    plt.figure(figsize=(12, 5))  # 宽度=12，高度=5，单位是英寸
    plt.plot(x, y, marker='o', label='Line')  # 画折线图并加点
    max_x = x[y.index(max(y))]
    max_y = max(y)
    print(max_x,max_y)
    # 添加一条竖线
    plt.axvline(x=max_x, color='red', linestyle='--', label='Max Value')

    # 添加文字标注
    plt.text(max_x, max_y + 1, f'Max: {max_x}', ha='center', color='red', fontsize=10)

    plt.savefig(os.path.join(result_dir,f"{name}.png"))
    plt.close()


rollout_step2metrics={}
wrong_rollout_idx = []
# print(data_dir)
wrong_rollout_num_dict={}
for rollout_file in os.listdir(data_dir):
    rollout_step = rollout_file.split("_")[1].split(".")[0]
    score_sum = 0
    max_calling_times = 0
    all_calling_times = 0
    all_observation_length = 0
    max_length = 0
    count = 0
    questions = []
    wrong_rollout_num=0
    wrong_rollout_sequences=[]
    duplicate_rollout_num=0
    multi_tool_calling_num=0
    # print(rollout_file)
    if str(specific_rollout_iter_num) not in rollout_file and specific_rollout_iter_num != -1:
        continue
    with jsonlines.open(os.path.join(data_dir, rollout_file)) as reader:
        
        
        
        for line in reader:
            count += 1
            score_sum += line["score"]
            all_calling_times += line["is_search"]+line["is_python"]
            max_calling_times = max(max_calling_times, line["is_search"]+line["is_python"])
            max_length = max(max_length, len(line["sequences_str"]))
            questions.append(get_question(line["sequences_str"]))
            pattern = re.compile(r"<result>(.+?)</result>")
            match = pattern.findall(line["sequences_str"])
            if line["is_python"] and line["is_search"]:
                multi_tool_calling_num+=1
            if match:
                all_observation_length += sum([len(i) for i in match])
            if find_wrong(line["reason"],line["sequences_str"]):
                wrong_rollout_num+=1
                # wrong_rollout_sequences.append(line["sequences_str"].split("<|im_start|>assistant\n")[1])     
                wrong_rollout_sequences.append(line)        
            if count==1024:
                break
    rollout_step2metrics[int(rollout_step)] = {
        "score": score_sum / count, 
        "max_calling_times": max_calling_times, 
        "all_calling_times": all_calling_times,
        "max_length": max_length, 
        "all_observation_length": all_observation_length,
        # "question": questions,
        "wrong_rollout_num": wrong_rollout_num,
        "wrong_rollout_sequences": wrong_rollout_sequences,
        "multi_tool_calling_num": multi_tool_calling_num,
        }
print(expertment_name)
result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"analyse",expertment_name)
os.makedirs(result_dir, exist_ok=True)
import matplotlib.pyplot as plt
x = [int(i) for i in rollout_step2metrics.keys()]
x = sorted(x)

y = [rollout_step2metrics[i]["wrong_rollout_num"] for i in x]
draw_with_max(x,y,result_dir,"wrong_rollout_num")

y = [rollout_step2metrics[i]["max_length"] for i in x]
draw_with_max(x,y,result_dir,"max_length")

y = [rollout_step2metrics[i]["max_calling_times"] for i in x]
draw_with_max(x,y,result_dir,"max_calling_times")

y = [rollout_step2metrics[i]["all_calling_times"] for i in x]
draw_with_max(x,y,result_dir,"all_calling_times")

y = [rollout_step2metrics[i]["all_observation_length"] for i in x]
draw_with_max(x,y,result_dir,"all_observation_length")

y = [rollout_step2metrics[i]["multi_tool_calling_num"] for i in x]
draw_with_max(x,y,result_dir,"multi_tool_calling_num")

# print(sorted(rollout_step2metrics.items(), key=lambda x: x[0]))
with open(os.path.join(result_dir,f"rollout_step2metrics.json"), "w") as f:
    json.dump(sorted(rollout_step2metrics.items(), key=lambda x: x[0]), f,indent=4)
if specific_rollout_iter_num != -1:
    with open(os.path.join(result_dir,f"rollout_step2metrics_specific_iter_{specific_rollout_iter_num}.json"), "w") as f:
        json.dump(rollout_step2metrics[specific_rollout_iter_num], f,indent=4) 

#### 分析search
# search_questions = get_search_questions(data_dir,result_dir)
# print(len(search_questions))
# with open(os.path.join(result_dir,"search_questions.json"), "w",encoding="utf-8") as f:
#     json.dump(list(search_questions), f, ensure_ascii=False, indent=4)
#     print(f'save to {os.path.join(result_dir,"search_questions.json")}')
# all_search_questions = set()
# rollout_dir = "/share/home/jfliang/Project/sxjiang/Tool-Star-OCT/Tool_Star_RL/verl_checkpoints"
# result_dir = os.path.dirname(os.path.abspath(__file__))
# all_search_questions = get_search_questions(rollout_dir,result_dir,except_experiment=expertment_name,resume=True)
# print("old search questions num:",len(all_search_questions))
# print("resume search questions num:",len(all_search_questions.intersection(search_questions)))