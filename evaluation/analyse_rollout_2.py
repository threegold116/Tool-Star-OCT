import jsonlines
import os
import json
import re
from tqdm import tqdm
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52")

# token_ids = tokenizer.encode(text,return_tensors="pt")
# exit()
dataset_name="hle"
mode_name="budget_no_limit_run"
data_dir = f"//home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/result/"
specific_rollout_iter_num = -1
def draw_with_max(x,y,result_dir,name):
    print(x,y)
   
    plt.figure(figsize=(16, 10))
    plt.plot(x, y, marker='o', label='Line')
    plt.xticks(rotation=45, ha='right')
    
    # 标记最大值
    max_index = y.index(max(y))
    max_x = x[max_index]
    max_y = y[max_index]
    plt.axvline(x=max_x, color='red', linestyle='--', label='Max Value')
    plt.text(max_x, max_y + 0.05, f'Max: {max_y}', ha='center', color='red', fontsize=10, rotation=90)
    plt.legend()

    # 保存图片
    plt.savefig(os.path.join(result_dir,f"{name}.png"))
    plt.close()

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

rollout_model2metrics={}
wrong_rollout_idx = []
# print(data_dir)
wrong_rollout_num_dict={}
print(data_dir)
for root,dirs,files in os.walk(data_dir):
    for file in files:
        if file=="result.json":
            file_path=os.path.join(root,file)
            multi_tool_calling_num = 0
            length_limit_num = 0
            with open(file_path,"r") as f:
                data=json.load(f)
                model_name = root.split("/")[-1]
                for item in data:
                    if "Python interpreter call costs" in item["Prompt"]:
                        python_cost = int(item["Prompt"].split("Python interpreter call costs")[1].split(" ")[1])
                        search_cost = int(item["Prompt"].split("search call costs")[1].split(" ")[1].split(".")[0])
                        if python_cost!=item["python_budget"] or search_cost!=item["search_budget"]:
                            print(file_path)
                            print(item["question"])
                            break
                # if length_limit_num>0:
                #     print(file_path)
                #     print(length_limit_num)
# import matplotlib.pyplot as plt
# x = [i for i in rollout_model2metrics.keys()]
# y = [rollout_model2metrics[i]["multi_tool_calling_num"] for i in x]
# x = [i.replace("tool_star_qwen_3b","") for i in rollout_model2metrics.keys()]
# result_dir = f"./analyse/{dataset_name}/{mode_name}"
# os.makedirs(result_dir, exist_ok=True)
# draw_with_max(x,y,result_dir,"multi_tool_calling_num")

# y = [rollout_step2metrics[i]["max_length"] for i in x]
# plt.plot(x, y)
# plt.savefig(f"{result_dir}/max_length.png")
# plt.close()

# print(sorted(rollout_step2metrics.items(), key=lambda x: x[0]))
# with open(f"/share/home/sxjiang/myproject/Tool-Star-OCT/Tool_Star_RL/{expertment_name}_{str(specific_rollout_iter_num)}_rollout_step2metrics.json", "w") as f:
#     json.dump(sorted(rollout_step2metrics.items(), key=lambda x: x[0]), f,indent=4)
# with open(f"/share/home/sxjiang/myproject/Tool-Star-OCT/Tool_Star_RL/{expertment_name}_{str(specific_rollout_iter_num)}_rollout_step2metrics.json", "w") as f:
#     json.dump(sorted(rollout_step2metrics.items(), key=lambda x: x[0]), f,indent=4)
