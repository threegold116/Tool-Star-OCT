import jsonlines
import os
import json
import re
import textwrap
import matplotlib.pyplot as plt
dataset_name="2wiki"
mode_name="no-tool-inference_new"
model_name="tool_star_qwen_3b_origin"
data_dir = f"/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/result/"
specific_rollout_iter_num = -1
def draw_column(labels,values,result_dir,name):
    print(labels)
    print(values)
    # 绘制柱形图
    plt.figure(figsize=(8, 8))
    wrapped_labels = ['\n'.join(textwrap.wrap(l, 6)) for l in labels]
    plt.bar(wrapped_labels, values)
    # 用 textwrap 自动按宽度切分
    # 添加标题和标签
    plt.title('Colome')
    plt.xlabel('calling_times')
    plt.ylabel('num')

    # 显示图形
    # 保存图片
    plt.savefig(os.path.join(result_dir,f"{name}.png"))
    plt.close()


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
modelname2dict = {}
for root,dirs,files in os.walk(data_dir):
    if "one_epoch_warmup" in root:
        continue
    if dataset_name!="" and dataset_name not in root:
        continue
    if mode_name!="" and mode_name not in root:
        continue
    if model_name!="" and model_name not in root:
        continue
    for file in files:
        path_model_name = root.split("/")[-1]
        path_dataset_name = root.split("/")[-4] if ("no_limit_run" not in root) and ("no-tool" not in root) else root.split("/")[-3]
        if path_model_name not in modelname2dict:
            modelname2dict[path_model_name] = {}
        if path_model_name in modelname2dict:
            if path_dataset_name not in modelname2dict[path_model_name]:
                modelname2dict[path_model_name][path_dataset_name] = {}
        #需要统计的指标
        calling_times = []
        acc_calling_times = []
        acc_calling_times_one = [] #调用次数为1且正确
        achieve_max_budget_num = [] #调用次数为1且正确
        if file=="result.json":
            file_path=os.path.join(root,file)
            multi_tool_calling_num = 0
            with open(file_path,"r") as f:
                data=json.load(f)
                for item in data:
                    if item["search_rounds"]>0 and item["python_rounds"]>0:
                        multi_tool_calling_num+=1
                    calling_times.append(item["search_rounds"]+item["python_rounds"])
                    if "Reached the maximum budget" in item["Full_output"]:
                        achieve_max_budget_num.append(1)
            modelname2dict[path_model_name][path_dataset_name]["calling_times"] = calling_times
            modelname2dict[path_model_name][path_dataset_name]["achieve_max_budget_num"] = achieve_max_budget_num
        if file=="result.metrics.json":
            file_path=os.path.join(root,file)
            with open(file_path,"r") as f:
                print(file_path)
                data=json.load(f)
                for item in data:
                    if item["Metrics"]["llm_equal"]==1:
                        acc_calling_times.append(item["search_rounds"]+item["python_rounds"])
                        if item["search_rounds"]+item["python_rounds"]==1:
                            acc_calling_times_one.append(item["search_rounds"]+item["python_rounds"])
            modelname2dict[path_model_name][path_dataset_name]["acc_calling_times"] = acc_calling_times
            modelname2dict[path_model_name][path_dataset_name]["acc_calling_times_one"] = acc_calling_times_one
# x = [i for i in rollout_model2metrics.keys()]
# y = [rollout_model2metrics[i]["multi_tool_calling_num"] for i in x]
# x = [i.replace("tool_star_qwen_3b","") for i in rollout_model2metrics.keys()]
# result_dir = f"./analyse/{dataset_name}/{mode_name}"
# os.makedirs(result_dir, exist_ok=True)
# draw_with_max(x,y,result_dir,"multi_tool_calling_num")

#每个模型调用次数的直方图
for path_model_name in modelname2dict:
    for path_dataset_name in modelname2dict[path_model_name]:
        calling_times = modelname2dict[path_model_name][path_dataset_name]["calling_times"]
        calling_times2dict = {}
        for calling_time in calling_times:
            if str(calling_time) not in calling_times2dict:
                calling_times2dict[str(calling_time)] = 0
            calling_times2dict[str(calling_time)] += 1
        calling_times2dict = dict(sorted(calling_times2dict.items(),key=lambda x:int(x[0])))
        result_dir = f"./analyse/{path_dataset_name}/{mode_name}/{path_model_name}"
        os.makedirs(result_dir, exist_ok=True)
        # draw_column(list(calling_times2dict.keys()),list(calling_times2dict.values()),result_dir,"calling_times_column")
compare_model_names = [
    "tool_star_qwen_3b_origin",
    "tool_star_qwen_3b_oct_clip_radio_gradclip_032_one_epoch_no_progressive_seq_mean_new_global_step_78",
    "tool_star_qwen_3b_oct_clip_radio_gradclip_02_one_epoch_no_progressive_seq_mean_new_global_step_78"
]
compare_keys = [
    "acc_calling_times",
    "acc_calling_times_one",
    "achieve_max_budget_num"
]
compare_dataset_names = [
    "bamboogle",
    "musique",
    "gsm8k",
    "aime24",
    "aime25"
]
#不同模型调用的直方图
for compare_key in compare_keys:
    for path_dataset_name in compare_dataset_names:
        compare_key2values = {}
        for compare_model_name in compare_model_names:
            if path_dataset_name not in modelname2dict[compare_model_name]:
                continue
            compare_key2values[compare_model_name] = sum(modelname2dict[compare_model_name][path_dataset_name][compare_key])
        result_dir = f"./a-compare/analyse/{path_dataset_name}/{mode_name}/"
        os.makedirs(result_dir, exist_ok=True)
        draw_column([x.split("radio_gradclip_")[-1] for x in list(compare_key2values.keys())],list(compare_key2values.values()),result_dir,f"compare_{compare_key}_sum")
