# 统计python、 search、 reason 的token数

import jsonlines
import os
import json
import re
import textwrap
import matplotlib.pyplot as plt

dataset_names=["musique", "bamboogle", "hotpotqa", "2wiki", "nq", "SimpleQA", "amc23", "aime24", "aime25", "math", "math500", "gsm8k", "OlymBench-math"]
mode_name="budget_no_limit_run"
model_names=["tool_star_qwen_7b_origin", "tool_star_qwen_7b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_new_global_step_78"]
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

def clean_model_name(model_name):
    """清理模型名称，保留主要部分"""
    # 针对tool_star系列
    if model_name.startswith('tool_star_qwen_3b_origin'):
        return 'tool_star_qwen_3b_origin'
    elif model_name.startswith('tool_star_qwen_3b_oct'):
        return 'tool_star_qwen_3b_oct'
    elif model_name.startswith('tool_star_qwen_7b_origin'):
        return 'tool_star_qwen_7b_origin'
    elif model_name.startswith('tool_star_qwen_7b_oct'):
        return 'tool_star_qwen_7b_oct'
    # 针对ARPO系列
    elif model_name.startswith('ARPO_3b_oct'):
        return 'ARPO_3b_oct'
    elif model_name.startswith('ARPO_3b'):
        return 'ARPO_3b'
    elif model_name.startswith('ARPO_7b_oct'):
        return 'ARPO_7b_oct'
    elif model_name.startswith('ARPO_7b'):
        return 'ARPO_7b'
    else:
        # 其他情况，移除常见的后缀
        cleaned = model_name
        suffixes_to_remove = [
            '_global_step_110', '_global_step_78', '_gpu18',
            '_clip_radio_gradclip_02_two_epoch_down_progressive_seq_mean_new_add'
        ]
        for suffix in suffixes_to_remove:
            if suffix in cleaned:
                cleaned = cleaned.replace(suffix, '')
        return cleaned

def save_detailed_results(modelname2dict):
    """保存详细的JSON结果文件，类似於analyse_rollout_Toolacc.py的功能"""
    
    # 创建基础输出目录
    base_output_dir = "/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/img/token_analysis"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # 如果有多个模型，为每个模型创建单独的结果文件
    for model_name, model_data in modelname2dict.items():
        cleaned_model_name = clean_model_name(model_name)
        
        # 为单个模型创建目录
        model_output_dir = os.path.join(base_output_dir, cleaned_model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # 计算该模型的总体统计信息
        all_metrics = [
            "avg_of_total_reason_token",
            "avg_of_total_search_token", 
            "avg_of_total_python_token",
            "avg_of_mean_sequence_reason_token",
            "avg_of_mean_sequence_search_token",
            "avg_of_mean_sequence_python_token",
            "avg_of_reason_token_per_call",
            "avg_of_search_token_per_call",
            "avg_of_python_token_per_call"
        ]
        
        overall_metrics = {}
        for metric in all_metrics:
            values = []
            for dataset_name, dataset_data in model_data.items():
                if metric in dataset_data:
                    values.append(dataset_data[metric])
            
            if values:
                overall_metrics[f"{metric}_mean"] = sum(values) / len(values)
                overall_metrics[f"{metric}_max"] = max(values)
                overall_metrics[f"{metric}_min"] = min(values)
                overall_metrics[f"{metric}_std"] = (sum((x - overall_metrics[f"{metric}_mean"])**2 for x in values) / len(values))**0.5
            else:
                overall_metrics[f"{metric}_mean"] = 0
                overall_metrics[f"{metric}_max"] = 0
                overall_metrics[f"{metric}_min"] = 0
                overall_metrics[f"{metric}_std"] = 0
        
        # 构建完整的结果字典
        result_data = {
            "model_info": {
                "original_model_name": model_name,
                "cleaned_model_name": cleaned_model_name,
                "analysis_mode": mode_name,
                "total_datasets": len(model_data)
            },
            "overall_metrics": overall_metrics,
            "per_dataset_results": []
        }
        
        # 添加每个数据集的详细结果
        for dataset_name, dataset_data in model_data.items():
            dataset_result = {
                "dataset_name": dataset_name,
                "metrics": dataset_data
            }
            result_data["per_dataset_results"].append(dataset_result)
        
        # 保存单个模型的详细结果
        output_path = os.path.join(model_output_dir, "detailed_token_analysis.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=4, ensure_ascii=False)
        
        print(f"\n模型 {cleaned_model_name} 的详细结果已保存到: {output_path}")
        
        # 打印该模型的总体统计摘要
        print(f"\n=== {cleaned_model_name} 总体统计摘要 ===")
        print(f"分析的数据集数量: {len(model_data)}")
        
        key_metrics = [
            "avg_of_total_reason_token_mean",
            "avg_of_total_search_token_mean", 
            "avg_of_total_python_token_mean"
        ]
        
        for metric in key_metrics:
            if metric in overall_metrics:
                metric_display_name = metric.replace("avg_of_total_", "").replace("_token_mean", "").upper()
                print(f"{metric_display_name} 平均Token数: {overall_metrics[metric]:.2f}")
    
    # 如果有多个模型，创建模型对比的汇总文件
    if len(modelname2dict) > 1:
        comparison_data = {
            "analysis_info": {
                "mode": mode_name,
                "datasets_analyzed": dataset_names,
                "total_models": len(modelname2dict)
            },
            "model_comparison": {}
        }
        
        for model_name, model_data in modelname2dict.items():
            cleaned_name = clean_model_name(model_name)
            
            # 计算每个模型的关键指标平均值
            key_metrics_summary = {}
            for metric in all_metrics:
                values = []
                for dataset_data in model_data.values():
                    if metric in dataset_data:
                        values.append(dataset_data[metric])
                
                if values:
                    key_metrics_summary[metric] = {
                        "mean": sum(values) / len(values),
                        "datasets_count": len(values)
                    }
                else:
                    key_metrics_summary[metric] = {
                        "mean": 0,
                        "datasets_count": 0
                    }
            
            comparison_data["model_comparison"][cleaned_name] = {
                "original_name": model_name,
                "metrics_summary": key_metrics_summary,
                "datasets_analyzed": list(model_data.keys())
            }
        
        # 保存模型对比文件
        comparison_output_path = os.path.join(base_output_dir, "models_comparison.json")
        with open(comparison_output_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=4, ensure_ascii=False)
        
        print(f"\n模型对比结果已保存到: {comparison_output_path}")
        
        # 打印模型对比摘要
        print(f"\n=== 模型对比摘要 ===")
        for model_name in comparison_data["model_comparison"]:
            model_info = comparison_data["model_comparison"][model_name]
            print(f"\n{model_name}:")
            
            # 显示关键指标
            key_display_metrics = [
                "avg_of_total_reason_token",
                "avg_of_total_search_token",
                "avg_of_total_python_token"
            ]
            
            for metric in key_display_metrics:
                if metric in model_info["metrics_summary"]:
                    mean_val = model_info["metrics_summary"][metric]["mean"]
                    count = model_info["metrics_summary"][metric]["datasets_count"]
                    metric_display = metric.replace("avg_of_total_", "").replace("_token", "").upper()
                    print(f"  {metric_display}: {mean_val:.2f} (基于{count}个数据集)")

rollout_model2metrics={}
wrong_rollout_idx = []
# print(data_dir)
wrong_rollout_num_dict={}
print(data_dir)
modelname2dict = {}
for root,dirs,files in os.walk(data_dir):
    if "one_epoch_warmup" in root:
        continue
    if dataset_names and not any(dataset_name in root for dataset_name in dataset_names):
        continue
    if mode_name!="" and mode_name not in root:
        continue
    
    # 修改模型名称匹配逻辑，确保精确匹配
    if model_names:
        path_model_name = root.split("/")[-1]
        model_matched = False
        for model_name in model_names:
            # 检查是否精确匹配模型名称（允许后缀，但要确保是完整的模型名称匹配）
            if path_model_name == model_name:
                model_matched = True
                break
        if not model_matched:
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
        # calling_times = []
        # acc_calling_times = []
        # acc_calling_times_one = [] #调用次数为1且正确
        # achieve_max_budget_num = [] #调用次数为1且正确
        
        total_search_token_len = [] # 计算所有<search>的总token长度
        total_python_token_len = [] # 计算所有<python>的总token长度
        total_reason_token_len = [] # 计算所有<think>的总token长度
        
        per_python_token_len = []  # 计算每段<python>的平均token长度
        per_search_token_len = []  # 计算每段<search>的平均token长度
        per_reason_token_len = []   # 计算每段<think>的平均token长度
        
        per_call_python_token_len = []  # 计算每次调用的<python>的平均token长度， 如果调用为0，和call_times=1一样
        per_call_search_token_len = []  # 计算每次调用的<search>的平均token长度， 如果调用为0，和call_times=1一样
        per_call_reason_token_len = []  # 计算每次调用的<think>的平均token长度， 如果调用为0，和call_times=1一样


        if file=="result.metrics.json":
            file_path=os.path.join(root,file)

            with open(file_path,"r") as f:
                data=json.load(f)
                for item in data: 
                    if item['Metrics']['llm_response'] == 'Correct' or 'Incorrect' in item['Metrics']['llm_response']:
                        total_think_sequence_len = 0
                        total_search_sequence_len = 0
                        total_python_sequence_len = 0
                        
                        # 提取item['Full_output'] 中， <think> 和 </think> 之间的token数量， 注意Full_output中可能会有多组<think> </think> ， 也就是直到遇到<answer>之前都会有<think>
                        think_tokens_sequence = re.findall(r'<think>(.*?)</think>', item['Full_output'], re.DOTALL)
                        search_tokens_sequence = re.findall(r'<search>(.*?)</search>', item['Full_output'], re.DOTALL)
                        python_tokens_sequence = re.findall(r'<python>(.*?)</python>', item['Full_output'], re.DOTALL)
                        call_times = item['calling_rounds']
                        
                        for think_tokens in think_tokens_sequence:
                            tokens = think_tokens.split()
                            total_think_sequence_len += len(tokens)

                        for search_tokens in search_tokens_sequence:
                            tokens = search_tokens.split()
                            total_search_sequence_len += len(tokens)
                            
                        for python_tokens in python_tokens_sequence:
                            tokens = python_tokens.split()
                            total_python_sequence_len += len(tokens)

                        # 计算所有<think>的总token长度
                        total_reason_token_len.append(total_think_sequence_len)
                        # 计算所有<search>的总token长度
                        total_search_token_len.append(total_search_sequence_len)
                        # 计算所有<python>的总token长度
                        total_python_token_len.append(total_python_sequence_len)

                        # 计算每段<think>的平均token长度
                        avg_of_think_sequence_len = total_think_sequence_len / len(think_tokens_sequence) if think_tokens_sequence else 0
                        per_reason_token_len.append(avg_of_think_sequence_len)
                        
                        # 计算每段<search>的平均token长度
                        avg_of_search_sequence_len = total_search_sequence_len / len(search_tokens_sequence) if search_tokens_sequence else 0
                        per_search_token_len.append(avg_of_search_sequence_len)
                        
                        # 计算每段<python>的平均token长度
                        avg_of_python_sequence_len = total_python_sequence_len / len(python_tokens_sequence) if python_tokens_sequence else 0
                        per_python_token_len.append(avg_of_python_sequence_len)
                        
                        # 计算每次调用的<think>的平均token长度， 如果调用为0，和call_times=1一样
                        if call_times > 0:
                            per_call_reason_token_len.append(total_think_sequence_len / call_times)
                        elif call_times == 0:
                            per_call_reason_token_len.append(total_think_sequence_len)
                            
                        # 计算每次调用的<search>的平均token长度， 如果调用为0，和call_times=1一样
                        if call_times > 0:
                            per_call_search_token_len.append(total_search_sequence_len / call_times)
                        elif call_times == 0:
                            per_call_search_token_len.append(total_search_sequence_len)
                            
                        # 计算每次调用的<python>的平均token长度， 如果调用为0，和call_times=1一样
                        if call_times > 0:
                            per_call_python_token_len.append(total_python_sequence_len / call_times)
                        elif call_times == 0:
                            per_call_python_token_len.append(total_python_sequence_len)
                            

                avg_of_total_reason_token = sum(total_reason_token_len) / len(total_reason_token_len) if total_reason_token_len else 0
                avg_of_total_search_token = sum(total_search_token_len) / len(total_search_token_len) if total_search_token_len else 0
                avg_of_total_python_token = sum(total_python_token_len) / len(total_python_token_len) if total_python_token_len else 0

                avg_of_mean_sequence_reason_token = sum(per_reason_token_len) / len(per_reason_token_len) if per_reason_token_len else 0
                avg_of_mean_sequence_search_token = sum(per_search_token_len) / len(per_search_token_len) if per_search_token_len else 0
                avg_of_mean_sequence_python_token = sum(per_python_token_len) / len(per_python_token_len) if per_python_token_len else 0
                
                avg_of_reason_token_per_call = sum(per_call_reason_token_len) / len(per_call_reason_token_len) if per_call_reason_token_len else 0
                avg_of_search_token_per_call = sum(per_call_search_token_len) / len(per_call_search_token_len) if per_call_search_token_len else 0
                avg_of_python_token_per_call = sum(per_call_python_token_len) / len(per_call_python_token_len) if per_call_python_token_len else 0

            # 实际这里不是真正的avg_token， 这里的token是按照空格划分的
            # 计算以下指标：
            # 1. avg_of_total_reason_token ，每段think的token之和
            # 2. avg_of_mean_sequence_reason_token， 每段think的平均token数
            # 3. avg_of_reason_token_per_call, 每段think的token之和除以tool调用次数 
            
            # 4. avg_of_total_search_token, 每段search的token之和
            # 5. avg_of_total_python_token, 每段python的token之和
            modelname2dict[path_model_name][path_dataset_name]["avg_of_total_reason_token"] = avg_of_total_reason_token
            modelname2dict[path_model_name][path_dataset_name]["avg_of_total_search_token"] = avg_of_total_search_token
            modelname2dict[path_model_name][path_dataset_name]["avg_of_total_python_token"] = avg_of_total_python_token
            
            modelname2dict[path_model_name][path_dataset_name]["avg_of_mean_sequence_reason_token"] = avg_of_mean_sequence_reason_token
            modelname2dict[path_model_name][path_dataset_name]["avg_of_mean_sequence_search_token"] = avg_of_mean_sequence_search_token
            modelname2dict[path_model_name][path_dataset_name]["avg_of_mean_sequence_python_token"] = avg_of_mean_sequence_python_token
            
            modelname2dict[path_model_name][path_dataset_name]["avg_of_reason_token_per_call"] = avg_of_reason_token_per_call
            modelname2dict[path_model_name][path_dataset_name]["avg_of_search_token_per_call"] = avg_of_search_token_per_call
            modelname2dict[path_model_name][path_dataset_name]["avg_of_python_token_per_call"] = avg_of_python_token_per_call

# 在原有的绘图代码之前，先保存JSON结果
save_detailed_results(modelname2dict)

# 统计每个模型在各个测试集下的avg_of_reason_token并绘图
# for path_model_name in modelname2dict:
#     dataset_names_list = []
#     avg_reason_tokens_list = []
    
#     # 收集该模型在各个数据集下的avg_of_reason_token
#     for path_dataset_name in modelname2dict[path_model_name]:
#         if "avg_of_reason_token" in modelname2dict[path_model_name][path_dataset_name]:
#             dataset_names_list.append(path_dataset_name)
#             avg_reason_tokens_list.append(modelname2dict[path_model_name][path_dataset_name]["avg_of_reason_token"])
    
#     if dataset_names_list and avg_reason_tokens_list:
#         # 按数据集名称排序
#         sorted_data = sorted(zip(dataset_names_list, avg_reason_tokens_list))
#         dataset_names_list, avg_reason_tokens_list = zip(*sorted_data)
        
#         # 绘制柱状图
#         plt.figure(figsize=(15, 8))
#         wrapped_labels = ['\n'.join(textwrap.wrap(name, 10)) for name in dataset_names_list]
#         bars = plt.bar(wrapped_labels, avg_reason_tokens_list, color='skyblue', alpha=0.7)
        
#         # 在柱子上显示数值
#         for bar, value in zip(bars, avg_reason_tokens_list):
#             plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_reason_tokens_list)*0.01, 
#                     f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
#         plt.title(f'Average Reasoning Token Length by Dataset\n({path_model_name})', fontsize=14, pad=20)
#         plt.xlabel('Dataset', fontsize=12)
#         plt.ylabel('Average Reasoning Token Length', fontsize=12)
#         plt.xticks(rotation=45, ha='right')
#         plt.grid(axis='y', alpha=0.3)
#         plt.tight_layout()
        
#         # 保存柱状图
#         result_path = "/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/img"
#         plt.savefig(os.path.join(result_path, f"{path_model_name}_avg_reason_token_bar.png"), dpi=300, bbox_inches='tight')
#         plt.close()
        
#         # 绘制折线图
#         plt.figure(figsize=(15, 8))
#         plt.plot(range(len(dataset_names_list)), avg_reason_tokens_list, marker='o', linewidth=2, markersize=8, color='red')
        
#         # 在点上显示数值
#         for i, (name, value) in enumerate(zip(dataset_names_list, avg_reason_tokens_list)):
#             plt.text(i, value + max(avg_reason_tokens_list)*0.02, f'{value:.1f}', 
#                     ha='center', va='bottom', fontsize=9)
        
#         # 标记最大值和最小值
#         max_index = avg_reason_tokens_list.index(max(avg_reason_tokens_list))
#         min_index = avg_reason_tokens_list.index(min(avg_reason_tokens_list))
        
#         plt.axvline(x=max_index, color='green', linestyle='--', alpha=0.7, label=f'Max: {dataset_names_list[max_index]}')
#         plt.axvline(x=min_index, color='orange', linestyle='--', alpha=0.7, label=f'Min: {dataset_names_list[min_index]}')
        
#         plt.title(f'Average Reasoning Token Length Trend by Dataset\n({path_model_name})', fontsize=14, pad=20)
#         plt.xlabel('Dataset', fontsize=12)
#         plt.ylabel('Average Reasoning Token Length', fontsize=12)
#         plt.xticks(range(len(dataset_names_list)), wrapped_labels, rotation=45, ha='right')
#         plt.grid(True, alpha=0.3)
#         plt.legend()
#         plt.tight_layout()
        
#         # 保存折线图
#         plt.savefig(os.path.join(result_path, f"{path_model_name}_avg_reason_token_line.png"), dpi=300, bbox_inches='tight')
#         plt.close()
        
#         # 打印统计信息
#         print(f"\n=== {path_model_name} ===")
#         print(f"Dataset count: {len(dataset_names_list)}")
#         print(f"Max avg_reason_token: {max(avg_reason_tokens_list):.2f} ({dataset_names_list[max_index]})")
#         print(f"Min avg_reason_token: {min(avg_reason_tokens_list):.2f} ({dataset_names_list[min_index]})")
#         print(f"Overall avg_reason_token: {sum(avg_reason_tokens_list)/len(avg_reason_tokens_list):.2f}")

# # 绘制模型对比图
# if len(modelname2dict) >= 2:
#     # 获取所有模型的数据
#     model_data = {}
#     all_datasets = set()
    
#     for path_model_name in modelname2dict:
#         model_data[path_model_name] = {}
#         for path_dataset_name in modelname2dict[path_model_name]:
#             if "avg_of_reason_token" in modelname2dict[path_model_name][path_dataset_name]:
#                 model_data[path_model_name][path_dataset_name] = modelname2dict[path_model_name][path_dataset_name]["avg_of_reason_token"]
#                 all_datasets.add(path_dataset_name)
    
#     # 找到两个模型都有数据的数据集
#     common_datasets = []
#     for dataset in sorted(all_datasets):
#         if all(dataset in model_data[model] for model in model_data):
#             common_datasets.append(dataset)
    
#     if common_datasets and len(model_data) >= 2:
#         # 绘制对比柱状图
#         plt.figure(figsize=(16, 10))
        
#         x = range(len(common_datasets))
#         width = 0.35
        
#         models = list(model_data.keys())
#         colors = ['skyblue', 'lightcoral', 'lightgreen', 'wheat', 'plum']  # 支持最多5个模型
        
#         for i, model in enumerate(models[:5]):  # 最多显示5个模型
#             values = [model_data[model][dataset] for dataset in common_datasets]
#             offset = (i - len(models)/2 + 0.5) * width
#             bars = plt.bar([pos + offset for pos in x], values, width, 
#                           label=model.replace('tool_star_qwen_3b_', '').replace('_global_step_110', '').replace('_global_step_78', ''), 
#                           color=colors[i], alpha=0.8)
            
#             # 在柱子上显示数值
#             for bar, value in zip(bars, values):
#                 plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01, 
#                         f'{value:.1f}', ha='center', va='bottom', fontsize=8)
        
#         wrapped_labels = ['\n'.join(textwrap.wrap(name, 8)) for name in common_datasets]
#         plt.xlabel('Dataset', fontsize=12)
#         plt.ylabel('Average Reasoning Token Length', fontsize=12)
#         plt.title('Model Comparison: Average Reasoning Token Length by Dataset', fontsize=14, pad=20)
#         plt.xticks(x, wrapped_labels, rotation=45, ha='right')
#         plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#         plt.grid(axis='y', alpha=0.3)
#         plt.tight_layout()
        
#         # 保存对比柱状图
#         result_path = "/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/img"
#         plt.savefig(os.path.join(result_path, "model_comparison_avg_reason_token_bar.png"), dpi=300, bbox_inches='tight')
#         plt.close()
        
#         # 绘制对比折线图
#         plt.figure(figsize=(16, 10))
#         plt.plot(range(len(common_datasets)), avg_reason_tokens_list, marker='o', linewidth=2, markersize=8, color='red')
        
#         # 在点上显示数值
#         for i, (name, value) in enumerate(zip(common_datasets, avg_reason_tokens_list)):
#             plt.text(i, value + max(avg_reason_tokens_list)*0.02, f'{value:.1f}', 
#                     ha='center', va='bottom', fontsize=8)
        
#         # 标记最大值和最小值
#         max_index = avg_reason_tokens_list.index(max(avg_reason_tokens_list))
#         min_index = avg_reason_tokens_list.index(min(avg_reason_tokens_list))
        
#         plt.axvline(x=max_index, color='green', linestyle='--', alpha=0.7, label=f'Max: {common_datasets[max_index]}')
#         plt.axvline(x=min_index, color='orange', linestyle='--', alpha=0.7, label=f'Min: {common_datasets[min_index]}')
        
#         plt.title('Average Reasoning Token Length Trend by Dataset', fontsize=14, pad=20)
#         plt.xlabel('Dataset', fontsize=12)
#         plt.ylabel('Average Reasoning Token Length', fontsize=12)
#         plt.xticks(range(len(common_datasets)), wrapped_labels, rotation=45, ha='right')
#         plt.grid(True, alpha=0.3)
#         plt.legend()
#         plt.tight_layout()
        
#         # 保存对比折线图
#         plt.savefig(os.path.join(result_path, "model_comparison_avg_reason_token_line.png"), dpi=300, bbox_inches='tight')
#         plt.close()
        
#         # 打印对比统计信息
#         print(f"\n=== Model Comparison ===")
#         print(f"Common datasets: {len(common_datasets)}")
#         print(f"Datasets: {', '.join(common_datasets)}")
        
#         for model in models:
#             values = [model_data[model][dataset] for dataset in common_datasets]
#             print(f"\n{model}:")
#             print(f"  Average: {sum(values)/len(values):.2f}")
#             print(f"  Max: {max(values):.2f}")
#             print(f"  Min: {min(values):.2f}")


# 绘制所有指标的模型对比图
if len(modelname2dict) >= 2:
    # 获取所有模型的数据
    model_data = {}
    all_datasets = set()
    
    # 定义所有需要对比的指标
    metrics = [
        "avg_of_total_reason_token",
        "avg_of_total_search_token",
        "avg_of_total_python_token",
        "avg_of_mean_sequence_reason_token",
        "avg_of_mean_sequence_search_token", 
        "avg_of_mean_sequence_python_token",
        "avg_of_reason_token_per_call",
        "avg_of_search_token_per_call",
        "avg_of_python_token_per_call"
    ]
    
    # 指标的中文名称映射
    metric_titles = {
        "avg_of_total_reason_token": "Average Total Reasoning Token Length",
        "avg_of_total_search_token": "Average Total Search Token Length",
        "avg_of_total_python_token": "Average Total Python Token Length",
        "avg_of_mean_sequence_reason_token": "Average Mean Sequence Reasoning Token Length",
        "avg_of_mean_sequence_search_token": "Average Mean Sequence Search Token Length",
        "avg_of_mean_sequence_python_token": "Average Mean Sequence Python Token Length",
        "avg_of_reason_token_per_call": "Average Reasoning Token Per Call",
        "avg_of_search_token_per_call": "Average Search Token Per Call",
        "avg_of_python_token_per_call": "Average Python Token Per Call"
    }
    
    for path_model_name in modelname2dict:
        model_data[path_model_name] = {}
        for path_dataset_name in modelname2dict[path_model_name]:
            for metric in metrics:
                if metric in modelname2dict[path_model_name][path_dataset_name]:
                    if metric not in model_data[path_model_name]:
                        model_data[path_model_name][metric] = {}
                    model_data[path_model_name][metric][path_dataset_name] = modelname2dict[path_model_name][path_dataset_name][metric]
                    all_datasets.add(path_dataset_name)
    
    # 找到两个模型都有数据的数据集
    common_datasets = []
    for dataset in sorted(all_datasets):
        has_all_models = True
        for model in model_data:
            # 检查该模型是否在所有指标上都有这个数据集的数据
            for metric in metrics:
                if metric not in model_data[model] or dataset not in model_data[model][metric]:
                    has_all_models = False
                    break
            if not has_all_models:
                break
        if has_all_models:
            common_datasets.append(dataset)
    
    if common_datasets and len(model_data) >= 2:
        models = list(model_data.keys())
        
        # 创建路径结构 - 使用清理后的模型名称
        cleaned_model_names = [clean_model_name(model) for model in models]
        model_names_str = "_vs_".join(cleaned_model_names)
        base_result_path = "/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/img"
        model_comparison_path = os.path.join(base_result_path, model_names_str)
        
        # 为每个指标绘制对比图
        for metric in metrics:
            # 创建指标路径
            metric_path = os.path.join(model_comparison_path, metric)
            os.makedirs(metric_path, exist_ok=True)
            
            # 绘制对比柱状图
            plt.figure(figsize=(16, 10))
            
            x = range(len(common_datasets))
            width = 0.35
            colors = ['skyblue', 'lightcoral', 'lightgreen', 'wheat', 'plum']  # 支持最多5个模型
            
            for i, model in enumerate(models[:5]):  # 最多显示5个模型
                values = [model_data[model][metric][dataset] for dataset in common_datasets]
                offset = (i - len(models)/2 + 0.5) * width
                bars = plt.bar([pos + offset for pos in x], values, width, 
                              label=clean_model_name(model), 
                              color=colors[i], alpha=0.8)
                
                # 在柱子上显示数值
                for bar, value in zip(bars, values):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01, 
                            f'{value:.1f}', ha='center', va='bottom', fontsize=8)
            
            wrapped_labels = ['\n'.join(textwrap.wrap(name, 8)) for name in common_datasets]
            plt.xlabel('Dataset', fontsize=12)
            plt.ylabel('Token Length', fontsize=12)
            plt.title(f'Model Comparison: {metric_titles[metric]} by Dataset', fontsize=14, pad=20)
            plt.xticks(x, wrapped_labels, rotation=45, ha='right')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            # 保存对比柱状图
            plt.savefig(os.path.join(metric_path, f"{metric}_comparison_bar.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 绘制对比折线图
            plt.figure(figsize=(16, 10))
            
            markers = ['o', 's', '^', 'D', 'v']  # 不同的标记样式
            line_styles = ['-', '--', '-.', ':', '-']  # 不同的线型
            
            for i, model in enumerate(models[:5]):
                values = [model_data[model][metric][dataset] for dataset in common_datasets]
                label = clean_model_name(model)
                plt.plot(x, values, marker=markers[i], linewidth=2, markersize=8, 
                        label=label, linestyle=line_styles[i], color=colors[i])
                
                # 在数据点上显示数值
                for j, value in enumerate(values):
                    plt.text(j, value + max(values)*0.02, f'{value:.1f}', 
                            ha='center', va='bottom', fontsize=8)
            
            plt.xlabel('Dataset', fontsize=12)
            plt.ylabel('Token Length', fontsize=12)
            plt.title(f'Model Comparison: {metric_titles[metric]} Trend by Dataset', fontsize=14, pad=20)
            plt.xticks(x, wrapped_labels, rotation=45, ha='right')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 保存对比折线图
            plt.savefig(os.path.join(metric_path, f"{metric}_comparison_line.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 打印该指标的对比统计信息
            print(f"\n=== {metric_titles[metric]} ===")
            print(f"Common datasets: {len(common_datasets)}")
            print(f"Datasets: {', '.join(common_datasets)}")
            
            for model in models:
                values = [model_data[model][metric][dataset] for dataset in common_datasets]
                print(f"\n{clean_model_name(model)}:")
                print(f"  Average: {sum(values)/len(values):.2f}")
                print(f"  Max: {max(values):.2f}")
                print(f"  Min: {min(values):.2f}")

