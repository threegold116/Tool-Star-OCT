# 统计python、 search、 reason 的token数

import jsonlines
import os
import json
import re
import textwrap
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from transformers import AutoTokenizer

# 添加字体文件
fm.fontManager.addfont('/home/sxjiang/myproject/analyse/radar/TIMES.TTF')
# 字体设置 
fontsize = 25

# 设置matplotlib全局字体参数
plt.rcParams.update({
    "legend.fontsize": fontsize,
    "legend.title_fontsize": fontsize,    # 如果有图例标题
    "font.size": 15,        # 全局字体大小
    "font.family": "Times New Roman",  # 全局字体族
    "axes.titlesize": fontsize,   # 标题字体
    "axes.labelsize": fontsize,   # 坐标轴标签字体
    "xtick.labelsize": fontsize,  # X轴刻度字体
    "ytick.labelsize": fontsize   # Y轴刻度字体
})

dataset_names=["musique", "bamboogle", "hotpotqa", "2wiki", "nq", "SimpleQA", "amc23", "aime24", "aime25", "math", "math500", "gsm8k", "OlymBench-math"]
# dataset_names=["musique", "2wiki", "math", "aime25"]

mode_name="budget_no_limit_run"
model_names=["tool_star_qwen_3b_origin_gpu18","tool_star_qwen_7b_origin", "tool_star_qwen_3b_oct_clip_radio_gradclip_02_two_epoch_down_progressive_seq_mean_new_add_global_step_110", "tool_star_qwen_7b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_new_global_step_78","ARPO_3b","ARPO_7b","ARPO_3b_oct_clip_radio_gradclip_02_two_epoch_down_progressive_seq_mean_smooth_global_step_156","ARPO_7b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_em_score_seq_mean_smooth_origin_global_step_78"]
# model_names=["tool_star_qwen_3b_origin_gpu18", "tool_star_qwen_3b_oct_clip_radio_gradclip_02_two_epoch_down_progressive_seq_mean_new_add_global_step_110"]
# model_names=["ARPO_3b", "ARPO_3b_oct_clip_radio_gradclip_02_two_epoch_down_progressive_seq_mean_smooth_global_step_156"]
# model_names=["ARPO_7b", "ARPO_7b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_em_score_seq_mean_smooth_origin_global_step_78"]

data_dir = f"/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/result/"
specific_rollout_iter_num = -1

# 添加tokenizer路径配置
TOKENIZER_PATH = "/home/sxjiang/model/Tool-Star-Qwen-7B"  

def get_tokenizer():
    """获取tokenizer实例"""
    if not TOKENIZER_PATH:
        raise ValueError("请在TOKENIZER_PATH变量中设置tokenizer路径")
    return AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)

def count_tokens_with_tokenizer(text, tokenizer):
    """使用tokenizer计算token数量"""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return len(token_ids)


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
    base_output_dir = "/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/img/token_analysis/debug"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # 如果有多个模型，为每个模型创建单独的结果文件
    for model_name, model_data in modelname2dict.items():
        cleaned_model_name = clean_model_name(model_name)
        
        # 为单个模型创建目录
        model_output_dir = os.path.join(base_output_dir, cleaned_model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # 计算该模型的总体统计信息
        all_metrics = [
            "avg_of_total_reason_token_per_sequence",
            "avg_of_total_search_token_per_sequence", 
            "avg_of_total_python_token_per_sequence",
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
                overall_metrics[f"{metric }_max"] = max(values)
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

# 初始化tokenizer
tokenizer = get_tokenizer()
print(f"成功加载tokenizer: {TOKENIZER_PATH}")

for root,dirs,files in os.walk(data_dir):
    if "one_epoch_warmup" in root:
        continue
    if "nq" in root:
        pass
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
                reason_call_token_len = 0
                tool_call_token_len = 0
                tool_call_token_len_2 = 0
                call_num = 0
                call_num_2 = 0
                for item in data: 
                    if item['Metrics']['llm_response'] ==  'Correct' or 'Incorrect'  in item['Metrics']['llm_response']:
                        total_think_sequence_len = 0
                        total_search_sequence_len = 0
                        total_python_sequence_len = 0
                        
                        # 提取item['Full_output'] 中， <think> 和 </think> 之间的token数量， 注意Full_output中可能会有多组<think> </think> ， 也就是直到遇到<answer>之前都会有<think>
                        think_tokens_sequence = re.findall(r'<think>(.*?)</think>', item['Full_output'], re.DOTALL)
                        search_tokens_sequence = re.findall(r'<search>(.*?)</search>', item['Full_output'], re.DOTALL)
                        python_tokens_sequence = re.findall(r'<python>(.*?)</python>', item['Full_output'], re.DOTALL)
                        call_times = item['calling_rounds']
                        call_num_2+=call_times
                        # 使用tokenizer计算token长度
                        for think_tokens in think_tokens_sequence:
                            token_count = count_tokens_with_tokenizer(think_tokens, tokenizer)
                            total_think_sequence_len += token_count
            
                        for think_tokens in think_tokens_sequence[:call_times]:# 只计算调用的think,不计算最后的答案think
                            reason_call_token_len += count_tokens_with_tokenizer(think_tokens, tokenizer)
                            call_num += 1
                        # assert call_times == len(think_tokens_sequence[:call_times])
                        for search_tokens in search_tokens_sequence:
                            token_count = count_tokens_with_tokenizer(search_tokens, tokenizer)
                            total_search_sequence_len += token_count
                            tool_call_token_len += token_count
                            
                        for python_tokens in python_tokens_sequence:
                            token_count = count_tokens_with_tokenizer(python_tokens, tokenizer)
                            total_python_sequence_len += token_count
                            tool_call_token_len += token_count

                        # 计算所有<think>的总token长度
                        total_reason_token_len.append(total_think_sequence_len)
                        # 计算所有<search>的总token长度
                        total_search_token_len.append(total_search_sequence_len)
                        # 计算所有<python>的总token长度
                        total_python_token_len.append(total_python_sequence_len)

                        # 计算每段<think>的平均token长度
                        avg_of_think_sequence_len = total_think_sequence_len / len(think_tokens_sequence) if think_tokens_sequence else 0
                        per_reason_token_len.append(avg_of_think_sequence_len)
                        tool_call_token_len_2 += total_search_sequence_len + total_python_sequence_len
                        # 计算每段<search>的平均token长度
                        avg_of_search_sequence_len = total_search_sequence_len / len(search_tokens_sequence) if search_tokens_sequence else 0
                        per_search_token_len.append(avg_of_search_sequence_len)
                        
                        # 计算每段<python>的平均token长度
                        avg_of_python_sequence_len = total_python_sequence_len / len(python_tokens_sequence) if python_tokens_sequence else 0
                        per_python_token_len.append(avg_of_python_sequence_len)
                        
                        # 计算每次调用的<think>的平均token长度， 如果调用为0，和call_times=1一样
                        if call_times > 0:
                            per_call_reason_token_len.append(total_think_sequence_len / call_times)#FIXME:存在只输出答案的think
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
                
                #per_sequence
                avg_of_total_reason_token_per_sequence = sum(total_reason_token_len) / len(total_reason_token_len) if total_reason_token_len else 0
                avg_of_total_search_token_per_sequence = sum(total_search_token_len) / len(total_search_token_len) if total_search_token_len else 0
                avg_of_total_python_token_per_sequence = sum(total_python_token_len) / len(total_python_token_len) if total_python_token_len else 0
                
                #per_call
                avg_of_mean_sequence_reason_token_per_call = sum(per_reason_token_len) / len(per_reason_token_len) if per_reason_token_len else 0
                avg_of_mean_sequence_search_token_per_call = sum(per_search_token_len) / len(per_search_token_len) if per_search_token_len else 0
                avg_of_mean_sequence_python_token_per_call = sum(per_python_token_len) / len(per_python_token_len) if per_python_token_len else 0
                            
                #per_call_wrong
                avg_of_reason_token_per_call = sum(per_call_reason_token_len) / len(per_call_reason_token_len) if per_call_reason_token_len else 0
                avg_of_search_token_per_call = sum(per_call_search_token_len) / len(per_call_search_token_len) if per_call_search_token_len else 0
                avg_of_python_token_per_call = sum(per_call_python_token_len) / len(per_call_python_token_len) if per_call_python_token_len else 0

            # 存储指标数据
            modelname2dict[path_model_name][path_dataset_name]["avg_of_total_reason_token_per_sequence"] = avg_of_total_reason_token_per_sequence
            modelname2dict[path_model_name][path_dataset_name]["avg_of_total_search_token_per_sequence"] = avg_of_total_search_token_per_sequence
            modelname2dict[path_model_name][path_dataset_name]["avg_of_total_python_token_per_sequence"] = avg_of_total_python_token_per_sequence
            
            modelname2dict[path_model_name][path_dataset_name]["avg_of_mean_sequence_reason_token"] = avg_of_mean_sequence_reason_token_per_call
            modelname2dict[path_model_name][path_dataset_name]["avg_of_mean_sequence_search_token"] = avg_of_mean_sequence_search_token_per_call
            modelname2dict[path_model_name][path_dataset_name]["avg_of_mean_sequence_python_token"] = avg_of_mean_sequence_python_token_per_call
            
            modelname2dict[path_model_name][path_dataset_name]["avg_of_reason_token_per_call"] = avg_of_reason_token_per_call
            modelname2dict[path_model_name][path_dataset_name]["avg_of_search_token_per_call"] = avg_of_search_token_per_call
            modelname2dict[path_model_name][path_dataset_name]["avg_of_python_token_per_call"] = avg_of_python_token_per_call
            print(f"call_num:{call_num},call_num2:{call_num_2}")
            print(f"tool_call_token_len:{tool_call_token_len},tool_call_token_len_2:{tool_call_token_len_2}")
            modelname2dict[path_model_name][path_dataset_name]["avg_of_reason_token_per_call_new"] = reason_call_token_len / call_num if call_num > 0 else reason_call_token_len
            modelname2dict[path_model_name][path_dataset_name]["avg_of_tool_token_per_call_new"] = tool_call_token_len / call_num if call_num > 0 else tool_call_token_len

# 在原有的绘图代码之前，先保存JSON结果
save_detailed_results(modelname2dict)
