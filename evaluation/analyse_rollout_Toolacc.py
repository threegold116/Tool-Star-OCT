# 统计Tool usage accuracy指标：EN（easy question not use tool）、ET(easy question use tool)、HT(hard question use tool)、HN(hard question not use tool)

import json
import os
import textwrap
import matplotlib.pyplot as plt

dataset_names = ["musique", "bamboogle", "hotpotqa", "2wiki", "nq", "SimpleQA", "amc23", "aime24", "aime25", "math", "math500", "gsm8k", "OlymBench-math"]
data_dir = "/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/result/"

# 手动配置模型名称映射（no-tool模型名 -> no-limit模型名）
model_mapping = {
    # 示例：你需要根据实际情况填写
    "tool_star_qwen_7b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_new_global_step_78_t10_p95": "tool_star_qwen_7b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_new_global_step_78",
    # 添加更多映射...
    # "no_tool_model_name": "corresponding_no_limit_model_name"
    "tool_star_qwen_7b_origin_t10_p95":"tool_star_qwen_7b_origin",
    "tool_star_qwen_3b_oct_clip_radio_gradclip_02_two_epoch_down_progressive_seq_mean_new_add_global_step_110_seed1234": "tool_star_qwen_3b_oct_clip_radio_gradclip_02_two_epoch_down_progressive_seq_mean_new_add_global_step_110",
    "tool_star_qwen_3b_origin_seed1234": "tool_star_qwen_3b_origin_gpu18",
    "ARPO_7b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_em_score_seq_mean_smooth_origin_global_step_78_t10_p95_new": "ARPO_7b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_em_score_seq_mean_smooth_origin_global_step_78",
    "ARPO_7b_t10_p95_new": "ARPO_7b",
    "ARPO_3b_oct_clip_radio_gradclip_02_two_epoch_down_progressive_seq_mean_smooth_global_step_156_seed1234": "ARPO_3b_oct_clip_radio_gradclip_02_two_epoch_down_progressive_seq_mean_smooth_global_step_156",
    "ARPO_3b_t10_p95_new": "ARPO_3b"
}

def load_metrics_data(file_path):
    """加载metrics文件数据"""
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return None
    
    with open(file_path, "r") as f:
        return json.load(f)

def is_correct_answer(item):
    """判断答案是否正确"""
    metrics = item.get('Metrics', {})
    
    if 'llm_response' in metrics:
        return metrics['llm_response'] == 'Correct'
    else:
        return False

def has_tool_usage(item):
    """判断是否使用了工具"""
    calling_rounds = item.get('calling_rounds', 0)
    return calling_rounds > 0

def calculate_tool_usage_accuracy(dataset_name, no_tool_model, no_limit_model):
    """计算单个数据集的Tool usage accuracy"""
    
    # 构建文件路径
    no_tool_path = os.path.join(data_dir, dataset_name, "no-tool-inference_origin_prompt", no_tool_model, "result.metrics.json")
    no_limit_path = os.path.join(data_dir, dataset_name, "budget_no_limit_run", no_limit_model, "result.metrics.json")
    
    # 加载数据
    no_tool_data = load_metrics_data(no_tool_path)
    no_limit_data = load_metrics_data(no_limit_path)
    
    if no_tool_data is None or no_limit_data is None:
        print(f"无法加载数据集 {dataset_name} 的数据")
        return None
    
    # 确保两个数据集的题目顺序一致（通过问题内容匹配）
    no_tool_dict = {}
    for item in no_tool_data:
        question = item.get('Question', '')
        no_tool_dict[question] = item
    
    # 统计四个指标
    EN = 0  # easy question not use tool
    ET = 0  # easy question use tool  
    HT = 0  # hard question use tool
    HN = 0  # hard question not use tool
    
    matched_questions = 0
    
    for no_limit_item in no_limit_data:
        question = no_limit_item.get('Question', '')
        
        if question not in no_tool_dict:
            continue
            
        matched_questions += 1
        no_tool_item = no_tool_dict[question]
        
        # 判断是否为简单题目（no-tool模式下能答对）
        is_easy = is_correct_answer(no_tool_item)
        
        # 判断在no-limit模式下是否使用了工具
        used_tool = has_tool_usage(no_limit_item)
        
        # 统计四个指标
        if is_easy:
            if used_tool:
                ET += 1
            else:
                EN += 1
        else:  # hard question
            if used_tool:
                HT += 1
            else:
                HN += 1
    
    # 计算Tool usage accuracy
    # 暂时不可能出现(HT + HN) == 0的情况
    if (EN + ET) == 0 or (HT + HN) == 0:
        if (EN + ET) == 0:
            tool_usage_accuracy = 1.0 * (HT / (HT + HN))
    else:
        easy_accuracy = EN / (EN + ET)
        hard_accuracy = HT / (HT + HN)
        tool_usage_accuracy = 0.5 * (easy_accuracy + hard_accuracy)
    
    result = {
        'dataset': dataset_name,
        'EN': EN,
        'ET': ET, 
        'HT': HT,
        'HN': HN,
        'easy_accuracy': EN / (EN + ET) if (EN + ET) > 0 else 0,
        'hard_accuracy': HT / (HT + HN) if (HT + HN) > 0 else 0,
        'tool_usage_accuracy': tool_usage_accuracy,
        'total_questions': matched_questions,
        'easy_questions': EN + ET,
        'hard_questions': HT + HN
    }
    
    return result

def visualize_results(results, model_pair_name):
    """可视化结果"""
    
    if not results:
        print("没有可视化的数据")
        return
    
    # 创建保存路径
    result_dir = f"/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/img/tool_usage_accuracy/{model_pair_name}"
    os.makedirs(result_dir, exist_ok=True)
    
    datasets = [r['dataset'] for r in results]
    tool_usage_accuracies = [r['tool_usage_accuracy'] for r in results]
    easy_accuracies = [r['easy_accuracy'] for r in results]
    hard_accuracies = [r['hard_accuracy'] for r in results]
    
    # 1. Tool Usage Accuracy 柱状图
    plt.figure(figsize=(16, 10))
    wrapped_labels = ['\n'.join(textwrap.wrap(name, 8)) for name in datasets]
    bars = plt.bar(wrapped_labels, tool_usage_accuracies, color='skyblue', alpha=0.8)
    
    # 在柱子上显示数值
    for bar, value in zip(bars, tool_usage_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Tool Usage Accuracy', fontsize=12)
    plt.title('Tool Usage Accuracy by Dataset', fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "tool_usage_accuracy.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Easy vs Hard Accuracy 对比图
    plt.figure(figsize=(16, 10))
    x = range(len(datasets))
    width = 0.35
    
    bars1 = plt.bar([i - width/2 for i in x], easy_accuracies, width, 
                   label='Easy Questions (EN/(EN+ET))', color='lightgreen', alpha=0.8)
    bars2 = plt.bar([i + width/2 for i in x], hard_accuracies, width,
                   label='Hard Questions (HT/(HT+HN))', color='lightcoral', alpha=0.8)
    
    # 显示数值
    for bar, value in zip(bars1, easy_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    for bar, value in zip(bars2, hard_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Easy vs Hard Questions Tool Usage Accuracy', fontsize=14, pad=20)
    plt.xticks(x, wrapped_labels, rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "easy_vs_hard_accuracy.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. EN, ET, HT, HN 分布图
    plt.figure(figsize=(16, 10))
    
    EN_values = [r['EN'] for r in results]
    ET_values = [r['ET'] for r in results]
    HT_values = [r['HT'] for r in results]
    HN_values = [r['HN'] for r in results]
    
    x = range(len(datasets))
    width = 0.2
    
    plt.bar([i - 1.5*width for i in x], EN_values, width, label='EN (Easy, No Tool)', color='lightblue')
    plt.bar([i - 0.5*width for i in x], ET_values, width, label='ET (Easy, Tool)', color='lightgreen')
    plt.bar([i + 0.5*width for i in x], HT_values, width, label='HT (Hard, Tool)', color='orange')
    plt.bar([i + 1.5*width for i in x], HN_values, width, label='HN (Hard, No Tool)', color='lightcoral')
    
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Number of Questions', fontsize=12)
    plt.title('Distribution of EN, ET, HT, HN by Dataset', fontsize=14, pad=20)
    plt.xticks(x, wrapped_labels, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "EN_ET_HT_HN_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """主函数"""
    
    print("Tool Usage Accuracy 计算工具")
    print("=" * 50)
    
    # 显示当前的模型映射配置
    print("\n当前模型映射配置:")
    for no_tool_model, no_limit_model in model_mapping.items():
        print(f"  No-tool: {no_tool_model}")
        print(f"  No-limit: {no_limit_model}")
        print()
    
    if not model_mapping:
        print("请在代码中的 model_mapping 字典中配置模型名称映射！")
        return
    
    # 为每个模型对计算指标
    for no_tool_model, no_limit_model in model_mapping.items():
        print(f"\n处理模型对:")
        print(f"  No-tool: {no_tool_model}")
        print(f"  No-limit: {no_limit_model}")
        print("-" * 50)
        
        results = []
        
        # 计算每个数据集的指标
        for dataset_name in dataset_names:
            print(f"\n处理数据集: {dataset_name}")
            
            result = calculate_tool_usage_accuracy(dataset_name, no_tool_model, no_limit_model)
            
            if result:
                results.append(result)
                print(f"  EN: {result['EN']}, ET: {result['ET']}, HT: {result['HT']}, HN: {result['HN']}")
                print(f"  Easy Accuracy: {result['easy_accuracy']:.3f}")
                print(f"  Hard Accuracy: {result['hard_accuracy']:.3f}")
                print(f"  Tool Usage Accuracy: {result['tool_usage_accuracy']:.3f}")
                print(f"  总题目数: {result['total_questions']}")
            else:
                print(f"  跳过数据集 {dataset_name}")
        
        if results:
            # 计算总体平均值
            overall_tool_usage_accuracy = sum(r['tool_usage_accuracy'] for r in results) / len(results)
            total_EN = sum(r['EN'] for r in results)
            total_ET = sum(r['ET'] for r in results)
            total_HT = sum(r['HT'] for r in results)
            total_HN = sum(r['HN'] for r in results)
            
            print(f"\n总体统计:")
            print(f"  总体 Tool Usage Accuracy: {overall_tool_usage_accuracy:.3f}")
            print(f"  总计 EN: {total_EN}, ET: {total_ET}, HT: {total_HT}, HN: {total_HN}")
            
            # 生成可视化图表 - 直接使用no-limit模型名称
            model_pair_name = no_limit_model
            visualize_results(results, model_pair_name)
            
            # 保存详细结果到JSON文件
            output_path = f"/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/img/tool_usage_accuracy/{model_pair_name}/detailed_results.json"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump({
                    'model_mapping': {
                        'no_tool_model': no_tool_model,
                        'no_limit_model': no_limit_model
                    },
                    'overall_metrics': {
                        'overall_tool_usage_accuracy': overall_tool_usage_accuracy,
                        'total_EN': total_EN,
                        'total_ET': total_ET,
                        'total_HT': total_HT,
                        'total_HN': total_HN
                    },
                    'per_dataset_results': results
                }, f, indent=4, ensure_ascii=False)
            
            print(f"\n详细结果已保存到: {output_path}")
        else:
            print(f"\n没有找到该模型对的有效数据")

if __name__ == "__main__":
    main()