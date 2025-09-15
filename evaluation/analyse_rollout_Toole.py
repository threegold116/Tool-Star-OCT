# 统计OCT模型与SFT模型对比指标：ME、LE、MA、LA

import json
import os
import textwrap
import matplotlib.pyplot as plt
import numpy as np

dataset_names = ["musique", "bamboogle", "hotpotqa", "2wiki", "nq", "SimpleQA", "amc23", "aime24", "aime25", "math", "math500", "gsm8k", "OlymBench-math"]
data_dir = "/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/result/"

# 手动配置模型名称映射（OCT模型名 -> SFT模型名）
model_mapping = {
    # 示例：你需要根据实际情况填写
    "tool_star_qwen_3b_oct_clip_radio_gradclip_02_two_epoch_down_progressive_seq_mean_new_add_global_step_110": "tool_star_qwen_3b_sft",
    # 添加更多映射...
    # "oct_model_name": "corresponding_sft_model_name"
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
    
    # 根据不同的评估指标判断正确性
    if 'llm_response' in metrics:
        return metrics['llm_response'] == 'Correct'
    elif 'em' in metrics:
        return metrics['em'] == 1
    elif 'acc' in metrics:
        return metrics['acc'] == 1
    elif 'f1' in metrics:
        return metrics['f1'] >= 0.5  # 可以根据需要调整阈值
    elif 'math_equal' in metrics:
        return metrics['math_equal'] == True
    else:
        return False

def get_tool_usage_count(item):
    """获取工具使用次数"""
    calling_rounds = item.get('calling_rounds', 0)
    return calling_rounds

def get_output(item):
    """获取模型输出"""
    return item.get('Output', '')

def calculate_comparison_metrics(dataset_name, oct_model, sft_model):
    """计算单个数据集的对比指标"""
    
    # 构建文件路径
    oct_path = os.path.join(data_dir, dataset_name, "budget_no_limit_run", oct_model, "result.metrics.json")
    sft_path = os.path.join(data_dir, dataset_name, "budget_no_limit_run", sft_model, "result.metrics.json")
    
    # 加载数据
    oct_data = load_metrics_data(oct_path)
    sft_data = load_metrics_data(sft_path)
    
    if oct_data is None or sft_data is None:
        print(f"无法加载数据集 {dataset_name} 的数据")
        return None
    
    # 确保两个数据集的题目顺序一致（通过问题内容匹配）
    sft_dict = {}
    for item in sft_data:
        question = item.get('Question', '')
        sft_dict[question] = item
    
    # 统计四个指标
    ME = 0  # OCT模型与SFT模型输出相同，但OCT使用更少工具
    LE = 0  # OCT模型与SFT模型输出相同，但OCT使用更多工具
    MA = 0  # OCT模型答对但SFT模型答错
    LA = 0  # OCT模型答错但SFT模型答对
    
    matched_questions = 0
    same_output_count = 0  # 用于计算ME和LE的分母
    
    for oct_item in oct_data:
        question = oct_item.get('Question', '')
        
        if question not in sft_dict:
            continue
            
        matched_questions += 1
        sft_item = sft_dict[question]
        
        # 获取相关信息
        oct_output = get_output(oct_item)
        sft_output = get_output(sft_item)
        oct_correct = is_correct_answer(oct_item)
        sft_correct = is_correct_answer(sft_item)
        oct_tool_count = get_tool_usage_count(oct_item)
        sft_tool_count = get_tool_usage_count(sft_item)
        
        # 统计MA和LA
        if oct_correct and not sft_correct:
            MA += 1
        elif not oct_correct and sft_correct:
            LA += 1
        
        # 统计ME和LE（需要输出相同）
        if oct_output == sft_output:
            same_output_count += 1
            if oct_tool_count < sft_tool_count:
                ME += 1
            elif oct_tool_count > sft_tool_count:
                LE += 1
    
    # 计算比例
    me_ratio = ME / matched_questions if matched_questions > 0 else 0
    le_ratio = LE / matched_questions if matched_questions > 0 else 0
    ma_ratio = MA / matched_questions if matched_questions > 0 else 0
    la_ratio = LA / matched_questions if matched_questions > 0 else 0
    
    result = {
        'dataset': dataset_name,
        'ME': ME,
        'LE': LE,
        'MA': MA,
        'LA': LA,
        'ME_ratio': me_ratio,
        'LE_ratio': le_ratio,
        'MA_ratio': ma_ratio,
        'LA_ratio': la_ratio,
        'total_questions': matched_questions,
        'same_output_count': same_output_count
    }
    
    return result

def visualize_results(results, model_pair_name):
    """可视化结果"""
    
    if not results:
        print("没有可视化的数据")
        return
    
    # 创建保存路径
    result_dir = f"/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/img/oct_vs_sft_comparison/{model_pair_name}"
    os.makedirs(result_dir, exist_ok=True)
    
    datasets = [r['dataset'] for r in results]
    me_ratios = [r['ME_ratio'] for r in results]
    le_ratios = [r['LE_ratio'] for r in results]
    ma_ratios = [r['MA_ratio'] for r in results]
    la_ratios = [r['LA_ratio'] for r in results]
    
    # 1. 四个指标比例对比图
    plt.figure(figsize=(16, 10))
    wrapped_labels = ['\n'.join(textwrap.wrap(name, 8)) for name in datasets]
    
    x = np.arange(len(datasets))
    width = 0.2
    
    bars1 = plt.bar(x - 1.5*width, me_ratios, width, label='ME (OCT用更少工具，输出相同)', color='lightblue', alpha=0.8)
    bars2 = plt.bar(x - 0.5*width, le_ratios, width, label='LE (OCT用更多工具，输出相同)', color='lightcoral', alpha=0.8)
    bars3 = plt.bar(x + 0.5*width, ma_ratios, width, label='MA (OCT对，SFT错)', color='lightgreen', alpha=0.8)
    bars4 = plt.bar(x + 1.5*width, la_ratios, width, label='LA (OCT错，SFT对)', color='orange', alpha=0.8)
    
    # 显示数值
    for bars, values in zip([bars1, bars2, bars3, bars4], [me_ratios, le_ratios, ma_ratios, la_ratios]):
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Ratio', fontsize=12)
    plt.title('OCT vs SFT Model Comparison Metrics', fontsize=14, pad=20)
    plt.xticks(x, wrapped_labels, rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, max(max(me_ratios), max(le_ratios), max(ma_ratios), max(la_ratios)) * 1.2)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "comparison_metrics_ratios.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ME vs LE 对比图
    plt.figure(figsize=(16, 10))
    x = range(len(datasets))
    width = 0.35
    
    bars1 = plt.bar([i - width/2 for i in x], me_ratios, width, 
                   label='ME (OCT用更少工具，输出相同)', color='lightblue', alpha=0.8)
    bars2 = plt.bar([i + width/2 for i in x], le_ratios, width,
                   label='LE (OCT用更多工具，输出相同)', color='lightcoral', alpha=0.8)
    
    # 显示数值
    for bar, value in zip(bars1, me_ratios):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    for bar, value in zip(bars2, le_ratios):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Ratio', fontsize=12)
    plt.title('Tool Usage Comparison: ME vs LE', fontsize=14, pad=20)
    plt.xticks(x, wrapped_labels, rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, max(max(me_ratios), max(le_ratios)) * 1.2)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "me_vs_le_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. MA vs LA 对比图
    plt.figure(figsize=(16, 10))
    x = range(len(datasets))
    width = 0.35
    
    bars1 = plt.bar([i - width/2 for i in x], ma_ratios, width, 
                   label='MA (OCT对，SFT错)', color='lightgreen', alpha=0.8)
    bars2 = plt.bar([i + width/2 for i in x], la_ratios, width,
                   label='LA (OCT错，SFT对)', color='orange', alpha=0.8)
    
    # 显示数值
    for bar, value in zip(bars1, ma_ratios):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    for bar, value in zip(bars2, la_ratios):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Ratio', fontsize=12)
    plt.title('Accuracy Comparison: MA vs LA', fontsize=14, pad=20)
    plt.xticks(x, wrapped_labels, rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, max(max(ma_ratios), max(la_ratios)) * 1.2)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "ma_vs_la_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """主函数"""
    
    print("OCT vs SFT 模型对比分析工具")
    print("=" * 50)
    
    # 显示当前的模型映射配置
    print("\n当前模型映射配置:")
    for oct_model, sft_model in model_mapping.items():
        print(f"  OCT模型: {oct_model}")
        print(f"  SFT模型: {sft_model}")
        print()
    
    if not model_mapping:
        print("请在代码中的 model_mapping 字典中配置模型名称映射！")
        return
    
    # 为每个模型对计算指标
    for oct_model, sft_model in model_mapping.items():
        print(f"\n处理模型对:")
        print(f"  OCT模型: {oct_model}")
        print(f"  SFT模型: {sft_model}")
        print("-" * 50)
        
        results = []
        
        # 计算每个数据集的指标
        for dataset_name in dataset_names:
            print(f"\n处理数据集: {dataset_name}")
            
            result = calculate_comparison_metrics(dataset_name, oct_model, sft_model)
            
            if result:
                results.append(result)
                print(f"  ME: {result['ME']} ({result['ME_ratio']:.3f})")
                print(f"  LE: {result['LE']} ({result['LE_ratio']:.3f})")
                print(f"  MA: {result['MA']} ({result['MA_ratio']:.3f})")
                print(f"  LA: {result['LA']} ({result['LA_ratio']:.3f})")
                print(f"  总题目数: {result['total_questions']}")
                print(f"  相同输出题目数: {result['same_output_count']}")
            else:
                print(f"  跳过数据集 {dataset_name}")
        
        if results:
            # 计算总体平均值
            avg_me_ratio = sum(r['ME_ratio'] for r in results) / len(results)
            avg_le_ratio = sum(r['LE_ratio'] for r in results) / len(results)
            avg_ma_ratio = sum(r['MA_ratio'] for r in results) / len(results)
            avg_la_ratio = sum(r['LA_ratio'] for r in results) / len(results)
            
            total_ME = sum(r['ME'] for r in results)
            total_LE = sum(r['LE'] for r in results)
            total_MA = sum(r['MA'] for r in results)
            total_LA = sum(r['LA'] for r in results)
            total_questions = sum(r['total_questions'] for r in results)
            
            print(f"\n总体统计:")
            print(f"  平均 ME 比例: {avg_me_ratio:.3f}")
            print(f"  平均 LE 比例: {avg_le_ratio:.3f}")
            print(f"  平均 MA 比例: {avg_ma_ratio:.3f}")
            print(f"  平均 LA 比例: {avg_la_ratio:.3f}")
            print(f"  总计 ME: {total_ME}, LE: {total_LE}, MA: {total_MA}, LA: {total_LA}")
            print(f"  总题目数: {total_questions}")
            
            # 生成可视化图表
            model_pair_name = f"{oct_model.split('_')[2]}_{oct_model.split('_')[3]}_vs_{sft_model}"
            visualize_results(results, model_pair_name)
            
            # 保存详细结果到JSON文件
            output_path = f"/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/img/oct_vs_sft_comparison/{model_pair_name}/detailed_results.json"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump({
                    'model_mapping': {
                        'oct_model': oct_model,
                        'sft_model': sft_model
                    },
                    'overall_metrics': {
                        'avg_ME_ratio': avg_me_ratio,
                        'avg_LE_ratio': avg_le_ratio,
                        'avg_MA_ratio': avg_ma_ratio,
                        'avg_LA_ratio': avg_la_ratio,
                        'total_ME': total_ME,
                        'total_LE': total_LE,
                        'total_MA': total_MA,
                        'total_LA': total_LA,
                        'total_questions': total_questions
                    },
                    'per_dataset_results': results
                }, f, indent=4, ensure_ascii=False)
            
            print(f"\n详细结果已保存到: {output_path}")
        else:
            print(f"\n没有找到该模型对的有效数据")

if __name__ == "__main__":
    main()