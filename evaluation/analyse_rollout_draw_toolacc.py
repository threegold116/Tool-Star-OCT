import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os

# 添加字体文件
fm.fontManager.addfont('/home/sxjiang/myproject/analyse/radar/TIMES.TTF')

# 字体设置 
fontsize = 24

# 设置matplotlib全局字体参数
plt.rcParams.update({
    "legend.fontsize": fontsize,
    "legend.title_fontsize": fontsize,
    "font.size": 15,
    "font.family": "Times New Roman",
    "axes.titlesize": fontsize,
    "axes.labelsize": fontsize,
    "xtick.labelsize": 20,
    "ytick.labelsize": fontsize
})


def load_json_data(file_path):
    """加载JSON文件数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_dataset_metrics(data, dataset_name):
    """从数据中获取指定数据集的指标"""
    for dataset_result in data['per_dataset_results']:
        if dataset_result['dataset'] == dataset_name:
            return dataset_result
    return None

def plot_tool_usage_accuracy():
    """绘制工具使用准确率的堆叠柱状图"""
    
    # 定义文件路径和对应的模型名称
    model_files = [
        ('/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/img/tool_usage_accuracy/tool_star_qwen_7b_origin/detailed_results.json', 'Tool-Star-7B'),
        ('/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/img/tool_usage_accuracy/tool_star_qwen_7b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_seq_mean_new_global_step_78/detailed_results.json', 'Tool-Star-7B(ours)'),
        ('/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/img/tool_usage_accuracy/ARPO_7b/detailed_results.json', 'ARPO-7B'),
        ('/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/img/tool_usage_accuracy/ARPO_7b_oct_clip_radio_gradclip_02_one_epoch_down_progressive_em_score_seq_mean_smooth_origin_global_step_78/detailed_results.json', 'ARPO-7B(ours)'),
       # ('/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/img/tool_usage_accuracy/tool_star_qwen_3b_origin_gpu18/detailed_results.json', 'Tool-Star-3B'),
       # ('/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/img/tool_usage_accuracy/tool_star_qwen_3b_oct_clip_radio_gradclip_02_two_epoch_down_progressive_seq_mean_new_add_global_step_110/detailed_results.json', 'Tool-Star-3B(ours)')
    ]
    
    # 选择的数据集（可扩展）
    selected_datasets = ['2wiki', 'nq', 'math', 'gsm8k']
    
    # 模型顺序
    model_order = ['Tool-Star-7B', 'Tool-Star-7B(ours)', 'ARPO-7B', 'ARPO-7B(ours)']
    
    # 定义指标标签和颜色
    metrics = ['EN', 'ET', 'HT', 'HN']
    colors = ['#F2FA5A', '#5EE6EB', '#4D77FF', '#95D1CC'] 
    metric_labels = {
        'EN': 'Easy & No Tool',
        'ET': 'Easy & Tool Used', 
        'HT': 'Hard & Tool Used',
        'HN': 'Hard & No Tool'
    }
    
    # 加载所有模型数据
    model_data = {}
    for file_path, model_name in model_files:
        if os.path.exists(file_path):
            model_data[model_name] = load_json_data(file_path)
        else:
            print(f"警告: 文件不存在 {file_path}")
    
    # 创建单个图形，根据数据集数量调整宽度
    num_datasets = len(selected_datasets)
    num_models = len(model_order)
    # total_bars = num_datasets * num_models
    
    # 动态调整图形大小
    # fig_width = max(20, total_bars * 1.2)  # 确保有足够宽度
    fig, ax = plt.subplots(1, 1, figsize=(40, 6))
    
    # 设置柱状图参数
    bar_width = 6
    model_spacing = 10  # 同一数据集内模型之间的间距
    group_spacing = 5  # 数据集组之间的间距
    
    # 用于收集图例信息
    legend_handles = []
    legend_labels = []
    
    # 用于存储x轴位置和标签
    all_x_positions = []
    all_x_labels = []
    
    # 用于存储折线数据
    line_x_positions = []
    line_y_values = []
    
    current_x = 0
    
    # 为每个数据集创建一组柱状图
    for dataset_idx, dataset in enumerate(selected_datasets):
        dataset_x_positions = []
        
        # 为该数据集的每个模型创建柱状图
        for model_idx, model_name in enumerate(model_order):
            if model_name in model_data:
                dataset_metrics = get_dataset_metrics(model_data[model_name], dataset)
                if dataset_metrics:
                    # 计算总问题数和各指标百分比
                    total_questions = dataset_metrics['total_questions']
                    
                    # 绘制堆叠柱状图
                    bottom = 0
                    for i, metric in enumerate(metrics):
                        count = dataset_metrics.get(metric, 0)
                        percentage = (count / total_questions) if total_questions > 0 else 0
                        
                        bar = ax.bar(current_x, percentage, bar_width, bottom=bottom,
                                   color=colors[i], alpha=0.8, label=metric_labels[metric])
                        bottom += percentage
                        
                        # 只在第一个柱子收集图例信息
                        if dataset_idx == 0 and model_idx == 0:
                            legend_handles.append(bar)
                            legend_labels.append(metric_labels[metric])
                    
                    # 收集折线数据
                    tool_usage_acc = dataset_metrics.get('tool_usage_accuracy', 0)
                    line_x_positions.append(current_x)
                    line_y_values.append(tool_usage_acc)
                    
                    dataset_x_positions.append(current_x)
                    all_x_positions.append(current_x)
                    all_x_labels.append(model_name)
                    current_x += model_spacing
                else:
                    print(f"警告: 在模型 {model_name} 中找不到数据集 {dataset}")
                    # 即使没有数据也要添加占位符，保持折线连续性
                    line_x_positions.append(current_x)
                    line_y_values.append(0)  # 或者使用None
                    current_x += model_spacing
            else:
                line_x_positions.append(current_x)
                line_y_values.append(0)  # 或者使用None
                current_x += 1
        
        # 在数据集组之间添加间距
        if dataset_idx < len(selected_datasets) - 1:
            current_x += group_spacing
    
    # 绘制折线图
    if line_x_positions and line_y_values:
        # 创建第二个y轴用于显示tool_usage_accuracy
        ax2 = ax.twinx()
        
        # 按数据集分组绘制折线，避免不同数据集间连接
        for dataset_idx, dataset in enumerate(selected_datasets):
            # 获取当前数据集的x位置和y值
            dataset_x_positions = []
            dataset_y_values = []
            
            start_idx = dataset_idx * num_models
            end_idx = start_idx + num_models
            
            # 收集当前数据集的数据点
            for i in range(start_idx, min(end_idx, len(line_x_positions))):
                if i < len(line_x_positions) and line_y_values[i] is not None:
                    dataset_x_positions.append(line_x_positions[i])
                    dataset_y_values.append(line_y_values[i])
            
            # 只有当数据集有足够数据点时才绘制折线
            if len(dataset_x_positions) >= 2:
                line = ax2.plot(dataset_x_positions, dataset_y_values, 
                               color='purple', marker='o', markersize=8, 
                               linewidth=3, alpha=0.8, 
                               label='Tool Usage Accuracy' if dataset_idx == 0 else "")
                
                # 只在第一个数据集添加图例
                if dataset_idx == 0:
                    legend_handles.append(line[0])
                    legend_labels.append('Tool Usage Accuracy')
            elif len(dataset_x_positions) == 1:
                # 如果只有一个数据点，只画点不画线
                ax2.plot(dataset_x_positions, dataset_y_values, 
                        color='purple', marker='o', markersize=8, 
                        alpha=0.8)
        
        # 设置第二个y轴的属性
        ax2.set_ylabel('Tool Usage Accuracy', fontsize=fontsize)
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='y', labelsize=fontsize)
    
    # 设置图表属性
    # ax.set_xlabel('Models grouped by Dataset', fontsize=fontsize, labelpad=20)
    ax.set_ylabel('Proportion', fontsize=fontsize)
    # ax.set_title('Tool Usage Accuracy Comparison Across Datasets', fontsize=fontsize, pad=30)
    
    # 设置x轴刻度和标签
    ax.set_xticks(all_x_positions)
    ax.set_xticklabels(all_x_labels, rotation=0, ha='center')
    ax.set_ylim(0, top=1)
    
    # 添加数据集分组标识
    group_centers = []
    current_pos = 0

    for dataset_idx, dataset in enumerate(selected_datasets):
        # 找到该数据集对应的所有x位置
        dataset_positions = []
        start_idx = dataset_idx * num_models
        end_idx = start_idx + num_models
        
        # 获取该数据集的所有柱状图位置
        for i in range(start_idx, min(end_idx, len(all_x_positions))):
            if i < len(all_x_positions):
                dataset_positions.append(all_x_positions[i])
        
        # 计算该数据集组的中心位置
        if dataset_positions:
            group_center = (min(dataset_positions) + max(dataset_positions)) / 2
            group_centers.append(group_center)
            
            # 在底部添加数据集标签
            ax.text(group_center, -0.1, dataset.upper(), ha='center', va='top', 
                fontsize=fontsize,)
            
            # # 添加垂直分隔线（除了最后一组）
            # if dataset_idx < len(selected_datasets) - 1:
            #     # 分隔线位置应该在当前组最后一个柱子和下一组第一个柱子之间
            #     if end_idx < len(all_x_positions):
            #         separator_x = (max(dataset_positions) + all_x_positions[end_idx]) / 2
            #         ax.axvline(x=separator_x, color='gray', linestyle='--', alpha=0.5)
    
    # 添加网格
    # ax.grid(axis='y', alpha=0.3)
    
    # 添加图例（包含柱状图和折线图）
    if legend_handles and legend_labels:
        ax.legend(legend_handles, legend_labels, 
                 loc='upper left', 
                 bbox_to_anchor=(1.01, 1),
                 frameon=True,
                 fancybox=True,
                 shadow=True)
    
    # 调整布局为图例留空间
    plt.subplots_adjust(right=0.97, bottom=0.15)
    
    # 保存图片
    output_dir = "/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/img/tool_usage_accuracy"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "tool_usage_accuracy_comparison_combined2.svg")
    
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"图片已保存到: {output_path}")
    
    # 打印统计信息
    print("\n=== 工具使用准确率统计 ===")
    for dataset in selected_datasets:
        print(f"\n{dataset.upper()} Dataset:")
        for model_name in model_order:
            if model_name in model_data:
                dataset_metrics = get_dataset_metrics(model_data[model_name], dataset)
                if dataset_metrics:
                    tool_usage_acc = dataset_metrics.get('tool_usage_accuracy', 0)
                    total_questions = dataset_metrics.get('total_questions', 0)
                    print(f"  {model_name}: Tool Usage Accuracy = {tool_usage_acc:.3f}, "
                          f"Total Questions = {total_questions}")
                    
                    # 打印EN, ET, HT, HN的具体数值
                    en = dataset_metrics.get('EN', 0)
                    et = dataset_metrics.get('ET', 0)
                    ht = dataset_metrics.get('HT', 0)
                    hn = dataset_metrics.get('HN', 0)
                    print(f"    EN={en}, ET={et}, HT={ht}, HN={hn}")

def main():
    """主函数"""
    print("开始绘制工具使用准确率对比图...")
    plot_tool_usage_accuracy()
    print("绘图完成!")

if __name__ == "__main__":
    main()