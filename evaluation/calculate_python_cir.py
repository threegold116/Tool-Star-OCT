import json
import re
from collections import defaultdict

def analyze_json_file(file_path):
    """
    分析JSON文件，统计calling_rounds和Full_output中```python的出现次数
    
    Args:
        file_path (str): JSON文件路径
    
    Returns:
        dict: 包含统计结果的字典
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"错误: 文件 {file_path} 不是有效的JSON格式")
        return None
    
    # 初始化统计变量
    stats = {
        'total_cases': 0,
        'calling_rounds_stats': defaultdict(int),
        'python_blocks_stats': defaultdict(int),
        'cases_with_python': 0,
        'cases_without_python': 0,
        'total_python_blocks': 0,
        'average_python_blocks_per_case': 0,
        'calling_rounds_sum': 0,
        'average_calling_rounds': 0
    }
    
    # 如果data是列表，遍历每个case
    if isinstance(data, list):
        cases = data
    else:
        # 如果data是字典，可能包含cases的键
        cases = [data]  # 假设单个case
    
    for case in cases:
        if not isinstance(case, dict):
            continue
            
        stats['total_cases'] += 1
        
        # # 统计calling_rounds (包括python_rounds)
        # calling_rounds = case.get('calling_rounds', 0)
        # python_rounds = case.get('python_rounds', 0)
        
        # # 只统计python_rounds作为calling_rounds
        # effective_calling_rounds = python_rounds
        # stats['calling_rounds_stats'][effective_calling_rounds] += 1
        # stats['calling_rounds_sum'] += effective_calling_rounds
        
        # 统计Full_output中```python的出现次数
        full_output = case.get('Full_output', '')
        
        # 使用正则表达式查找```python，但排除```\n```python的情况
        # 查找所有```python的位置
        python_pattern = r'```python'
        all_matches = list(re.finditer(python_pattern, full_output))
        
        # 过滤掉```\n```python的情况
        valid_matches = []
        for match in all_matches:
            start_pos = match.start()
            # 检查```python前面是否是```\n
            if start_pos >= 4:  # 至少需要4个字符来检查```\n
                preceding_text = full_output[start_pos-4:start_pos]
                if preceding_text == '```\n':
                    continue  # 跳过```\n```python的情况
            valid_matches.append(match)
        
        python_count = len(valid_matches)
        stats['python_blocks_stats'][python_count] += 1
        stats['total_python_blocks'] += python_count
        
        if python_count > 0:
            stats['cases_with_python'] += 1
        else:
            stats['cases_without_python'] += 1
    
    # 计算平均值
    if stats['total_cases'] > 0:
        stats['average_python_blocks_per_case'] = stats['total_python_blocks'] / stats['total_cases']
        # stats['average_calling_rounds'] = stats['calling_rounds_sum'] / stats['total_cases']
    
    return stats

def print_statistics(stats):
    """打印统计结果"""
    if stats is None:
        return
    
    print("=" * 60)
    print("JSON文件统计结果")
    print("=" * 60)
    
    print(f"总case数量: {stats['total_cases']}")
    print()
    
    print("```python代码块统计:")
    print("-" * 40)
    for python_count, case_count in sorted(stats['python_blocks_stats'].items()):
        percentage = (case_count / stats['total_cases']) * 100 if stats['total_cases'] > 0 else 0
        print(f"  {python_count} 个代码块: {case_count} cases ({percentage:.1f}%)")
    
    print(f"\n包含Python代码的cases: {stats['cases_with_python']}")
    print(f"不包含Python代码的cases: {stats['cases_without_python']}")
    print(f"总Python代码块数量: {stats['total_python_blocks']}")
    print(f"平均每个case的Python代码块数量: {stats['average_python_blocks_per_case']:.2f}")

def main():
    """主函数"""
    # 在这里修改文件路径
    file_path = "/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/result/OlymBench-math/budget_no_limit_run/cir_qwen_instruct_3b_new_step140/result.json"
    
    print(f"正在分析文件: {file_path}")
    print()
    
    # 执行统计分析
    stats = analyze_json_file(file_path)
    
    # 打印结果
    print_statistics(stats)
    
    # 可选：保存结果到文件
    save_to_file = input("\n是否将结果保存到文件? (y/n): ").lower().strip()
    if save_to_file == 'y':
        output_file = input("请输入输出文件名 (默认: analysis_result.txt): ").strip()
        if not output_file:
            output_file = "analysis_result.txt"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("JSON文件统计结果\n")
                f.write("=" * 60 + "\n")
                f.write(f"分析文件: {file_path}\n\n")
                
                f.write(f"总case数量: {stats['total_cases']}\n\n")
                
                f.write("```python代码块统计:\n")
                f.write("-" * 40 + "\n")
                for python_count, case_count in sorted(stats['python_blocks_stats'].items()):
                    percentage = (case_count / stats['total_cases']) * 100 if stats['total_cases'] > 0 else 0
                    f.write(f"  {python_count} 个代码块: {case_count} cases ({percentage:.1f}%)\n")
                
                f.write(f"\n包含Python代码的cases: {stats['cases_with_python']}\n")
                f.write(f"不包含Python代码的cases: {stats['cases_without_python']}\n")
                f.write(f"总Python代码块数量: {stats['total_python_blocks']}\n")
                f.write(f"平均每个case的Python代码块数量: {stats['average_python_blocks_per_case']:.2f}\n")
            
            print(f"结果已保存到: {output_file}")
        except Exception as e:
            print(f"保存文件时出错: {e}")

if __name__ == "__main__":
    main()