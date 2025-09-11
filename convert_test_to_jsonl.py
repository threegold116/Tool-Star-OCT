#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
转换 Test.txt 文件为 JSONL 格式
每个测试用例占8行：空行、答案字母、上下文、问题、四个选项(A-D)
"""

import json


def parse_test_file(file_path):
    """解析Test.txt文件并转换为JSONL格式"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 去除每行末尾的换行符
    lines = [line.rstrip('\n') for line in lines]
    
    jsonl_data = []
    
    # 每8行为一个测试用例
    for i in range(0, len(lines), 8):
        if i + 7 >= len(lines):
            break
        
        # 获取8行数据
        blank_line = lines[i]
        answer_letter = lines[i + 1].strip().lower()  # 答案字母(a/b/c/d)
        context = lines[i + 2].strip()
        question = lines[i + 3].strip()
        option_a = lines[i + 4].strip()
        option_b = lines[i + 5].strip()
        option_c = lines[i + 6].strip()
        option_d = lines[i + 7].strip()
        
        # 组合问题：上下文 + 问题 + 四个选项
        full_question = f"{context}\n{question}\n{option_a}\n{option_b}\n{option_c}\n{option_d}"
        
        # 找到正确答案的文字内容
        options = {
            'a': option_a,
            'b': option_b, 
            'c': option_c,
            'd': option_d
        }
        
        # 获取正确答案的文字，去掉前面的字母标识
        correct_option_text = options.get(answer_letter, "")
        if correct_option_text.startswith(answer_letter.upper() + '.'):
            correct_option_text = correct_option_text[2:].strip()
        elif correct_option_text.startswith(answer_letter.upper()):
            correct_option_text = correct_option_text[1:].strip()
        
        # 构造JSON对象，将答案字母和文字合并为一个字符串
        combined_answer = f"{answer_letter.upper()}. {correct_option_text}"
        json_obj = {
            "question": full_question,
            "answer": [combined_answer]
        }
        
        jsonl_data.append(json_obj)
    
    return jsonl_data


def save_jsonl(data, output_path):
    """保存数据为JSONL格式"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    input_file = '/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/data/logiqa/Test.txt'
    output_file = '/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/data/logiqa/Test.jsonl'
    
    print("开始转换Test.txt文件...")
    
    # 解析文件
    jsonl_data = parse_test_file(input_file)
    
    print(f"解析完成，共找到 {len(jsonl_data)} 个测试用例")
    
    # 保存JSONL文件
    save_jsonl(jsonl_data, output_file)
    
    print(f"转换完成！输出文件：{output_file}")
    
    # 显示前几个例子
    print("\n前3个转换示例：")
    for i, item in enumerate(jsonl_data[:3]):
        print(f"\n示例 {i+1}:")
        print(json.dumps(item, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
