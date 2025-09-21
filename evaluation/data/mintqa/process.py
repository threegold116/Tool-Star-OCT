import json

# 输入文件路径
input_file = "/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/data/mintqa/MINTQA-POP.json"
# 输出文件路径
output_file = "/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/data/mintqa/test.jsonl"

# 打开输入文件并读取数据
with open(input_file, "r", encoding="utf-8") as infile:
    lines = infile.readlines()

# 打开输出文件准备写入
with open(output_file, "w", encoding="utf-8") as outfile:
    for line in lines:
        # 解析每一行 JSON 数据
        data = json.loads(line)
        # 提取 question 和 answer
        question = data.get("question", "")
        answer = data.get("answer", "")
        # 确保 answer 是列表格式
        if not isinstance(answer, list):
            answer = [answer]
        # 写入到输出文件
        json.dump({"question": question, "answer": answer}, outfile, ensure_ascii=False)
        outfile.write("\n")

print(f"提取完成，结果已保存到 {output_file}")