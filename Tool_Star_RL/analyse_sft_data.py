import json
import jsonlines
data_path="/share/home/sxjiang/dataset/ARPO-SFT-54K/final_5w4_still.jsonl"
data_path="/share/home/sxjiang/dataset/Tool-Star-SFT-54K/final_sft_edition9_v2.json"
if data_path.endswith(".jsonl"):    
    with jsonlines.open(data_path) as reader:
        data = list(reader)
elif data_path.endswith(".json"):
    with open(data_path) as f:
        data = json.load(f)
else:
    raise ValueError(f"Unsupported file extension: {data_path}")
search_item_num=0
python_item_num=0
search_and_python_item_num=0
python_item_num_without_tool=0
for item in data:
    if "conversations" in item:
        output = item["conversations"][-1]["value"]
    else:
        output = item["output"]
    
    if output.count("</search>")>=1:
        search_item_num+=1
    if output.count("</python>")>=1:
        python_item_num+=1
    if output.count("</python>")>=1 and output.count("</search>")>=1:
        search_and_python_item_num+=1
    if output.count("</python>")==0 and output.count("</search>")==0:
        python_item_num_without_tool+=1
print(f"search_item_num: {search_item_num}")
print(f"python_item_num: {python_item_num}")
print(f"search_and_python_item_num: {search_and_python_item_num}")
print(f"python_item_num_without_tool: {python_item_num_without_tool}")


