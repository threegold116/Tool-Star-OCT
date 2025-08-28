import json
data_path="/share/home/sxjiang/dataset/Tool-Star-SFT-54K/final_sft_edition9_v2.json"
with open(data_path) as f:
    data = json.load(f)
search_item_num=0
python_item_num=0
search_and_python_item_num=0
python_item_num_without_tool=0
for item in data:
    if item["output"].count("</search>")>=1:
        search_item_num+=1
    if item["output"].count("</python>")>=1:
        python_item_num+=1
    if item["output"].count("</python>")>=1 and item["output"].count("</search>")>=1:
        search_and_python_item_num+=1
    if item["output"].count("</python>")==0 and item["output"].count("</search>")==0:
        python_item_num_without_tool+=1
print(f"search_item_num: {search_item_num}")
print(f"python_item_num: {python_item_num}")
print(f"search_and_python_item_num: {search_and_python_item_num}")
print(f"python_item_num_without_tool: {python_item_num_without_tool}")


