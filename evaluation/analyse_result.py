import os
import json

result_dir= "/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/result"
# dataset_name = "hle"

for dataset_name in os.listdir(result_dir):
    dataset_path = os.path.join(result_dir, dataset_name)
    for root,dirs,files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file), "r") as f:
                    data = json.load(f)
                normal_answer_num = 0
                for line in data:
                    output = line["Full_output"]
                    if "<answer>" in output:
                        normal_answer_num += 1
                print(f"path:\n{os.path.join(root, file)}, normal_answer_radio:{normal_answer_num/len(data)}")
                    
                    
