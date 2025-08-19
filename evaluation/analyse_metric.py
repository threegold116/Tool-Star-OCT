import os
import json

result_dir= "/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/result"
# dataset_name = "hle"
new_metrics = []
for dataset_name in os.listdir(result_dir):
    dataset_path = os.path.join(result_dir, dataset_name)
    for root,dirs,files in os.walk(dataset_path):
        for file in files:
            if "result.metrics.overall" in file:
                with open(os.path.join(root, file), "r") as f:
                    metric = json.load(f)
                print(os.path.join(root, file))
                print(os.path.join(root, file).split("/")[-3])
                print(os.path.join(root, file).split("/")[-2])
                new_metric= {
                    "inference_mode": os.path.join(root, file).split("/")[-3],
                    "model_name": os.path.join(root, file).split("/")[-2],
                    "dataset_name": dataset_name,
                    "em": metric["em"],
                    "acc": metric["acc"],
                    "f1": metric["f1"],
                    "math_equal": metric["math_equal"],
                    "llm_equal": metric["llm_equal"],
                    "m1m2": metric["m1m2"],
                    "tool_productivity": metric["tool_productivity"],
                    "tool_call": metric["tool_call"],
                    "average_datas_used_tool_number": metric["average_datas_used_tool_number"],
                    "num_valid_answer": metric["num_valid_answer"],
                }
                new_metrics.append(new_metric)
with open("/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation//new_metrics.json", "w") as f:
    json.dump(new_metrics, f, indent=4)
import pandas as pd
df = pd.read_json("/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/new_metrics.json")
df.to_csv("/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/new_metrics.csv", index=False)