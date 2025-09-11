import os
import json

result_dir= "/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/result"
# dataset_name = "hle"
new_metrics = []

# For budget_no_limit_run
for dataset_name in os.listdir(result_dir):
    dataset_path = os.path.join(result_dir, dataset_name)
    for root,dirs,files in os.walk(dataset_path):
        if "budget_no_limit" not in root:
            continue
        if "budget_no_limit_run_all_wiki" in root:
            continue
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
                    "llm_equal": metric.get("llm_equal",0),
                    "m1m2": metric.get("m1m2",0),
                    "tool_productivity": metric["tool_productivity"],
                    "tool_call": metric["tool_call"],
                    "tool_budget": metric["tool_budget"],
                    "num_multi_tool": metric["num_multi_tool"],
                    "average_datas_used_tool_number": metric["average_datas_used_tool_number"],
                    "num_valid_answer": metric["num_valid_answer"],
                }
                new_metrics.append(new_metric)

with open("/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/budget_no_limit_run_new_metrics.json", "w") as f:
    json.dump(new_metrics, f, indent=4)
import pandas as pd
df = pd.read_json("/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/budget_no_limit_run_new_metrics.json")
df.to_csv("/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/budget_no_limit_run_new_metrics.csv", index=False)


# For no-tool-inference
for dataset_name in os.listdir(result_dir):
    dataset_path = os.path.join(result_dir, dataset_name)
    for root,dirs,files in os.walk(dataset_path):
        if "no-tool-inference" not in root:
            continue
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
                    "llm_equal": metric.get("llm_equal",0),
                    "m1m2": metric.get("m1m2",0),
                    "tool_productivity": metric["tool_productivity"],
                    "tool_call": metric["tool_call"],
                    "tool_budget": metric["tool_budget"],
                    "num_multi_tool": metric["num_multi_tool"],
                    "average_datas_used_tool_number": metric["average_datas_used_tool_number"],
                    "num_valid_answer": metric["num_valid_answer"],
                }
                new_metrics.append(new_metric)

with open("/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/no-tool-inference_metrics.json", "w") as f:
    json.dump(new_metrics, f, indent=4)
import pandas as pd
df = pd.read_json("/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/no-tool-inference_metrics.json")
df.to_csv("/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/no-tool-inference_metrics.json.csv", index=False)



# For budget_limit_run
new_metrics = []
budget2modeldict = {"5":{}, "10":{}, "15":{}, "20":{}}
budgets = [(4, 3),
(7, 5),
(3, 4),
(3, 7),
(5, 4),
(4, 6),
(5, 7),
(7, 6),
(5, 6),
(6, 3)]
for dataset_name in os.listdir(result_dir):
    dataset_path = os.path.join(result_dir, dataset_name)
    for root,dirs,files in os.walk(dataset_path):
        if "budget_limit_run" not in root:
            continue
        for file in files:
            if "result.metrics.overall" in file:
                with open(os.path.join(root, file), "r") as f:
                    metric = json.load(f)
                print(os.path.join(root, file))
                print(os.path.join(root, file).split("/")[-3])
                print(os.path.join(root, file).split("/")[-2])
                new_metric= {
                    "inference_mode": "budget_limit_run",
                    "inference_budget": os.path.join(root, file).split("/")[-4].split("_")[-1],
                    "inference_budget_set_idx": os.path.join(root, file).split("/")[-3].split("_")[-1],
                    "model_name": os.path.join(root, file).split("/")[-2],
                    "dataset_name": dataset_name,
                    "em": metric["em"],
                    "acc": metric["acc"],
                    "f1": metric["f1"],
                    "math_equal": metric["math_equal"],
                    "llm_equal": metric.get("llm_equal",0),
                    "m1m2": metric.get("m1m2",0),
                    "tool_productivity": metric["tool_productivity"],
                    "tool_call": metric["tool_call"],
                    "tool_budget": metric["tool_budget"],
                    # "python_budget": budgets[int(os.path.join(root, file).split("/")[-3].split("_")[-1])][0],
                    # "search_budget": budgets[int(os.path.join(root, file).split("/")[-3].split("_")[-1])][1],
                    "num_multi_tool": metric["num_multi_tool"],
                    "average_datas_used_tool_number": metric["average_datas_used_tool_number"],
                    "num_valid_answer": metric["num_valid_answer"],
                }
                new_metrics.append(new_metric)
                if  new_metric["model_name"] not in budget2modeldict[new_metric["inference_budget"]]:
                    budget2modeldict[new_metric["inference_budget"]][new_metric["model_name"]] = []
                budget2modeldict[new_metric["inference_budget"]][new_metric["model_name"]].append(new_metric)
                
for budget in budget2modeldict.keys():
    for model in budget2modeldict[budget].keys():
        dataset2avgmetric = {}
        for metric in budget2modeldict[budget][model]:
            if metric["dataset_name"] not in dataset2avgmetric:
                dataset2avgmetric[metric["dataset_name"]] = {}
                for key in metric.keys():
                    if not isinstance(metric[key], str):
                        dataset2avgmetric[metric["dataset_name"]][key] = []
            for key in metric.keys():
                if not isinstance(metric[key], str):
                    dataset2avgmetric[metric["dataset_name"]][key].append(metric[key])
        for dataset in dataset2avgmetric.keys():
            num_budget_idx = 0
            for key in dataset2avgmetric[dataset].keys():
                num_budget_idx = len(dataset2avgmetric[dataset][key])
                dataset2avgmetric[dataset][key] = sum(dataset2avgmetric[dataset][key]) / len(dataset2avgmetric[dataset][key])
            new_metric = dataset2avgmetric[dataset].copy()
            new_metric["model_name"] = model
            new_metric["inference_mode"] = "budget_limit_run"
            new_metric["inference_budget"] = budget
            new_metric["dataset_name"] = dataset
            new_metric["inference_budget_set_idx"] = f"avg_of_{num_budget_idx}_with_budget_{budget}"
            new_metrics.append(new_metric)
         
            
            
with open("/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/budget_limit_run_new_metrics.json", "w") as f:
    json.dump(new_metrics, f, indent=4)
import pandas as pd
df = pd.read_json("/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/budget_limit_run_new_metrics.json")
df.to_csv("/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/budget_limit_run_new_metrics.csv", index=False)


# # new_metrics 是 list[dict]
# df = pd.DataFrame(new_metrics)

# # 保存成 CSV
# df.to_csv(
#     "/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/budget_limit_run_new_metrics.csv",
#     index=False
# )